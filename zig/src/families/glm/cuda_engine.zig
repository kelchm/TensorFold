//! One TP rank's GLM-5.3-Flash CUDA engine (the Python family's decode.Engine); all ranks draw the same tokens.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const sampler = @import("cuda_sampler.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const heat = @import("heat");
const modelopt = @import("core").modelopt;
const exl3 = @import("cuda_exl3.zig");
const images_mod = @import("cuda_images.zig");
const tile = @import("cuda_tile.zig");
const mm_mod = @import("cuda_mm.zig");
const dflash = @import("dflash.zig");
const prof_mod = @import("cuda_prof.zig");
const pages_mod = @import("cuda_pages.zig");
const l2pf_mod = @import("cuda_l2pf.zig");
const prompt_pair = @import("cuda_prompt_pair.zig");
const tp = @import("tp");
const held_shares = @import("cuda_engine_shares.zig");
const mesh_mod = @import("mesh");
pub const Options = @import("cuda_engine_types.zig").Options;

/// The most drafts a chain holds (decode.DepthPolicy's top depth).
pub const max_drafts = 15;
/// A greedy pick as it crosses ranks: (value, id) and padding, so the RoCE gather (16-byte packs) can carry it.
const pair_bytes = 16;
const pick_rows = @import("cuda_engine_pick.zig").pick_rows;

/// A prompt pass's chunk rows (decode.prefill's).
pub const prompt_rows = fw.prefill_rows;
pub const solo_rows = fw.solo_rows;
pub const max_together = @import("cuda_state.zig").prompt_heads;
pub const Prompt = struct { slot: u32, ids: []const u32 };

/// A shared verify's max rows and streams: under hc_pre's 64-row block change; 16 streams take 2 rows each.
pub const shared_rows = types.shared_rows;
pub const max_shared = 16;

const types = @import("cuda_engine_types.zig");
const snapshots = @import("cuda_snapshots.zig");
const prompt_solo = @import("cuda_prompt_solo.zig");
pub const prompt_cuts = @import("cuda_prompt_cuts.zig"); // cuda_lanes' resume check too
pub const Part = types.Part;
pub const Kept = types.Kept;
const Seq = types.Seq;
const Snap = types.Snap;
pub const Resume = types.Resume;
pub const Cancel = types.Cancel;
pub const Marks = types.Marks;

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Config,
    share: shard.Share,
    spans: []shard.Span, // every rank's vocabulary span
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    group: ?*const cuda.tp_group.Group = null, // the main stream's all-gathers (RoCE on two ranks); null: lib and comm
    verify_tiles: @import("cuda_verify_tiles.zig").Options = .{},
    scheduled_l2: bool = false,
    mesh: mesh_mod.Mesh, // this rank's place: one tensor group of `world` ranks
    backend: *tp.group.Backend, // the boundary's communicator over lib and comm, then the group once it is up
    collective: tp.Collective, // backend's, made once so a const engine hands it out
    images: ?*images_mod.Images = null, // --vision: a prompt pass's image rows (rank 0 encodes, the others receive)
    packs: mm_mod.Packed = .{}, // the linears packed at load and their bits (TF_GLM_DENSE_BITS, _HEAD_BITS)
    drafter: ?*dflash.Drafter = null, // --drafter: DFlash2 (every rank runs its passes; rank 0 selects the drafts)
    profs: [fw.window_rows]?*prof_mod.Prof = @splat(null), // TF_GLM_PROF=1: verify sections' GPU time, by rows
    shared_prof: ?*prof_mod.Prof = null, // TF_GLM_PROF=1: shared verify sections' GPU time
    profiling_prompt: ?*prof_mod.Prof = null,
    prompt_profs: [4]?*prof_mod.Prof = @splat(null), // and prompt chunks': to 64, 256, 1024, 2048 rows
    checkpoint_digest: [32]u8,
    settings: ranks.Settings,
    m: fw.Modules,
    stream: cuda.Stream,
    pieces: fw.Pieces,
    w: weights.Weights,
    bp: fw.Buffers, // prompt chunks
    pair: ?prompt_pair.Pair = null, // TF_GLM_PROMPT_PAIR: a paired prompt step's second chunk
    bd: fw.Buffers, // verify windows
    bm: fw.Buffers, // the MTP head's rows
    bs: ?fw.Buffers = null, // shared verify windows (with the MTP head, as the lane core shares only then)
    pairs: cuda.DeviceBuffer, // every rank's (value, id) of a greedy draw
    smp: sampler.Sampler,
    seqs: std.ArrayList(?*Seq) = .empty, // every stream's sequence by slot (the lane backend's), slot 0 made at init
    cur: *Seq = undefined, // the bound slot's: every step below runs on it
    slot: u32 = 0,
    capacity: usize,
    mtp: bool,
    heat_gate: heat.Gate, // upstream's bands (off unless set), read before each prompt chunk on the gathered maximum
    shared_buf: cuda.DeviceBuffer, // one f64 every rank all-reduces to its maximum
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty, // kept prompt states by id (the same ids on every rank)
    snap_pages: ?snapshots.pages.Pool = null,
    snap_setup: ?snapshots.Setup = null,
    next_snap: u64 = 1, // rank 0: the next id
    announce: u64 = 0, // rank 0: a state kept at this chunk's mark, for the other ranks to copy at its gate
    dropped: std.ArrayList(u32) = .empty, // rank 0: ids dropped since the other ranks last heard (cuda_follow)
    prompt_logprobs: ?prompt_lp.Want = null, // a prompt pass's rows asked for (every rank runs the head on them)
    prompt_bufs: ?prompt_lp.Buffers = null, // their head passes' buffers, made at the first pass that asks
    prompt_tried: bool = false, // that pass came: without buffers after it, the passes take the window's
    prompt_out: prompt_lp.Output, // a head pass's logprob words, then its picks, host-mapped (rank 0 reads them)
    structured: bool = false, // a structured output step: every rank's draws gather whole rows, rank 0 masks them
    pages: ?*pages_mod.Pages = null, // the other streams' KV pages and sequences (cuda_pages), once the server attached them
    index_split: bool = false, // the next prompt step's indexer rows split across the ranks: rank 0's choice, the chunk's flag 2
    cap_rows: u32 = 0, // full GLM-5.3: a capped prompt step's rows (cuda_prompt_cuts.cappedRows), this step's alone
    index_splits: u64 = 0, // prompt steps whose indexer rows split (glm-split-check counts them)
    share_pages: bool = false, // TF_KV_SHARE: a kept state leaves its prefix's pages to the shared store that holds them
    keep_seqs: bool = false, // TF_GLM_KEEP_SEQS: a released slot keeps its sequence for its next request
    l2pf: ?l2pf_mod.L2pf = null, // TF_GLM_L2PF: the prefetch's side stream, kernels and table, made after the weights tile
    side: ?@import("cuda_moe.zig").Side = null, // TF_GLM_SHARED_SIDE: decode shared-expert stream (not L2PF's)
    local: []held_shares.Local = &.{}, // full GLM-5.3 at TP=1: the other shares this process runs (cuda_engine_shares)
    rank_buf: [held_shares.max_held]fw.Rank = undefined, // those shares as the forward's ranks

    /// Opens rank `rank` of `world` over `comm`; the ranks agree on settings (cuda_ranks) before the weights load.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: Options) !*Engine {
        if (world > ranks.max_world) return error.TooManyRanks; // before anything is allocated
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24));
        const is_tfq3 = @import("cuda_tfq3.zig").isTfq3(gpa, io, checkpoint);
        var why: modelopt.Why = .{};
        if (!is_tfq3) (if (exl3.isExl3(gpa, text)) exl3.admit(gpa, text, &why) else @import("cuda_formats.zig").admit(gpa, text, &why)) catch |err| {
            std.log.err("{s}: {s}", .{ cfg_path, why.text() });
            return err;
        };
        var c = try @import("config_dsa.zig").of(gpa, text);
        if (is_tfq3) c.head_unit = 128;
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        const config_hash = try ranks.hashFile(a, io, cfg_path);
        const program_hash = try ranks.hashFile(a, io, "/proc/self/exe");
        const checkpoint_digest = try ranks.artifact(a, io, checkpoint);
        var fp = ranks.Fingerprint.of(world, rank, o.settings, o.capacity, o.mtp, config_hash, program_hash, checkpoint_digest);
        if (o.vision) |lim| ranks.mediaFingerprint(&fp, lim);
        fp.mtp += @intFromBool(o.drafter);
        if (o.drafter) ranks.drafterFingerprint(&fp, o.drafter_bits, if (o.drafter_dir) |dd| try ranks.artifact(a, io, dd) else @splat(0));
        try ranks.agree(d, lib, comm, stream, fp, try a.alloc(ranks.Fingerprint, world));
        try ranks.probeStart(d, lib, comm, stream, o.settings, rank, world);
        const place = mesh_mod.Mesh.init(.{ .tensor = std.math.cast(u32, world) orelse 0 }, std.math.cast(u32, rank) orelse std.math.maxInt(u32)) catch return error.BadRank;
        const tg = place.group(.tensor);
        const held = try held_shares.held(&c, tg.rank, tg.world);
        const share = try shard.Share.of(&c, held.first, held.total);
        const backend = try gpa.create(tp.group.Backend);
        errdefer gpa.destroy(backend);
        backend.* = .{ .d = d, .lib = lib, .comm = comm, .rank = tg.rank, .world = tg.world, .owner = o.owner };
        const kb = @import("cuda_kda_block.zig");
        kb.tile_pad = o.settings.kda_pad; // its common width for glm_mm, else this rank's own rows to 16 bytes
        kb.pad_to = if (kb.tile_pad) kb.padWidth(c.kda_heads, world) else kb.alignedWidth(share.kda.len());
        const spans = try gpa.alloc(shard.Span, held.total); // every share's vocabulary rows
        errdefer gpa.free(spans);
        for (spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, held.total)).vocab;
        const block32 = if (is_tfq3 or exl3.isExl3(gpa, text)) null else try @import("cuda_experts.zig").wantsBlock32(gpa, text);
        var m = try fw.Modules.load(d, ctx, block32, c.rope != 0, is_tfq3);
        errdefer m.deinit();
        const pair = prompt_pair.wanted(o.settings.pair, o.settings.solo > fw.prefill_rows, o.settings.pair_solo, world);
        const joint_experts = pair and o.settings.pair_experts;
        var pieces = try fw.Pieces.init(d, o.settings.pieces, pair);
        errdefer pieces.deinit();
        if (o.settings.layers > c.layers) return error.BadLayers;
        const layers = try a.alloc(u32, if (o.settings.layers > 0) o.settings.layers else c.layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .gpa = gpa, .io = io, .heat_gate = .{ .bands = o.settings.heat, .root = @import("cuda_engine_prof.zig").heatRoot() }, .shared_buf = undefined, .d = d, .c = c, .share = share, .spans = spans, .lib = lib, .comm = comm, .mesh = place, .backend = backend, .collective = backend.collective(), .checkpoint_digest = checkpoint_digest, .settings = o.settings, .m = m, .stream = stream, .pieces = pieces, .w = undefined, .capacity = o.capacity, .bp = undefined, .bd = undefined, .bm = undefined, .pairs = undefined, .smp = undefined, .prompt_out = undefined, .mtp = o.mtp and !o.settings.dflash_only };
        e.keep_seqs = o.settings.keep_seqs; // release keeps a slot's sequence with TF_GLM_KEEP_SEQS=1
        const has_packs = o.settings.dense_bits != 16 or o.settings.head_bits != 16;
        if (has_packs and !o.settings.tiled) return error.PackedNeedsGlmMm;
        if (held.count > 1) try held_shares.refused(&o, o.drafter, o.vision != null or has_packs);
        const tiler: tile.Tiler = .{ .gpa = gpa, .mk = &e.m.mm, .s = stream, .d = d, .enabled = o.settings.tiled, .dense = o.settings.dense_bits, .head = o.settings.head_bits, .packs = &e.packs };
        const draft_weights: weights.DraftWeights = if (o.settings.dflash_only) .external else if (o.mtp) .mtp else .none;
        e.w = try @import("cuda_load_owner.zig").finish(try weights.loadFor(gpa, io, d, stream, e.m.tools(), checkpoint, c, share, layers, draft_weights), tiler, tile.Tiler.apply);
        errdefer e.w.deinit();
        errdefer {
            gpa.free(e.packs.ptrs);
            gpa.free(e.packs.bits);
        }
        @import("cuda_ranks_explain.zig").logNumerics(o.settings.numerics, rank);
        e.m.mm.prompt_fixed_k = o.settings.prompt_fixed_k;
        e.m.tiled = o.settings.tiled;
        if (o.settings.tiled) e.m.packs = &e.packs;
        const cap = try ctx.capability();
        if (o.settings.dflash_q4_down) try e.m.q4.prepareWide(e.m.mods[5]);
        if ((o.settings.prompt_s2 or e.m.mm.fast4) and has_packs and cap == 121) try e.m.mm.preparePrompt2(e.m.mods[7]);
        if (o.settings.exl3_decode and world == 2 and cap == 121) try e.m.exl3.prepareGlmDecode(o.settings.exl3_nmajor);
        if (o.settings.verify_tiles.enabled()) e.verify_tiles = o.settings.verify_tiles.gate(cap, world, o.settings.pages, o.settings.kv_host);
        if (held.count == 1 and o.settings.l2pf.want(cap, world)) e.l2pf = try l2pf_mod.L2pf.init(gpa, d, &e.w, o.settings.l2pf);
        errdefer if (e.l2pf) |*p| p.deinit();
        e.scheduled_l2 = e.l2pf != null and world == 2 and cap == 121;
        if (e.settings.pages and pages_mod.pageBytes(e) == 0) e.settings.pages = false;
        try @import("cuda_solo_memory.zig").check(e, ctx, o);
        if (e.settings.pages) try pages_mod.ownSequence(e) else try e.bind(0); // on pages: no window, its pool places it
        errdefer {
            e.free(0);
            e.seqs.deinit(gpa);
        }
        e.bp = if (joint_experts) try fw.Buffers.initPairExperts(d, &e.w, o.settings.solo, o.capacity) else try fw.Buffers.init(d, &e.w, o.settings.solo, true, o.capacity);
        errdefer e.bp.deinit();
        if (o.settings.solo > fw.prefill_rows and rank == 0) std.log.info("glm: prompt passes of {d} rows, {d} while another stream decodes", .{ o.settings.solo, fw.prefill_rows });
        if (world > 1 and rank == 0) std.log.info("glm: a prompt chunk's exchanges in {d} row pieces", .{o.settings.pieces});
        if (pair) e.pair = try prompt_pair.Pair.init(d, &e.w, &e.bp, &e.pieces, o.capacity, joint_experts);
        errdefer if (e.pair) |*p| p.deinit();
        if (pair and o.settings.pair_solo and rank == 0) std.log.info("glm: paired prompt steps of two {d}-row chunks: {d:.2} GiB for the second's buffers, index rings of {d} rows", .{ e.bp.rows, @as(f64, @floatFromInt(e.pair.?.bp.arena.buf.len)) / (1 << 30), e.cur.st.ring_rows });
        e.bd = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        if (!o.settings.dflash_only) e.bm = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer if (!o.settings.dflash_only) e.bm.deinit();
        if (o.settings.dflash_only and rank == 0) std.log.info("glm: DFlash2 uses its draft head without loading the checkpoint MTP layer or its KV pages", .{});
        if (o.mtp) e.bs = try fw.Buffers.initShared(d, &e.w, shared_rows, o.capacity);
        errdefer if (e.bs) |*b| b.deinit();
        e.pairs = try cuda.DeviceBuffer.alloc(d, pair_bytes * held.total * pick_rows);
        errdefer e.pairs.free();
        try d.check(d.api.cuMemsetD8_v2(e.pairs.ptr, 0, pair_bytes * held.total * pick_rows), "cuMemsetD8"); // each pair's padding stays zero
        try held_shares.load(e, held, .{ .tiler = tiler, .checkpoint = checkpoint, .layers = layers, .mtp = e.mtp });
        errdefer held_shares.deinit(e);
        try held_shares.fill(e);
        e.smp = try sampler.Sampler.init(d, c.vocab, fw.window_rows);
        errdefer e.smp.deinit();
        errdefer e.freeProfs();
        try @import("cuda_engine_prof.zig").initProfs(e, layers.len);
        e.prompt_out = try prompt_lp.Output.init(d);
        errdefer e.prompt_out.free();
        e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
        errdefer e.shared_buf.free();
        if (o.settings.shared_side) e.side = try @import("cuda_moe.zig").Side.init(d);
        errdefer if (e.side) |*x| x.deinit(); // drained before the buffers and weights its work reads go
        errdefer @import("cuda_load_owner.zig").drain(stream, d);
        try e.stream.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.side) |*x| x.deinit(); // drained: shared-expert work in flight reads buffers and weights freed below
        held_shares.deinit(e);
        e.gpa.free(e.packs.ptrs);
        e.gpa.free(e.packs.bits);
        if (e.drafter) |dr| dr.deinit();
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        if (e.snap_pages) |*p| p.deinit();
        e.dropped.deinit(e.gpa);
        e.shared_buf.free();
        e.freeProfs();
        if (e.prompt_bufs) |*b| b.free();
        e.prompt_out.free();
        e.smp.deinit();
        e.pairs.free();
        if (e.bs) |*b| b.deinit();
        if (!e.settings.dflash_only) e.bm.deinit();
        e.bd.deinit();
        if (e.pair) |*p| p.deinit();
        e.bp.deinit();
        if (e.pages) |p| p.deinit();
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        if (e.l2pf) |*p| p.deinit();
        e.w.deinit();
        e.pieces.deinit();
        e.m.deinit();
        e.stream.deinit();
        e.gpa.free(e.spans);
        e.gpa.destroy(e.backend);
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    const freeProfs = @import("cuda_engine_prof.zig").freeProfs;

    /// The serving group, its RoCE wires open: the main stream's and the boundary's gathers take its route now.
    pub fn installGroup(e: *Engine, g: *cuda.tp_group.Group, from: []const u8) void {
        e.group = g;
        e.backend.group = g;
        std.log.info("[tensorfold] glm: rank {d} of {d} ({s}) sums through the {t} collective (RoCE gathers {s})", .{ g.rank, g.world, from, e.collective.kind(), if (g.fast != null or g.multi != null) "on" else "off" });
    }

    pub fn exchange(e: *const Engine) fw.Exchange {
        if (e.local.len > 0) return .local; // the held shares meet in this process
        return .{ .nccl = .{ .lib = e.lib, .comm = e.comm, .collective = e.collective, .group = e.group, .fuse = e.settings.roce_fuse } };
    }

    pub fn view(e: *Engine, b: *fw.Buffers, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        const im: ?fw.ImageRows = if (e.images) |v| .{ .v = v, .slot = e.slot } else null;
        return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = if (e.local.len > 0) held_shares.ranks(e, b) else out, .exchange = e.exchange(), .pieces = prompt_pair.piecesOf(e, b), .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .shared_kda_wide = e.settings.shared_kda_wide, .hc_fuse = e.settings.hc_fuse, .numerics = e.settings.numerics, .images = im, .taps = if (e.drafter) |dr| dr.taps(b == &e.bp) else null, .l2pf = if (e.l2pf) |*p| p else null, .side = e.side, .pair_experts = e.pair != null and e.settings.pair_experts };
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it (a zeroed state) on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        const made = e.seqs.items[slot] == null;
        const previous = if (e.images != null) e.cur else null;
        const previous_slot = if (e.images != null) e.slot else 0;
        errdefer {
            if (made) e.free(slot);
            if (previous) |q| {
                e.cur = q;
                e.slot = previous_slot;
            }
        }
        if (made) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = try held_shares.seq(e);
            e.seqs.items[slot] = q;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
        if (e.images) |v| try v.activate(slot);
    }

    /// Frees slot `slot`'s sequence unless the slot keeps it (a paged one gives its pages back) and binds slot 0.
    pub fn release(e: *Engine, slot: u32) void {
        if (e.images) |v| v.release(slot);
        const paged = if (e.pages) |p| p.paged(slot) else false;
        if (paged) e.pages.?.close(slot) else if (!keepsSequence(e, slot)) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    /// Whether a released slot keeps its sequence: slot 0 always, every slot with TF_GLM_KEEP_SEQS.
    pub fn keepsSequence(e: *const Engine, slot: u32) bool {
        return slot == 0 or e.keep_seqs;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        cuda.completion.drain(e.d, e.stream, "GLM sequence release did not drain");
        q.st.deinit();
        held_shares.freeStates(e.gpa, q.more);
        q.deinitMeta(e.gpa);
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }

    /// Bytes one more stream's sequence takes on this rank: its state arena, and at TP=1 the held shares' too.
    pub fn seqBytes(e: *const Engine) u64 {
        return held_shares.seqBytes(e.seqs.items[0].?);
    }

    /// The committed length: the pending token sits at this position, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }

    /// A follower's begin: rank 0's next request; no image rows from an earlier one splice into it.
    pub fn beginFollowing(e: *Engine, sampling: ?sampler.Sampling) !void {
        if (e.images) |v| v.clear();
        return e.begin(sampling);
    }

    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        e.cur.resetMeta(); // a request without starts cuts on the grid alone
        try held_shares.reset(e);
        if (e.drafter) |dr| try dr.restart(e.slot, 0);
    }

    /// decode.prefill: `ids` chunk by chunk to each cut, then the first token from the last row; `from` resumes.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var start = try e.prefillStart(ids.len, from);
        while (true) {
            const end = prompt_cuts.nextCut(ids.len, e.cur.cuts.items, start);
            if (try e.prefillChunk(ids[start..@min(ids.len, end + 1)], start, ids.len, null, cancel, marks)) |first| return first;
            start = end;
        }
    }

    /// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + fw.window_rows >= e.cur.st.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len or !prompt_cuts.coldCut(len, e.cur.cuts.items, r.at)) return error.BadResume;
        if ((e.snaps.get(r.id) orelse return error.NoKeptState).pos != r.at) return error.BadResume;
        try e.restoreState(r.id);
        return r.at;
    }

    /// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
        const split = e.index_split; // this step's alone
        e.index_split = false;
        const capped = e.cap_rows;
        e.cap_rows = 0;
        if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
        const paired = capped == 0 and if (e.pair) |p| part.len > p.rows + 1 else false; // a pair holds > p.rows + next id
        const rows = if (capped == 0) try prompt_cuts.stepRows(len, e.cur.cuts.items, start, part.len, if (paired) 2 * e.pair.?.rows else e.bp.rows) else if (e.c.dsa) try prompt_cuts.cappedRows(len, e.cur.cuts.items, start, capped) else return error.BadChunk;
        if (paired) return prompt_pair.prefill(e, part, start, len, cancel, marks, split);
        if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
        if (rows > fw.prefill_rows and !prompt_solo.allowed(e, start, len, rows)) return error.BadChunk;
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bp, &one);
        f.index_split = split;
        if (start + rows <= len -| @as(usize, @intCast(dflash.WIN))) f.taps = null; // none read (dflash.promptRun)
        const as_i32: []const i32 = @ptrCast(part);
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        f.prof = e.prompt_profs[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3];
        const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 2048 rows" };
        if (f.prof) |p| try p.beginPrompt(e.stream, labels[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3], .{ .rank = e.share.rank, .slot = e.slot, .start = start, .rows = rows }, e.c.run, e.pieces.count);
        if (f.prof) |p| e.profiling_prompt = p;
        defer if (f.prof != null) {
            e.profiling_prompt = null;
        };
        defer if (f.prof) |p| p.end();
        errdefer if (f.prof) |p| {
            p.prompt.?.invalid = true;
        };
        const gate_started: i96 = if (f.prof != null) std.Io.Clock.awake.now(e.io).toNanoseconds() else 0;
        const heat_before = if (f.prof != null) e.heat_gate.waited_s else 0;
        var stamp = prof_mod.span(f.prof, e.stream, .gate);
        e.gate(cancel) catch |err| {
            if (err == error.Cancelled) try snapshots.interrupted(e, if (prompt_cuts.coldCut(len, e.cur.cuts.items, start)) marks else null, @intCast(start)); // a resume reads cold cuts alone
            return err;
        };
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .gate);
        if (f.prof != null) std.log.info("glm prof gate rank={d} slot={d} start={d} rows={d} wall_ms={d:.4} heat_wait_s={d:.3}", .{ e.share.rank, e.slot, start, rows, prof_mod.Prof.elapsedMs(e.io, gate_started), e.heat_gate.waited_s - heat_before });
        const last = start + rows == len;
        try f.runMode(as_i32[0..rows], last);
        e.index_splits += @intFromBool(split);
        stamp = prof_mod.span(f.prof, e.stream, .prompt_score);
        if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
        stamp.end();
        stamp = prof_mod.span(f.prof, e.stream, .drafter);
        if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len);
        stamp.end();
        stamp = prof_mod.span(f.prof, e.stream, .draw);
        const draw_started: i96 = if (f.prof != null) std.Io.Clock.awake.now(e.io).toNanoseconds() else 0;
        var first: ?u32 = null;
        if (last) {
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, 0, &t);
            first = t[0];
        }
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .draw);
        if (last and f.prof != null) std.log.info("glm prof draw rank={d} slot={d} start={d} rows={d} wall_ms={d:.4}", .{ e.share.rank, e.slot, start, rows, prof_mod.Prof.elapsedMs(e.io, draw_started) });
        if (last) try e.holdPrompt(e.bp.fnormed + (rows - 1) * e.c.hidden * 2);
        stamp = prof_mod.span(f.prof, e.stream, .mtp);
        if (e.mtp) try prompt_solo.mtpRows(e, &f, as_i32);
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .mtp);
        stamp = prof_mod.span(f.prof, e.stream, .commit);
        try f.commit(rows);
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .commit);
        return first;
    }

    pub const gate = @import("cuda_engine_gate.zig").gate;
    pub const shared = @import("cuda_engine_gate.zig").shared;

    pub const stateBytes = snapshots.bytes;
    pub const prepareSnapshots = snapshots.prepare;
    pub const keepState = snapshots.keep;
    pub const keepAs = snapshots.keepAs;
    pub const restoreState = snapshots.restore;
    pub const holds = snapshots.holds;
    pub const dropState = snapshots.drop;
    const freeSnap = snapshots.free;

    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
        if (ids.len == 0 or ids.len > fw.window_rows) return error.WindowTooWide;
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bd, &one);
        const at = e.cur.st.pos;
        f.scheduled_l2 = e.scheduled_l2;
        f.verify_tiles = e.verify_tiles.window(at, ids.len);
        f.prof = e.profs[ids.len - 1];
        if (f.prof) |p| try p.begin(e.stream, verifyLabel(ids.len));
        try f.run(@ptrCast(ids));
        if (e.drafter) |dr| try dr.noteVerify(e.slot, 0, ids.len, at, true);
        e.cur.rows_at = e.bd.fnormed;
        try e.draw(e.bd.logits, ids.len, at + 1, bits, out);
        if (f.prof) |p| p.end();
    }

    const verifyLabel = @import("cuda_engine_prof.zig").verifyLabel;

    /// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        const b = if (e.bs) |*x| x else return error.NoSharedWindows;
        if (e.local.len > 0) return error.SharesOneRankEach;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .shared_kda_wide = e.settings.shared_kda_wide, .hc_fuse = e.settings.hc_fuse, .numerics = e.settings.numerics, .taps = if (e.drafter) |dr| dr.taps(false) else null, .l2pf = if (e.l2pf) |*p| p else null, .side = e.side, .prof = e.shared_prof };
        if (f.prof) |p| try p.begin(e.stream, "verify shared");
        errdefer if (f.prof) |p| {
            p.open = false;
        };
        try f.run(@ptrCast(ids[0..rows]));
        if (e.drafter) |dr| for (parts, segs[0..parts.len], 0..) |p, g, i| try dr.noteVerify(p.slot, g.row0, g.rows, g.st.pos, i == 0);
        try @import("cuda_engine_pick.zig").drawShared(e, b, parts, segs[0..parts.len], rows);
        if (f.prof) |p| p.end();
    }

    /// DFlash2's round (every rank, the same `reqs`): each stream's kept rows into its context, then the block pass.
    pub fn draftBlock(e: *Engine, reqs: []dflash.Req) !void {
        const dr = e.drafter orelse return error.NoDrafter;
        for (reqs) |*q| q.pos = @intCast((e.seqs.items[q.slot] orelse return error.UnknownSlot).st.pos);
        try dr.draft(e, reqs);
    }

    pub const holdPrompt = @import("cuda_prompt_hold.zig").save;
    pub const prefillMany = @import("cuda_prompts.zig").prefillMany;

    /// forward.commit of the last window's first `n` rows (the KDA replay when fewer than it ran).
    pub fn keep(e: *Engine, n: usize) !void {
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        try f.commit(n);
    }

    /// decode.draft: the MTP head absorbs the kept rows (committed first) with `follow`, then chains `depth` drafts.
    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32, chances: ?[]f64) !void {
        if (!e.mtp) return error.NoDraftHead;
        if (depth > max_drafts) return error.TooDeep;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bm, &one);
        const D = e.c.hidden;
        const hidden = if (from_prompt) @import("cuda_prompt_hold.zig").input(e) else e.cur.rows_at;
        try held_shares.setMtpLen(e, e.cur.st.mtp_len - e.cur.in_head);
        e.cur.in_head = 0;
        const next: []const i32 = @ptrCast(follow);
        var s0: usize = 0;
        while (s0 < next.len) : (s0 += fw.window_rows) {
            const part = next[s0..@min(next.len, s0 + fw.window_rows)];
            try f.mtpMode(part, &.{hidden + s0 * D * 2}, depth > 0);
            try held_shares.setMtpLen(e, e.cur.st.mtp_len + part.len);
        }
        for (0..depth) |j| {
            if (chances != null and !e.structured and e.cur.sampling == null and e.local.len == 0) {
                const p = try e.greedyChance(e.bm.logits);
                out[j] = p.token;
                chances.?[j] = p.chance;
            } else {
                try e.draw(e.bm.logits, 1, e.cur.st.pos + 1 + j, 0, out[j..][0..1]);
                if (chances) |c| c[j] = 1; // a sampled or structured draft keeps chance 1: its window runs whole
            }
            if (j + 1 < depth) {
                try f.mtp(@ptrCast(out[j..][0..1]), &.{e.bm.fnormed});
                try held_shares.setMtpLen(e, e.cur.st.mtp_len + 1);
                e.cur.in_head += 1;
            }
        }
    }

    /// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
    pub fn draw(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
        if (e.structured) return e.drawWhole(logits, rows, first, bits, out);
        if (e.local.len > 0) return held_shares.draw(e, logits, rows, first, out);
        if (e.cur.sampling) |s| {
            try e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
            return e.checkGathers(); // the draw read rows the forward's gathers made
        }
        try @import("cuda_engine_pick.zig").greedyRows(e, logits, rows, out);
    }

    pub const promptHead = @import("cuda_structured.zig").promptHead;
    pub const drawWhole = @import("cuda_structured.zig").drawWhole;
    pub const redrawFirst = @import("cuda_structured.zig").redrawFirst;
    /// decode.sample_rows' greedy pick: per-rank top-1 (tf_glm_top1), all-gathered, largest value then lowest id.
    const greedy = @import("cuda_engine_pick.zig").greedy;
    const greedyChance = @import("cuda_engine_pick.zig").greedyChance;

    /// A failed RoCE gather (its rows hold nothing) or TF_GLM_KV_HOST gather past its slots, after a synchronize.
    pub fn checkGathers(e: *const Engine) !void {
        if (e.group) |g| try g.check();
        try @import("cuda_kv_host.zig").checkActive(); // a TF_GLM_KV_HOST gather past its slots
    }

    pub fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};
