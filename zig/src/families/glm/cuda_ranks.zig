//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.

const std = @import("std");
const cuda = @import("cuda");
const heat = @import("heat");
const l2pf = @import("cuda_l2pf.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    kv8: bool = false, // --kv-cache fp8, supplied by the native options rather than an environment override
    pages: bool = false, // TF_KV_PAGES as `open` resolves it (with the world and --kv-cache): every sequence on pages
    prompt_fixed_k: bool = false, // TF_GLM_PROMPT_K: one (default) or shape (experimental fixed K slices)
    prompt_s2: bool = false, // TF_GLM_PACKED_PROMPT_S2=1: two-stage 4/6/8-bit packed prompt projections
    tiled: bool = true, // TF_GLM_MM: 1 (the bf16 linears on glm_mm, tiled at load) or 0
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows), ring or gather
    ring: bool = false, // TF_GLM_PREFILL_REDUCE=ring: split, its exchanges passed along the ring (cuda_relay)
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    shared_kda_wide: bool = true, // TF_GLM_SHARED_KDA_WIDE: shared verify KDA chains (unset: inherits kda_wide)
    kda_pad: bool = false, // TF_GLM_KDA_PAD: 1 (an uneven fused KDA input, padded at load, tiled for glm_mm) or 0
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole); unset, 8 under solo on the wire.
    solo: bool = false, // TF_GLM_CHUNK_SOLO: 4096 (prompt buffers for two chunks a pass, while no stream decodes) or 2048
    keep_seqs: bool = false, // TF_GLM_KEEP_SEQS: 1 (a released slot keeps its sequence for its next request)
    pair: bool = true, // TF_GLM_PROMPT_PAIR: 1 (while no stream decodes, a prompt step takes two chunks, each one's exchanges under the other's sites) or 0
    pair_dflash: bool = false, // TF_GLM_PAIR_DFLASH=1: opt into paired prompt steps beside DFlash2
    exl3_decode: bool = false, // TF_GLM_EXL3_DECODE=1: specialized uniform 4-bit TP2 routed experts for short verify windows
    exl3_nmajor: bool = false, // TF_GLM_EXL3_N_MAJOR=1: output-major CTA traversal at rows 4 and 8..32
    pair_experts: bool = false, // TF_GLM_PAIR_EXPERTS=1: paired chunks share 128-pair NVFP4 routed-expert items (0: two 64-pair passes)
    index_split: u32 = 2048, // TF_GLM_INDEX_SPLIT: from this position on, rank 0 splits a prompt chunk's indexer rows (0: never)
    heat: ?heat.Bands = null, // TF_HEAT_HIGH and TF_HEAT_LOW (upstream's names): a prompt chunk waits on every rank's hottest zone
    layers: u32 = 0, // TF_GLM_LAYERS: the first this many layers only, a thin model for one-GPU checks (0: all)
    dense_bits: u8 = 16, // TF_GLM_DENSE_BITS: the dense linears packed at load to 4, 6 or 8 bits (16: bf16)
    head_bits: u8 = 16, // TF_GLM_HEAD_BITS: the head, likewise
    roce_multi: bool = false, // TF_GLM_ROCE_MULTI: 1 (three ranks or more: small all-gathers on fabric/tp's multirank RoCE beside NCCL) or 0
    roce_fuse: bool = false, // TF_GLM_ROCE_FUSE=1: TP2 whole-window gather + HC post or MTP residual in one kernel
    roce_wide: bool = false, // TF_ROCE=1: at three ranks or more the RoCE gather and the drafter MLP split (two ranks: always)
    kv_host: u8 = 0, // TF_GLM_KV_HOST=1: KV pages in host memory (1); with TF_GLM_KV_HOST_SHARED=1, one copy a host (2)
    kv_host_gib: f64 = 0, // TF_GLM_KV_HOST_GIB: the host pages' GiB (0: one full window)
    l2pf: l2pf.Options = .{}, // TF_GLM_L2PF and its knobs: a decode window's next weights read into L2 beside it
    dflash_only: bool = false, // TF_GLM_DFLASH_ONLY=1: omit checkpoint MTP layer and its KV pages beside DFlash2
    dflash_q4_down: bool = false, // TF_GLM_DFLASH_Q4_DOWN=1: quantize DFlash2 MLP down projections to Q4 (fp32 partials)
    dflash_snap_context: bool = false, // TF_GLM_DFLASH_SNAP_CONTEXT=1: keep up to 640 initialized DFlash2 KV rows in snapshots
    verify_tiles: @import("cuda_verify_tiles.zig").Options = .{},
    kv_live_half: bool = false, // TF_GLM_KV_LIVE_HALF=1: a keep carries the live KDA half alone (cuda_snapshot_layout.zig)
    draft_split: bool = false, // TF_GLM_DRAFT_SPLIT=1: three ranks or more split the drafter MLP without TF_ROCE=1
    shared_side: bool = true, // TF_GLM_SHARED_SIDE: decode window shared expert on its own stream; same bytes, ranks may differ
    hc_fuse: bool = false, // TF_GLM_HC_FUSE: prompt hc_post writes the next hc_pre's partials; same bytes, ranks may differ

    /// Whether three ranks or more split the DFlash2 drafter's MLP (dflash.mlpSplit's `wide`); two ranks always do.
    pub fn draftWide(s: Settings) bool {
        return s.roce_wide or s.draft_split;
    }

    /// The settings `get` gives, blank text as unset; `problem` names a value that is not one of the choices.
    pub fn read(comptime get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        return readSet(&struct {
            fn set(name: [:0]const u8) ?[]const u8 {
                const v = get(name) orelse return null;
                return if (std.mem.trim(u8, v, " \t\r\n").len == 0) null else v; // environment.parse's blank
            }
        }.set, problem);
    }

    /// `read` from a lookup with no blank text, so every name below and in l2pf's reader is unset when blank.
    fn readSet(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        var s: Settings = .{};
        if (get("TF_GLM_PROMPT_K")) |v| {
            if (std.mem.eql(u8, v, "shape")) s.prompt_fixed_k = true else if (!std.mem.eql(u8, v, "one")) {
                problem.* = "TF_GLM_PROMPT_K: one or shape";
                return error.BadSetting;
            }
        }
        // TF_GLM_PROMPT_TAIL: rank 0's prompt cache reads it (cuda_snaps' cacheRules); a typo is refused on every rank
        if (get("TF_GLM_PROMPT_TAIL")) |v| {
            if (!std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "1")) {
                problem.* = "TF_GLM_PROMPT_TAIL: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_MM")) |v| {
            if (std.mem.eql(u8, v, "0")) s.tiled = false else if (!std.mem.eql(u8, v, "1")) {
                problem.* = "TF_GLM_MM: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KDA_PAD")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kda_pad = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KDA_PAD: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PREFILL_REDUCE")) |v| {
            if (std.mem.eql(u8, v, "split")) s.split = true else if (std.mem.eql(u8, v, "ring")) {
                s.split = true;
                s.ring = true;
            } else if (std.mem.eql(u8, v, "gather")) s.split = false else {
                problem.* = "TF_GLM_PREFILL_REDUCE: split, ring or gather";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KDA_DECODE_WIDE")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kda_wide = true else if (std.mem.eql(u8, v, "0")) s.kda_wide = false else {
                problem.* = "TF_GLM_KDA_DECODE_WIDE: 0 or 1";
                return error.BadSetting;
            }
        }
        s.shared_kda_wide = @import("sys").environment.value(bool, "TF_GLM_SHARED_KDA_WIDE", get("TF_GLM_SHARED_KDA_WIDE"), s.kda_wide) catch {
            problem.* = "TF_GLM_SHARED_KDA_WIDE: 0 or 1";
            return error.BadSetting;
        };
        if (get("TF_GLM_ROCE_MULTI")) |v| {
            if (std.mem.eql(u8, v, "1")) s.roce_multi = true else if (std.mem.eql(u8, v, "0")) s.roce_multi = false else {
                problem.* = "TF_GLM_ROCE_MULTI: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_ROCE")) |v| s.roce_wide = std.mem.eql(u8, v, "1");
        if (get("TF_GLM_CHUNK_SOLO")) |v| {
            if (std.mem.eql(u8, v, "4096")) s.solo = true else if (std.mem.eql(u8, v, "2048")) s.solo = false else {
                problem.* = "TF_GLM_CHUNK_SOLO: 2048 or 4096";
                return error.BadSetting;
            }
        }
        s.l2pf = try l2pf.Options.read(get, problem);
        if (get("TF_GLM_PROMPT_PAIR")) |v| {
            if (std.mem.eql(u8, v, "1")) s.pair = true else if (std.mem.eql(u8, v, "0")) s.pair = false else {
                problem.* = "TF_GLM_PROMPT_PAIR: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KEEP_SEQS")) |v| {
            if (std.mem.eql(u8, v, "1")) s.keep_seqs = true else if (std.mem.eql(u8, v, "0")) s.keep_seqs = false else {
                problem.* = "TF_GLM_KEEP_SEQS: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PAIR_EXPERTS")) |v| {
            if (std.mem.eql(u8, v, "1")) s.pair_experts = true else if (std.mem.eql(u8, v, "0")) s.pair_experts = false else {
                problem.* = "TF_GLM_PAIR_EXPERTS: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_OVERLAP_PIECES")) |v| {
            s.pieces = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.pieces < 1 or s.pieces > 8) {
                problem.* = "TF_GLM_OVERLAP_PIECES: 1 to 8";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_INDEX_SPLIT")) |v| {
            s.index_split = std.fmt.parseInt(u32, v, 10) catch {
                problem.* = "TF_GLM_INDEX_SPLIT: the position a prompt chunk's indexer rows split from, or 0 for never";
                return error.BadSetting;
            };
        }
        if (get("TF_GLM_LAYERS")) |v| {
            s.layers = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.layers == 0) {
                problem.* = "TF_GLM_LAYERS: the first how many layers, 1 or more";
                return error.BadSetting;
            }
        }
        inline for (.{ .{ "TF_GLM_DENSE_BITS", "dense_bits" }, .{ "TF_GLM_HEAD_BITS", "head_bits" } }) |e| if (get(e[0])) |v| {
            const b = std.fmt.parseInt(u8, v, 10) catch {
                problem.* = e[0] ++ ": 4, 6, 8 or 16";
                return error.BadSetting;
            };
            @field(s, e[1]) = switch (b) {
                4, 6, 8, 16 => b,
                0 => 16, // the old family's spelling of bf16
                else => {
                    problem.* = e[0] ++ ": 4, 6, 8 or 16";
                    return error.BadSetting;
                },
            };
        };
        s.heat = heat.bandsFrom(get("TF_HEAT_HIGH"), get("TF_HEAT_LOW")) catch {
            problem.* = "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high";
            return error.BadSetting;
        };
        if (s.prompt_fixed_k and !s.tiled) {
            problem.* = "TF_GLM_PROMPT_K=shape requires TF_GLM_MM=1";
            return error.BadSetting;
        }
        if (get("TF_GLM_KV_HOST")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kv_host = 1 else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KV_HOST: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KV_HOST_SHARED")) |v| {
            if (std.mem.eql(u8, v, "1") and s.kv_host == 1) s.kv_host = 2 else if (!std.mem.eql(u8, v, "0")) {
                problem.* = if (std.mem.eql(u8, v, "1")) "TF_GLM_KV_HOST_SHARED=1 requires TF_GLM_KV_HOST=1" else "TF_GLM_KV_HOST_SHARED: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KV_HOST_GIB")) |v| {
            s.kv_host_gib = std.fmt.parseFloat(f64, v) catch 0;
            if (!(s.kv_host_gib > 0) or s.kv_host == 0) {
                problem.* = "TF_GLM_KV_HOST_GIB: the host pages' GiB, more than 0, with TF_GLM_KV_HOST=1";
                return error.BadSetting;
            }
        }
        inline for (.{ .{ "TF_GLM_DECODE_MLA_WIDE", "mla" }, .{ "TF_GLM_DECODE_TILE_RANGES", "ranges" } }) |flag| {
            @field(s.verify_tiles, flag[1]) = @import("sys").environment.value(bool, flag[0], get(flag[0]), false) catch {
                problem.* = flag[0] ++ ": 0 or 1";
                return error.BadSetting;
            };
        }
        inline for (.{ .{ "TF_GLM_DFLASH_ONLY", "dflash_only" }, .{ "TF_GLM_DFLASH_Q4_DOWN", "dflash_q4_down" }, .{ "TF_GLM_EXL3_DECODE", "exl3_decode" }, .{ "TF_GLM_EXL3_N_MAJOR", "exl3_nmajor" }, .{ "TF_GLM_PACKED_PROMPT_S2", "prompt_s2" }, .{ "TF_GLM_DFLASH_SNAP_CONTEXT", "dflash_snap_context" }, .{ "TF_GLM_PAIR_DFLASH", "pair_dflash" }, .{ "TF_GLM_ROCE_FUSE", "roce_fuse" }, .{ "TF_GLM_HC_FUSE", "hc_fuse" } }) |flag| {
            @field(s, flag[1]) = @import("sys").environment.value(bool, flag[0], get(flag[0]), false) catch {
                problem.* = flag[0] ++ ": 0 or 1";
                return error.BadSetting;
            };
        }
        s.shared_side = @import("sys").environment.value(bool, "TF_GLM_SHARED_SIDE", get("TF_GLM_SHARED_SIDE"), true) catch {
            problem.* = "TF_GLM_SHARED_SIDE: 0 or 1";
            return error.BadSetting;
        };
        if (get("TF_GLM_DRAFT_SPLIT")) |v| {
            if (std.mem.eql(u8, v, "1")) s.draft_split = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_DRAFT_SPLIT: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KV_LIVE_HALF")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kv_live_half = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KV_LIVE_HALF: 0 or 1";
                return error.BadSetting;
            }
        }
        // refused here, before the weights load: the wire's bootstrap meets at rank 0, its kernel comes from the image
        if (s.roce_multi and (get("TF_TP_MASTER") == null or get("TF_TP_KERNEL_IMAGE") == null)) {
            problem.* = "TF_GLM_ROCE_MULTI=1 requires TF_TP_MASTER and TF_TP_KERNEL_IMAGE";
            return error.BadSetting;
        }
        if (s.kda_pad and !s.tiled) {
            problem.* = "TF_GLM_KDA_PAD=1 requires TF_GLM_MM=1 (glm_mm is what reads the padded rows)";
            return error.BadSetting;
        }
        return s;
    }
};

/// The most ranks a world holds: rank 0's greedy pick gathers every rank's top-1 into one host array of this size.
pub const max_world = 16;

/// A setting by name (null when unset): a registered switch through the profile registry, the rest as before.
pub fn environment(name: [:0]const u8) ?[]const u8 {
    return @import("profile").lookup(name);
}

/// Whether the group brings up its own RoCE path: not where TF_GLM_ROCE_MULTI=1 lends fabric/tp's wire (3+ ranks).
pub fn ownRoce(s: Settings, world: usize) bool {
    return !(s.roce_multi and world >= 3);
}

/// TF_GLM_CHUNK_SOLO with TF_GLM_ROCE_MULTI at three ranks or more (the wire opens; TP=2 stays on NCCL): 8 pieces.
pub fn wirePieces(s: *Settings, world: usize, comptime get: *const fn ([:0]const u8) ?[]const u8) void {
    const set = if (get("TF_GLM_OVERLAP_PIECES")) |v| std.mem.trim(u8, v, " \t\r\n").len > 0 else false; // blank: unset, as `read`
    if (!set and s.solo and s.roce_multi and world >= 3) s.pieces = 8;
}

/// The tools' form (tf-cuda-test): the world from TF_GLM_WORLD, their input as it always was, not their own arguments.
pub fn toolPieces(s: *Settings, comptime get: *const fn ([:0]const u8) ?[]const u8) void {
    const v = get("TF_GLM_WORLD") orelse return;
    wirePieces(s, std.fmt.parseInt(usize, v, 10) catch return, get);
}

/// What every rank must agree on, as bytes the communicator carries.
pub const Fingerprint = extern struct {
    layout: u32 = 33, // 32, plus shared_kda_wide bit in kda_wide
    world: u32,
    rank: u32,
    kv8: u32 = 0,
    pages: u32 = 0, // TF_KV_PAGES: the engine's own sequence is made on pages at init, so every rank pages alike
    prompt_fixed_k: u32,
    tiled: u32,
    split: u32, // TF_GLM_PREFILL_REDUCE: 0 gather, 1 split, 2 ring
    kda_wide: u32,
    kda_pad: u32 = 0, // TF_GLM_KDA_PAD: every rank's KDA input rows carry one width
    kv_live_half: u32 = 0, // TF_GLM_KV_LIVE_HALF: every rank keeps the live KDA half alone (1) or the head whole (0)
    pieces: u32,
    pair: u32, // TF_GLM_PROMPT_PAIR: rank 0's paired steps need the second chunk's buffers on every rank
    pair_experts: u32 = 0, // TF_GLM_PAIR_EXPERTS: every rank sizes the joint plan and twin scratch alike
    layers: u32, // TF_GLM_LAYERS (0: all)
    mtp: u32, // 1: the MTP head loads and runs (drafts on); 2: and DFlash2 (--drafter), whose rounds gather
    vision: u32 = 0, // --vision's image tokens (0: off): rank 0 sends image rows every rank must hold
    video_tokens: u32 = 0,
    max_images: u32 = 0,
    max_videos: u32 = 0,
    max_frames: u32 = 0,
    roce_multi: u32 = 0, // TF_GLM_ROCE_MULTI: every rank opens the borrowed wire or none does (a lone rank would wait on it)
    roce_wide: u32 = 0, // TF_ROCE=1: three ranks or more split the drafter MLP and gather by RoCE, or none does
    draft_split: u32 = 0, // TF_GLM_DRAFT_SPLIT=1: three ranks or more split the drafter MLP, or none does
    kv_host: u32 = 0, // TF_GLM_KV_HOST(_SHARED): every rank sizes the pool alike and maps the shared file, or none does
    solo: u32 = 0, // TF_GLM_CHUNK_SOLO: rank 0's solo passes need the wider prompt buffers on every rank
    keep_seqs: u32 = 0, // TF_GLM_KEEP_SEQS: every rank keeps a released slot's sequence alike
    l2pf: [4]u32 = @splat(0), // TF_GLM_L2PF's words: every rank prefetches alike, so none waits on a slower one's windows
    bits: u32 = 16 << 8 | 16, // TF_GLM_DENSE_BITS << 8 | TF_GLM_HEAD_BITS: every rank's matmuls read the same codes
    capacity: u64, // tokens a sequence holds
    drafter_bits: u64 = 0, // --drafter-bits (0: no drafter, 16: bf16, 4: q4)
    roce_fuse: u64 = 0, // TF_GLM_ROCE_FUSE: whole-window posts fuse only where every rank enabled TP2 RoCE fusion
    heat_high: f64, // the heat bands in degrees (0 and 0: off): every rank must wait alike
    heat_low: f64,
    kv_host_gib: f64 = 0, // TF_GLM_KV_HOST_GIB: each rank's row marks span the host pages rank 0 counts
    config: [32]u8, // sha256 of config.json
    program: [32]u8, // sha256 of this executable
    checkpoint: [64]u8, // hex digest of the checkpoint's metadata and every shard byte, through the digest cache
    drafter_checkpoint: [64]u8 = @splat(0), // hex digest of --drafter's metadata and shard bytes (0s: none)

    pub const Field = enum {
        layout,
        rank,
        world,
        keep_seqs,
        kv8,
        pages,
        prompt_fixed_k,
        tiled,
        split,
        kda_wide,
        kda_pad,
        kv_live_half,
        pieces,
        pair,
        pair_experts,
        layers,
        mtp,
        vision,
        video_tokens,
        max_images,
        max_videos,
        max_frames,
        roce_multi,
        roce_wide,
        draft_split,
        kv_host,
        solo,
        l2pf,
        bits,
        capacity,
        drafter_bits,
        roce_fuse,
        heat_high,
        heat_low,
        kv_host_gib,
        config,
        program,
        checkpoint,
        drafter_checkpoint,
    };

    pub fn of(
        world: usize,
        rank: usize,
        s: Settings,
        capacity: usize,
        mtp: bool,
        config: [32]u8,
        program: [32]u8,
        checkpoint_digest: [32]u8,
    ) Fingerprint {
        const f: Fingerprint = .{
            .kv8 = @intFromBool(s.kv8),
            .pages = @intFromBool(s.pages),
            .world = @intCast(world),
            .rank = @intCast(rank),
            .prompt_fixed_k = @intFromBool(s.prompt_fixed_k),
            .tiled = @as(u32, @intFromBool(s.tiled)) | (@as(u32, @intFromBool(s.verify_tiles.mla)) << 1) | (@as(u32, @intFromBool(s.verify_tiles.ranges)) << 2),
            .split = if (!s.split) 0 else if (s.ring) 2 else 1,
            .kda_wide = @as(u32, @intFromBool(s.kda_wide)) | (@as(u32, @intFromBool(s.shared_kda_wide)) << 1),
            .kda_pad = @intFromBool(s.kda_pad),
            .kv_live_half = @intFromBool(s.kv_live_half),
            .pieces = s.pieces,
            .pair = @as(u32, @intFromBool(s.pair)) | (@as(u32, @intFromBool(s.pair_dflash)) << 1),
            .pair_experts = @as(u32, @intFromBool(s.pair_experts)) | (@as(u32, @intFromBool(s.exl3_decode)) << 1) | (@as(u32, @intFromBool(s.exl3_nmajor)) << 2),
            .layers = s.layers,
            .mtp = @as(u32, @intFromBool(mtp)) | (@as(u32, @intFromBool(s.dflash_only)) << 2) | (@as(u32, @intFromBool(s.dflash_q4_down)) << 3) | (@as(u32, @intFromBool(s.dflash_snap_context)) << 4),
            .roce_multi = @intFromBool(s.roce_multi),
            .roce_fuse = @intFromBool(s.roce_fuse),
            .roce_wide = @intFromBool(s.roce_wide and world >= 3), // two ranks ask by default, so a lone TF_ROCE=1 there is no difference
            .draft_split = @intFromBool(s.draft_split and world >= 3), // two ranks split whatever it says
            .kv_host = s.kv_host,
            .kv_host_gib = s.kv_host_gib,
            .solo = @intFromBool(s.solo),
            .keep_seqs = @intFromBool(s.keep_seqs),
            .l2pf = s.l2pf.words(),
            .bits = (@as(u32, @intFromBool(s.prompt_s2)) << 16) | (@as(u32, s.dense_bits) << 8) | s.head_bits,
            .capacity = capacity,
            .heat_high = if (s.heat) |b| b.high_c else 0,
            .heat_low = if (s.heat) |b| b.low_c else 0,
            .config = config,
            .program = program,
            .checkpoint = std.fmt.bytesToHex(checkpoint_digest, .lower),
        };
        return f;
    }
};

pub const Mismatch = struct { field: Fingerprint.Field, rank: usize };

/// The first field where rank `r`'s fingerprint differs from rank 0's (or holds the wrong rank), null when all agree.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    for (all, 0..) |f, r| {
        if (f.layout != all[0].layout) return .{ .field = .layout, .rank = r };
        if (f.rank != r) return .{ .field = .rank, .rank = r };
        const fields = .{
            "kv8",
            "pages",
            "prompt_fixed_k",
            "world",
            "tiled",
            "split",
            "kda_wide",
            "kda_pad",
            "kv_live_half",
            "pieces",
            "pair",
            "pair_experts",
            "layers",
            "mtp",
            "vision",
            "video_tokens",
            "max_images",
            "max_videos",
            "max_frames",
            "roce_multi",
            "roce_wide",
            "draft_split",
            "kv_host",
            "solo",
            "keep_seqs",
            "l2pf",
            "bits",
            "capacity",
            "drafter_bits",
            "roce_fuse",
            "heat_high",
            "heat_low",
            "kv_host_gib",
            "config",
            "program",
            "checkpoint",
            "drafter_checkpoint",
        };
        inline for (fields) |name| {
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(f, name)), std.mem.asBytes(&@field(all[0], name))))
                return .{ .field = @field(Fingerprint.Field, name), .rank = r };
        }
    }
    return null;
}

/// sha256 of a file's bytes (read whole: config.json, the executable).
pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return hashBytes(bytes);
}

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return out;
}

/// Every rank's `mine` in rank order, over one ncclAllGather.
pub fn gather(
    comptime T: type,
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    rank: usize,
    mine: T,
    out: []T,
) !void {
    const size = @sizeOf(T);
    var dev = try cuda.DeviceBuffer.alloc(d, size * out.len);
    defer dev.free();
    try dev.uploadAsync(rank * size, std.mem.asBytes(&mine), stream.handle);
    const gathered = lib.api.ncclAllGather(dev.ptr + rank * size, dev.ptr, size, .u8, comm, stream.handle);
    try lib.check(gathered, "ncclAllGather");
    try stream.synchronize();
    try dev.download(0, std.mem.sliceAsBytes(out));
}

/// All-gathers `mine` over `comm`; any disagreement is error.RankMismatch on every rank, printing what differs.
pub fn agree(
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    mine: Fingerprint,
    out: []Fingerprint,
) !void {
    try gather(Fingerprint, d, lib, comm, stream, mine.rank, mine, out);
    if (mismatch(out)) |m| {
        std.debug.print("REFUSE rank settings: rank {d}'s {s} differs from rank 0's ({s})\n", .{ m.rank, @tagName(m.field), explain(m.field) });
        return error.RankMismatch;
    }
}

/// Before loading: direct split reaches every rank, so each pair is probed; the ring only passes rows to neighbours.
pub fn probeStart(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, s: Settings, rank: usize, world: usize) !void {
    if (s.split and !s.ring) try probeSplit(d, lib, comm, stream, rank, world);
}

/// Before loading, split's peers: one value each way with each, in rank order (no pair waits on a later one).
fn probeSplit(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, rank: usize, world: usize) !void {
    var dev = try cuda.DeviceBuffer.alloc(d, 8);
    defer dev.free();
    for (0..world) |peer| if (peer != rank) {
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        try lib.check(lib.api.ncclSend(dev.ptr, 1, .f32, @intCast(peer), comm, stream.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(dev.ptr + 4, 1, .f32, @intCast(peer), comm, stream.handle), "ncclRecv");
        lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd") catch return unreachablePeer(rank, peer);
        stream.synchronize() catch return unreachablePeer(rank, peer);
    };
}

/// The latest pair split refused, which a test build keeps for its test instead of printing the refusal.
pub var refused_pair: ?[2]usize = null;

fn unreachablePeer(rank: usize, peer: usize) error{SplitUnreachable} {
    if (@import("builtin").is_test) {
        refused_pair = .{ rank, peer };
        return error.SplitUnreachable;
    }
    std.debug.print("REFUSE TF_GLM_PREFILL_REDUCE=split: rank {d} cannot exchange with rank {d}; split sends every rank's rows to every other, and a fabric where some ranks reach only their neighbours (a ring of four without a switch) needs TF_GLM_PREFILL_REDUCE=ring or gather\n", .{ rank, peer });
    return error.SplitUnreachable;
}

fn explain(f: Fingerprint.Field) []const u8 {
    return switch (f) {
        .layout => "a different build of this check",
        .rank => "a rank number given twice or out of order",
        .world => "the ranks were started with different world sizes",
        .keep_seqs => "TF_GLM_KEEP_SEQS",
        .kv8 => "--kv-cache fp8",
        .pages => "TF_KV_PAGES (unset or auto pages from two ranks)",
        .prompt_fixed_k => "TF_GLM_PROMPT_K",
        .tiled => "TF_GLM_MM, TF_GLM_DECODE_MLA_WIDE or TF_GLM_DECODE_TILE_RANGES",
        .split => "TF_GLM_PREFILL_REDUCE",
        .kda_wide => "TF_GLM_KDA_DECODE_WIDE or TF_GLM_SHARED_KDA_WIDE",
        .kda_pad => "TF_GLM_KDA_PAD",
        .kv_live_half => "TF_GLM_KV_LIVE_HALF",
        .pieces => "TF_GLM_OVERLAP_PIECES",
        .pair => "TF_GLM_PROMPT_PAIR or TF_GLM_PAIR_DFLASH",
        .pair_experts => "TF_GLM_PAIR_EXPERTS / TF_GLM_EXL3_DECODE / TF_GLM_EXL3_N_MAJOR",
        .layers => "TF_GLM_LAYERS",
        .mtp => "--no-drafts, --drafter, TF_GLM_DFLASH_ONLY, TF_GLM_DFLASH_Q4_DOWN or TF_GLM_DFLASH_SNAP_CONTEXT differs across ranks",
        .vision => "--vision (or its image tokens) given to some ranks and not others",
        .video_tokens => "--vision-video-tokens",
        .max_images => "the image count limit",
        .max_videos => "the video count limit",
        .max_frames => "the frame limit",
        .roce_multi => "TF_GLM_ROCE_MULTI",
        .roce_fuse => "TF_GLM_ROCE_FUSE",
        .roce_wide => "TF_ROCE=1 at three ranks or more (the RoCE gather and the drafter split)",
        .draft_split => "TF_GLM_DRAFT_SPLIT=1 at three ranks or more (the drafter split)",
        .kv_host => "TF_GLM_KV_HOST or TF_GLM_KV_HOST_SHARED",
        .kv_host_gib => "TF_GLM_KV_HOST_GIB",
        .solo => "TF_GLM_CHUNK_SOLO",
        .l2pf => "TF_GLM_L2PF, _SHARED_FIRST, _MB, _CTAS or _PACE_GBPS",
        .bits => "TF_GLM_DENSE_BITS, TF_GLM_HEAD_BITS or TF_GLM_PACKED_PROMPT_S2",
        .capacity => "the sequence capacity (the context)",
        .drafter_bits => "--drafter-bits",
        .heat_high => "TF_HEAT_HIGH",
        .heat_low => "TF_HEAT_LOW",
        .config => "config.json",
        .program => "the executable",
        .checkpoint => "checkpoint metadata or shard bytes",
        .drafter_checkpoint => "--drafter checkpoint metadata or shard bytes",
    };
}

pub fn mediaFingerprint(f: *Fingerprint, limits: @import("cuda_images.zig").Limits) void {
    f.vision = limits.image_tokens;
    f.video_tokens = limits.video_tokens;
    f.max_images = limits.max_images;
    f.max_videos = limits.max_videos;
    f.max_frames = limits.max_frames;
}

pub fn drafterFingerprint(f: *Fingerprint, bits: u8, digest: [32]u8) void {
    f.drafter_bits = if (bits == 0) 16 else bits;
    f.drafter_checkpoint = std.fmt.bytesToHex(digest, .lower);
}

/// The checkpoint digest a rank sends: its metadata and every shard byte, read again only for a file that changed.
pub fn artifact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    return @import("core").checkpoint_identity.fullIdentity(gpa, io, dir);
}
