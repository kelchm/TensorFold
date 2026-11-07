//! Production GLM rank startup and resources, shared by serving and developer sessions.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const images_mod = @import("cuda_images.zig");
const glm_lanes = @import("cuda_lanes.zig");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const dflash = @import("dflash.zig");
const dflash_load = @import("dflash_load.zig");
const pages_mod = @import("cuda_pages.zig");
const kv = @import("core").kv_pool;

const native = @import("cuda_native.zig");
const Options = native.Options;
const Place = native.Place;
const Peer = native.Peer;
pub const Check = struct { ctx: *anyopaque, run: *const fn (*anyopaque, *const cuda.tp_group.Group, *const cuda.Context) anyerror!void };

pub const Rank = struct {
    gpa: std.mem.Allocator,
    group: cuda.tp_group.Group, // the ranks' communicator; two ranks add RoCE writes for the main stream's small gathers
    e: *engine.Engine,
    link: follow.Link,
    lanes: glm_lanes.Lanes,
    layout: ?kv.Layout,
    peer: ?Peer = null,
};

const env = @import("sys").env;

/// The send/recv the prompt exchange will use, connected now, before the weights fill memory (connected on first use,
fn warmPeers(d: *const cuda.Driver, g: *const cuda.tp_group.Group, set: ranks.Settings) !void {
    if (!set.split) return; // gather: all-gathers only, which NCCL's rings connect at the first collective
    const bytes = 4 << 20; // a prompt piece's exchange: every channel a large send/recv takes
    var buf = try cuda.DeviceBuffer.alloc(d, 2 * bytes * g.world);
    defer buf.free();
    var s = try cuda.Stream.init(d, false);
    defer s.deinit();
    const next = (g.rank + 1) % g.world;
    const prev = (g.rank + g.world - 1) % g.world;
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        for (0..g.world) |peer| if (peer != g.rank) {
            if (!set.ring or peer == next) g.send(s, buf.ptr + peer * bytes, bytes, peer) catch |err| break :post err;
            if (!set.ring or peer == prev) g.recv(s, buf.ptr + (g.world + peer) * bytes, bytes, peer) catch |err| break :post err;
        };
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    try s.synchronize();
}

const startup = @import("cuda_start.zig");

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, check: ?Check) !*Rank {
    const p = try native.place();
    const roce_bytes = try native.roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null);
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    var problem: []const u8 = "";
    var set = ranks.Settings.read(&ranks.environment, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    set.kv8 = o.kv8;
    return startup.open(io, p, if (std.c.getenv("TF_KV_PAGES")) |v| std.mem.span(v) else null, set, cuda.tp_watch, StartArgs{ .gpa = gpa, .io = io, .ctx = ctx, .dir = dir, .o = o, .p = p, .roce_bytes = roce_bytes, .check = check }, openRank);
}

const StartArgs = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: *const cuda.Context,
    dir: []const u8,
    o: Options,
    p: Place,
    roce_bytes: usize,
    check: ?Check,
};

/// After liveness and page settings: NCCL, the engine and the link, then a follower's loop or rank zero's lanes.
fn openRank(a: StartArgs, set: ranks.Settings) !*Rank {
    const gpa = a.gpa;
    const io = a.io;
    const ctx = a.ctx;
    const dir = a.dir;
    const o = a.o;
    const p = a.p;
    const roce_bytes = a.roce_bytes;
    cuda.memory_receipt.emit(@intCast(p.rank), .load_begin, true); // the four phases tools/memory_receipts.py reads
    const own = try gpa.create(Rank);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    // every connection registered at init, while memory is free (lazily, a full GB10's ibv_reg_mr can refuse it)
    if (std.c.getenv("NCCL_RUNTIME_CONNECT") == null) _ = env.setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    // NCCL's cuMem host buffers wore out GB10's driver pool under load (NV_ERR_NO_MEMORY; next ranks failed to start)
    if (std.c.getenv("NCCL_CUMEM_HOST_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_HOST_ENABLE", "0", 0);
    if (std.c.getenv("NCCL_CUMEM_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_ENABLE", "0", 0);
    // NCCL_COMM_ID holds NCCL's bootstrap to its subnet past NCCL_SOCKET_IFNAME: a follower naming one lets it pick
    if (p.rank > 0 and std.c.getenv("NCCL_SOCKET_IFNAME") != null) _ = env.unsetenv("NCCL_COMM_ID");
    var diag = cuda.p2p_report.begin(p.rank, p.world);
    defer diag.end();
    var lib = try cuda.nccl.Library.open();
    errdefer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (p.rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // the root at NCCL_COMM_ID
        cuda.tp_watch.sendId(&id.internal);
    } else id.internal = try cuda.tp_watch.takeId(io); // rank 0's, over this rank's line
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    own.group = .{ .lib = lib, .comm = comm, .rank = p.rank, .world = p.world };
    errdefer _ = own.group.lib.api.ncclCommDestroy(own.group.comm);
    if (p.world > 1) try warmPeers(ctx.d, &own.group, set);
    cuda.p2p_report.afterInit(&diag, .{ .lib = &own.group.lib, .comm = comm, .d = ctx.d, .later = true });
    if (a.check) |c| try c.run(c.ctx, &own.group, ctx);
    const cap = native.capacity(o.context);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.group.lib, own.group.comm, p.rank, p.world, .{ .capacity = cap, .drafts = o.drafts, .settings = set, .vision = o.vision, .drafter = o.drafter != null });
    errdefer own.e.deinit();
    if (o.vision) |lim| { // before the rooms: the tower's bytes count against rank 0's
        var problem: []const u8 = "";
        const ql: cuda.qlinear.Linear = .{ .lane = &own.e.m.lane, .nvfp4_prompt = &own.e.m.prompt }; // the tower's MXFP8, NVFP4
        own.e.images = (try images_mod.Images.init(gpa, io, ctx.d, own.e.stream, dir, lim, fw.prefill_rows + 1, p.rank == 0, ql, &problem)) orelse {
            std.log.err("{s}", .{problem});
            return error.NoVisionTower;
        };
    }
    return @import("cuda_open_owner.zig").finish(own.e, OpenArgs{
        .own = own,
        .io = io,
        .ctx = ctx,
        .o = o,
        .p = p,
        .cap = cap,
        .roce_bytes = roce_bytes,
    }, finishRank);
}

const OpenArgs = struct {
    own: *Rank,
    io: std.Io,
    ctx: *const cuda.Context,
    o: Options,
    p: Place,
    cap: usize,
    roce_bytes: usize,
};

fn finishRank(a: OpenArgs) !*Rank {
    const own = a.own;
    const set = own.e.settings;
    const gpa = own.gpa;
    const io = a.io;
    const ctx = a.ctx;
    const o = a.o;
    const p = a.p;
    const roce_bytes = a.roce_bytes;
    const cap = a.cap;
    if (o.drafter) |dd| { // before the rooms: its weights and scratch count against every rank's
        if (!o.drafts) return error.DrafterNeedsDrafts;
        own.e.drafter = try dflash_load.load(gpa, io, own.e, dd, p.rank, p.world, o.drafter_bits);
    }
    try own.group.openRoce(gpa, io, ctx.d, own.e.stream, roce_bytes); // both ranks bring it up or neither does
    errdefer if (own.group.fast) |f| f.deinit();
    try @import("cuda_p2p.zig").open(gpa, ctx.d, &own.group, own.e.stream, own.e.m.glue.p2p_gather); // both ranks, or neither
    errdefer if (own.group.p2p) |x| {
        x.deinit();
        own.group.p2p = null;
    };
    own.e.ce = try @import("cuda_ce.zig").open(gpa, ctx.d, &own.group, own.e.stream, &own.e.bp, own.e.c.hidden, set.split and !set.ring);
    errdefer if (own.e.ce) |x| {
        x.deinit();
        own.e.ce = null;
    };
    own.e.group = &own.group;
    own.link = try follow.Link.init(gpa, ctx.d, &own.group.lib, own.group.comm, own.e.stream, try native.linkCapacity(cap, o.vision));
    errdefer own.link.deinit();
    if (p.rank == 0 and p.world > 1) own.e.link = &own.link; // its disk steps (cuda_disk_ranks) cross on the link
    // Rank 0 admits by the smallest room of the ranks; its sizes are the largest (the first ranks take the remainder).
    const rooms = try gpa.alloc(u64, p.world);
    defer gpa.free(rooms);
    const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    try ranks.gather(u64, ctx.d, &own.group.lib, own.group.comm, own.e.stream, p.rank, mine, rooms);
    var peer: ?Peer = null;
    for (rooms[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
        peer = .{ .bytes = r, .rank = q };
    };
    // every rank loaded and agreed (cuda_ranks) before the first step; the digest names what they agreed on
    if (p.world > 1) try cuda.tp_watch.ready(io, own.e.checkpoint_digest);
    cuda.memory_receipt.emit(@intCast(p.rank), .load_end, false);
    own.peer = peer;
    own.layout = null;
    own.link.asleep = p.rank == 0 and cuda.tp_watch.open();
    return own;
}
/// What a rank owns, freed by rank 0's close and by a follower before it leaves.
pub fn release(own: *Rank) void {
    own.link.deinit();
    if (own.e.ce) |x| x.deinit(); // before the buffers it maps go with the engine
    if (own.e.images) |v| v.deinit();
    own.e.deinit();
    own.group.close();
    own.gpa.destroy(own);
}
