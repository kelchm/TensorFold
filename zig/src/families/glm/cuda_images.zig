//! Chunk-scoped media rows on every rank: rank 0 encodes one image/group at a time; all ranks splice the window.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const vision = @import("cuda_vision.zig");
const vcfg = @import("vision_config.zig");
const image_core = @import("image_core.zig");
const image = image_core.For(@import("core").picture);
const video_mod = @import("video_core.zig").For(@import("core").picture);

/// Request admission budgets, independent of the shared active-window scratch allocation.
pub const Limits = struct { image_tokens: u32, max_images: u32, video_tokens: u32 = 0, max_videos: u32 = 0, max_frames: u32 = 0 };

/// Prompt rows [at, at + n) take image rows [row, row + n).
pub const Span = struct { at: u32, n: u32, row: u32 };

// Only request-owned descriptors survive a prompt step. Device rows are shared scratch.
const Saved = struct { images: []const lanes.stream.Image = &.{}, from: usize = 0 };
pub const step_rows: u32 = 2 * @import("cuda_forward.zig").prefill_rows;
pub const row_bytes: usize = vcfg.out_hidden * 2;

pub const Images = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    module: cuda.Module,
    run: vision.Run,
    tower: ?vision.Tower = null, // rank 0's; the other ranks receive the rows
    cfg: ?vcfg.Config = null,
    rows: cuda.DeviceBuffer, // shared bf16 [step_rows + one image/group][out_hidden]
    map: cuda.DeviceBuffer, // i32 [max_rows]: a splice's feature row per prompt row
    host_map: []i32,
    limits: Limits,
    spans: std.ArrayList(Span) = .empty,
    saved: std.AutoHashMapUnmanaged(u32, Saved) = .empty,
    source: Saved = .{},
    slot: u32 = 0, // the sequence the spans belong to: another stream's rows at the same positions take none
    marks: struct { image: u32, open: u32, close: u32 } = .{ .image = 0, .open = 0, .close = 0 },
    mark_ids: [3]u32 = @splat(0), // begin, image, end: text that spells one stays text
    video_ids: ?[3]u32 = null, // begin, video, end: the checkpoint reads videos (each frame pair as an image)

    /// The chat template writes each image's begin and end itself (the server only expands the image token).
    pub const template_wraps = true;
    pub const fixed_image_tokens: u32 = 2048;
    pub const lazy_video = true;
    /// A video's frames to patch pairs (Glm5NextVideoProcessor), over core's decoder: each pair goes in as an image.
    pub const video = video_mod;

    /// Rank 0 (`tower`) or another rank: the rows buffer, and rank 0's tower; null with `problem` when no tower.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, limits: Limits, max_rows: usize, tower: bool, ql: cuda.qlinear.Linear, problem: *[]const u8) !?*Images {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cwd = std.Io.Dir.cwd();
        const cfg_text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
        const proc = cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "processor_config.json" }), a, .limited(1 << 24)) catch null;
        const cfg = (try vcfg.Config.parse(a, cfg_text, proc)) orelse {
            problem.* = "this checkpoint has no vision tower: serve it without --vision";
            return null;
        };
        const v = try gpa.create(Images);
        errdefer gpa.destroy(v);
        var module = try cuda.Module.load(d, cuda.kernels.glm_vision);
        errdefer module.unload();
        v.* = .{ .gpa = gpa, .io = io, .d = d, .module = module, .run = .{ .stream = stream, .f = try vision.Funcs.load(module) }, .cfg = cfg, .rows = undefined, .map = undefined, .host_map = undefined, .limits = limits };
        v.video_ids = cfg.video;
        v.rows = try cuda.DeviceBuffer.alloc(d, v.scratchBytes());
        errdefer v.rows.free();
        v.map = try cuda.DeviceBuffer.alloc(d, max_rows * 4);
        errdefer v.map.free();
        v.host_map = try gpa.alloc(i32, max_rows);
        errdefer gpa.free(v.host_map);
        if (tower) v.tower = try vision.Tower.load(gpa, io, d, dir, ql);
        v.marks = .{ .image = cfg.image_token, .open = cfg.image_start, .close = cfg.image_end };
        v.mark_ids = .{ cfg.image_start, cfg.image_token, cfg.image_end };
        return v;
    }

    // A mixed request holds both budgets; validate the addition before allocating or receiving rows.
    pub fn rowLimit(v: *const Images) !u32 {
        return std.math.add(u32, v.limits.image_tokens, if (v.video_ids != null) v.limits.video_tokens else 0) catch error.ImageRows;
    }

    pub fn deinit(v: *Images) void {
        @import("cuda_load_owner.zig").drain(v.run.stream, v.d);
        v.saved.deinit(v.gpa);
        v.spans.deinit(v.gpa);
        if (v.tower) |*t| t.deinit();
        v.gpa.free(v.host_map);
        v.map.free();
        v.rows.free();
        v.module.unload();
        v.gpa.destroy(v);
    }

    /// Device bytes the tower and the buffers hold.
    pub fn bytes(v: *const Images) u64 {
        return (if (v.tower) |t| t.bytes else 0) + v.rows.len + v.map.len;
    }

    /// One global reservation before pool admission, independent of history length and live stream count.
    /// The second region lets a whole image/group cross a chunk boundary without changing its tower inputs.
    pub fn groupLimit(v: *const Images) u32 {
        return @max(@min(v.limits.image_tokens, fixed_image_tokens), if (v.video_ids != null) @min(v.limits.video_tokens, 16384) else 0);
    }

    pub fn scratchBytes(v: *const Images) usize {
        return (@as(usize, step_rows) + v.groupLimit()) * row_bytes;
    }

    pub fn streamBytes(_: *const Images) usize {
        return 0; // the shared scratch was allocated before the pool/admission budget was measured
    }

    pub fn activate(v: *Images, slot: u32) !void {
        if (v.slot == slot) return;
        // Reserve before changing either owner, including when an idle slot has no source descriptors.
        try v.saved.ensureUnusedCapacity(v.gpa, 1);
        const next = if (v.saved.fetchRemove(slot)) |entry| entry.value else Saved{};
        v.saved.putAssumeCapacity(v.slot, v.source);
        v.finishWindow();
        v.source = next;
        v.slot = slot;
    }

    /// Drop borrowed request descriptors even when a peer-link failure prevents a normal release step.
    pub fn forget(v: *Images, slot: u32) void {
        if (slot == v.slot) v.source = .{} else _ = v.saved.remove(slot);
    }

    pub fn release(v: *Images, slot: u32) void {
        if (slot != v.slot) {
            _ = v.saved.remove(slot);
            return;
        }
        v.finishWindow();
        v.source = .{};
        if (slot != 0) {
            if (v.saved.fetchRemove(0)) |zero| v.source = zero.value;
            v.slot = 0;
        }
    }

    /// Reset the bound request; borrowed descriptors cannot outlive the request arena.
    pub fn clear(v: *Images) void {
        v.finishWindow();
        v.source = .{};
    }

    /// No feature rows survive a prompt step, a failure or a suspension. The shared allocation is retained.
    pub fn finishWindow(v: *Images) void {
        if (v.spans.items.len != 0) @import("cuda_load_owner.zig").drain(v.run.stream, v.d);
        v.spans.clearRetainingCapacity();
    }

    /// Register the request's descriptors only. A cached prefix never materializes here or at a later chunk.
    pub fn begin(v: *Images, slot: u32, images: []const lanes.stream.Image, from: usize) !void {
        try v.activate(slot);
        v.clear();
        var end: u64 = 0;
        var total: u64 = 0;
        for (images) |img| {
            if (img.tokens == 0 or img.tokens > v.groupLimit() or @as(u64, img.gh) * img.gw != @as(u64, img.tokens) * vcfg.patches_per_token) return error.ImageOutOfStep;
            if (img.at < end) return error.ImageOutOfStep;
            end = @as(u64, img.at) + img.tokens;
            if (end > std.math.maxInt(u32)) return error.ImageOutOfStep;
            total += img.tokens;
            if (total > try v.rowLimit()) return error.ImageRows;
        }
        v.source = .{ .images = images, .from = from };
    }

    pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

    fn encode(v: *Images, pixels: []const f32, img: lanes.stream.Image, out: u64) !void {
        const t = if (v.tower) |*tower| tower else return error.NoVision;
        try t.encode(&v.run, pixels, .{ 1, img.gh, img.gw }, out);
    }

    pub fn window(v: *Images, lo: usize, hi: usize, cancel: ?Cancel) !void {
        return v.windowWith(lo, hi, cancel, v, encode);
    }

    /// The production owner/selection/copy path, with the tower operation parameterized for recording drivers.
    pub fn windowWith(v: *Images, lo: usize, hi: usize, cancel: ?Cancel, ctx: anytype, comptime encodeFn: anytype) !void {
        v.finishWindow();
        if (hi < lo or hi - lo > step_rows) return error.BadWindow;
        // Drain even if the first encode/copy fails before a span is published.
        errdefer {
            @import("cuda_load_owner.zig").drain(v.run.stream, v.d);
            v.spans.clearRetainingCapacity();
        }
        const from = @max(lo, v.source.from);
        var row: u32 = 0;
        for (v.source.images) |img| {
            const end = @as(usize, img.at) + img.tokens;
            const first = @max(from, img.at);
            const last = @min(hi, end);
            if (first >= last) continue;
            var scratch: std.heap.ArenaAllocator = .init(v.gpa);
            defer scratch.deinit();
            const pixels = try img.materializeChecked(scratch.allocator(), if (cancel) |c| .{ .ptr = c.ptr, .check = c.check } else null);
            const count: u32 = @intCast(last - first);
            const dst = v.rows.ptr + @as(u64, row) * row_bytes;
            const whole = first == img.at and count == img.tokens;
            const out = if (whole) dst else v.rows.ptr + @as(u64, step_rows) * row_bytes;
            // Inputs retain the original full grid, even when this chunk consumes only its middle.
            try encodeFn(ctx, pixels, img, out);
            try img.checkWork(if (cancel) |c| .{ .ptr = c.ptr, .check = c.check } else null);
            if (!whole) try v.d.check(v.d.api.cuMemcpyDtoDAsync_v2(dst, out + (first - img.at) * row_bytes, @as(usize, count) * row_bytes, v.run.stream.handle), "cuMemcpyDtoDAsync media slice");
            // One input's scoped CPU arena and pinned upload ownership end before the next input starts.
            try v.run.stream.synchronize();
            try v.spans.append(v.gpa, .{ .at = @intCast(first), .n = count, .row = row });
            row += count;
        }
    }

    /// Other ranks take the active window's spans (at, n, row), then receive exactly rowCount() rows.
    pub fn take(v: *Images, slot: u32, words: []const u32) !void {
        try v.activate(slot);
        v.spans.clearRetainingCapacity();
        errdefer v.spans.clearRetainingCapacity();
        if (words.len % 3 != 0) return error.BadStep;
        var i: usize = 0;
        while (i < words.len) : (i += 3) {
            const sp: Span = .{ .at = words[i], .n = words[i + 1], .row = words[i + 2] };
            if (sp.n == 0 or sp.row != v.rowCount() or sp.row > step_rows or sp.n > step_rows - sp.row or @as(u64, sp.at) + sp.n > std.math.maxInt(u32)) return error.ImageRows;
            if (v.spans.items.len > 0) {
                const prev = v.spans.items[v.spans.items.len - 1];
                if (sp.at < @as(u64, prev.at) + prev.n) return error.ImageRows;
            }
            try v.spans.append(v.gpa, sp);
        }
        v.slot = slot;
    }

    /// Image rows the spans hold (what crosses to the other ranks).
    pub fn rowCount(v: *const Images) usize {
        var n: usize = 0;
        for (v.spans.items) |sp| n = @max(n, sp.row + sp.n);
        return n;
    }

    /// Prompt rows [lo, hi) of `slot`, embedded at `emb` in `copies` streams, take the image rows they cross.
    pub fn splice(v: *Images, s: cuda.Stream, slot: u32, lo: usize, hi: usize, emb: u64, copies: usize) !void {
        try v.activate(slot);
        if (v.spans.items.len == 0) return;
        const n = hi - lo;
        if (n > v.host_map.len) return error.BadWindow;
        var any = false;
        for (v.host_map[0..n], lo..) |*m, p| {
            m.* = -1;
            for (v.spans.items) |sp| if (p >= sp.at and p < sp.at + sp.n) {
                m.* = @intCast(sp.row + (p - sp.at));
                any = true;
            };
        }
        if (!any) return;
        try v.map.uploadAsync(0, std.mem.sliceAsBytes(v.host_map[0..n]), s.handle); // ordered after the prior splice
        var a: cuda.Args = .{};
        a.add(v.rows.ptr);
        a.add(emb);
        a.add(v.map.ptr);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(vcfg.out_hidden)));
        a.add(@as(c_int, @intCast(copies)));
        try cuda.launch.launch(v.run.f.inject, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The words for a prepare error the server cannot word (a JPEG library this machine lacks).
    pub const explain = image.decoderProblem;

    /// A video's sampled frames as patch pairs within `max_tokens` and `max_frames` (the server's prepare).
    pub fn prepareVideo(v: *const Images, a: std.mem.Allocator, s: video.Source, max_tokens: u32, max_frames: u32) !video.Prepared {
        return video.prepareLazy(a, s, v.cfg.?, @min(max_tokens, 16384), @min(max_frames, 128));
    }

    /// An image's bytes as the tower reads them, at most `max_tokens` tokens (the server's prepare).
    pub fn prepare(v: *const Images, a: std.mem.Allocator, data: []const u8, max_tokens: u32) !image_core.Prepared {
        return image.plan(v.io, a, data, v.cfg.?, max_tokens);
    }
};

test "a splice's map: rows inside a span take its rows, others keep their embedding" {
    var host: [8]i32 = undefined;
    const spans = [_]Span{ .{ .at = 3, .n = 2, .row = 0 }, .{ .at = 6, .n = 4, .row = 2 } };
    for (&host, 2..) |*m, p| {
        m.* = -1;
        for (spans) |sp| if (p >= sp.at and p < sp.at + sp.n) {
            m.* = @intCast(sp.row + (p - sp.at));
        };
    }
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, -1, 2, 3, 4, 5 }, &host);
}

test "a request with images and a video holds the sum of both row budgets" {
    var v: Images = undefined;
    v.limits = .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 };
    v.video_ids = .{ 1, 2, 3 };
    try std.testing.expectEqual(@as(u32, 6144), try v.rowLimit());
    v.video_ids = null;
    try std.testing.expectEqual(@as(u32, 2048), try v.rowLimit());
}

test "a follower's failed image admission clears its partial spans, and the same admission then succeeds" {
    const a = std.testing.allocator;
    var words: [3 * 128]u32 = undefined;
    for (0..128) |i| words[3 * i ..][0..3].* = .{ @intCast(i), 1, @intCast(i) };
    for (0..8) |offset| {
        var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = offset, .resize_fail_index = 0 });
        var v: Images = undefined;
        v.gpa = fail.allocator();
        v.spans = .empty;
        defer v.spans.deinit(v.gpa);
        v.slot = 9;
        v.limits = .{ .image_tokens = 64, .max_images = 8, .video_tokens = 128 };
        v.video_ids = .{ 1, 2, 3 };
        v.take(9, &words) catch |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(@as(usize, 0), v.spans.items.len);
            try std.testing.expectEqual(@as(u32, 9), v.slot);
        };
        fail.fail_index = std.math.maxInt(usize);
        try v.take(9, &words);
        try std.testing.expectEqual(@as(usize, 128), v.spans.items.len);
        try std.testing.expectEqual(@as(usize, 128), v.rowCount());
    }
}

test "a row budget whose image and video sum overflows is refused" {
    var v: Images = undefined;
    v.limits = .{ .image_tokens = 1, .max_images = 1, .video_tokens = std.math.maxInt(u32) };
    v.video_ids = .{ 1, 2, 3 };
    try std.testing.expectError(error.ImageRows, v.rowLimit());
}
