//! The sparse chunks with tf_glm_merge in the kernel replace the chunk kernels then the merge, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const Chunks = @import("contract_glm_attn.zig").Chunks;
const lw = 512;
const heads = 32;
const scale: f32 = 0.0625;

pub const Case = struct {
    rig: Rig,
    rows: Chunks, // queries, keys, token lists and the float64 reference over them
    fns: *const glm.glue.Kernels,
    hb: u32,
    fp8: bool,
    drop: bool, // control: the fused launch leaves out the row's last chunk
    name: [100]u8 = undefined,
    msg: [100]u8 = undefined,
    ins: [5]u64 = @splat(0), // queries, BF16 rows or E4M3 codes, scales, tokens, counts
    out: u64 = 0,
    base: u64 = 0, // the replaced launches' partials, then their output
    nch: usize = 0,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        self.rows.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) !g.Kernel {
        const name = try std.fmt.bufPrint(&self.name, "{s}GLM tf_glm_chunks_{d}{s}_fused{s}", .{ if (self.drop) "control: " else "cuda ", self.hb, if (self.fp8) "_fp8" else "", if (self.drop) " without its last chunk" else "" });
        const s: c.Spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .out = .bf16,
            .replaces = "the sparse chunk kernel then tf_glm_merge",
            .nan = .{ .stated = "Every merged output equals chunks + merge, including NaNs, infinities and rows left alone" },
            .served = &.{ .{ .rows = 8, .n = heads * lw, .k = glm.sparse.width }, .{ .rows = 64, .n = heads * lw, .k = 300 } },
            .tiles = .{ .{ .size = 64 }, .{}, .{ .size = 32 } },
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "online softmax over 512-key chunks, then the chunk merge: no input makes its sums exact in fp32" },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.drop) .shapes else null };
    }

    fn depth(s: c.Shape) usize {
        return lw + glm.latent.chunk + s.k / glm.latent.chunk + 1;
    }

    /// Chunks' rows and token lists; E4M3 rows decode within [-1, 1]; the last row's count is 0 except under stress.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.rows.prepareRows(shape, fill, seed)) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const cnt = try a.alloc(i32, shape.rows);
        defer a.free(cnt);
        @memset(cnt, @intCast(shape.k));
        if (fill != .stress and shape.rows > 1) cnt[shape.rows - 1] = 0; // a row tf_glm_merge leaves alone
        const tok = try a.alloc(i32, self.rows.keys.len);
        defer a.free(tok);
        for (tok, self.rows.keys) |*v, key| v.* = @intCast(key);
        self.ins[0] = try self.rig.input(std.mem.sliceAsBytes(self.rows.q));
        if (self.fp8) {
            var prng = std.Random.DefaultPrng.init(seed ^ 0x667038);
            const rng = prng.random();
            const span = self.rows.cache.len / lw;
            const codes = try a.alloc(u8, span * lw);
            defer a.free(codes);
            const scales = try a.alloc(f32, span);
            defer a.free(scales);
            for (scales) |*v| v.* = (0.125 + rng.float(f32) * 0.875) / 448;
            for (codes, 0..) |*v, i| {
                const code = rng.int(u8);
                v.* = if (code & 127 == 127) code ^ 1 else code;
                self.rows.cache[i] = glm.kv8_ref.decoded(v.*, scales[i / lw]);
            }
            if (fill == .nonfinite) codes[self.rows.keys[0] * lw + 9] = 0x7f;
            self.ins[1] = try self.rig.input(codes);
            self.ins[2] = try self.rig.input(std.mem.sliceAsBytes(scales));
        } else {
            self.ins[1] = try self.rig.input(std.mem.sliceAsBytes(self.rows.cache));
            self.ins[2] = 0;
        }
        self.ins[3] = try self.rig.input(std.mem.sliceAsBytes(tok));
        self.ins[4] = try self.rig.input(std.mem.sliceAsBytes(cnt));
        self.nch = glm.latent.chunksFor(shape.k);
        self.out = try self.rig.output(self.count() * 2);
        self.base = try self.rig.scratch(self.nch * shape.rows * heads * (lw + 2) * 4 + self.count() * 2);
        return true;
    }

    fn smem(self: *const Case) u32 {
        return if (self.hb == 32) glm.glue.chunksSmem(32) else glm.glue.chunksSmem(16);
    }

    fn args(self: *Case, a: *cuda.Args) void {
        a.add(self.ins[0]);
        a.add(self.ins[1]);
        if (self.fp8) a.add(self.ins[2]);
    }

    pub fn run(self: *Case) !void {
        const s = self.rows.shape;
        const f = if (self.hb == 16) (if (self.fp8) self.fns.chunks_16_fp8_fused else self.fns.chunks_16_fused) else if (self.fp8) self.fns.chunks_32_fp8_fused else self.fns.chunks_32_fused;
        const nch = if (self.drop) @max(self.nch, 2) - 1 else self.nch;
        var a: cuda.Args = .{};
        self.args(&a);
        inline for (.{ self.ins[3], self.ins[4], self.out, @as(i64, heads), @as(i64, @intCast(s.k)), scale, @as(c_int, @intCast(nch)) }) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.rows), .y = heads / self.hb }, .block = .{ .x = glm.latent.chunk_threads }, .shared = self.smem() }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.rows.count();
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        return self.rows.reference(want, mag);
    }
    pub fn bounds(self: *Case, want: []const f64, mag: []const f64, allow: []f64) !void {
        return self.rows.bounds(want, mag, allow);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.rows.simulate(acc, order, out);
    }

    /// The sparse chunk kernel (grid z the chunks) into fp32 partials, then tf_glm_merge into a poisoned output.
    pub fn replaced(self: *Case) ![]const u8 {
        const s = self.rows.shape;
        const n = self.nch * s.rows * heads;
        const p = [3]u64{ self.base, self.base + n * lw * 4, self.base + n * (lw + 1) * 4 };
        const out: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base + n * (lw + 2) * 4, .len = self.count() * 2 };
        try out.fill8(0xff, self.rig.s.handle);
        const f = if (self.hb == 16) (if (self.fp8) self.fns.chunks_16_fp8 else self.fns.chunks_16) else if (self.fp8) self.fns.chunks_32_fp8 else self.fns.chunks_32;
        const rows: i64 = @intCast(s.rows);
        var a: cuda.Args = .{};
        self.args(&a);
        // pos (the counts here) is unread when sparse
        inline for (.{ self.ins[4], self.ins[3], self.ins[4], p[0], p[1], p[2], rows, @as(i64, heads), @as(i64, @intCast(s.k)), scale, @as(c_int, 1) }) |v| a.add(v);
        if (!self.fp8) inline for (.{ @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.rows), .y = heads / self.hb, .z = @intCast(self.nch) }, .block = .{ .x = glm.latent.chunk_threads }, .shared = self.smem() }, self.rig.s, &a);
        var m: cuda.Args = .{};
        inline for (.{ p[0], p[1], p[2], out.ptr, self.ins[4], rows, @as(i64, heads), @as(c_int, @intCast(self.nch)), @as(c_int, 1) }) |v| m.add(v);
        try cuda.launch.launch(self.fns.merge, .{ .grid = .{ .x = @intCast(s.rows), .y = heads }, .block = .{ .x = 128 } }, self.rig.s, &m);
        return self.rig.bytes(out.ptr, out.len);
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from chunks + merge", .{i});
        return null;
    }
};

pub const Group = struct {
    cases: [8]Case,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, fns: *const glm.glue.Kernels, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        for (&self.cases, 0..) |*x, i| {
            const hb: u32 = if (i & 1 == 0) 16 else 32;
            const rows: Chunks = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.merge, .merge = fns.merge, .form = .{ .hb = hb, .sparse = true } };
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = rows, .fns = fns, .hb = hb, .fp8 = i & 2 != 0, .drop = i >= 4 };
            try kernels.append(a, try x.kernel());
        }
        return self;
    }
    pub fn destroy(self: *Group, a: std.mem.Allocator) void {
        for (&self.cases) |*x| x.deinit();
        a.destroy(self);
    }
};
