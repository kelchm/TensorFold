//! A shared window's KDA chains in one launch a kernel (kda.chainSegs) against each segment's own chain launches.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const kda = glm.kda;
const dk = kda.dk;
const window = 8; // fw.window_rows: a segment's most rows
const taps = 3; // a conv window's rows

/// A planted fault in chainSegs' table; each must go red at the shapes gate.
pub const Fault = enum { none, neighbour_state, shared_scratch };

/// rows: the window's rows over k segments (ragged, one to eight each); n: a rank's heads x 128 (TP 2, TP 3's widest).
const served = [_]c.Shape{
    .{ .rows = 32, .n = 32 * dk, .k = 4 },
    .{ .rows = 4, .n = 32 * dk, .k = 4 },
    .{ .rows = 20, .n = 32 * dk, .k = 4 },
    .{ .rows = 37, .n = 22 * dk, .k = 7 },
    .{ .rows = 128, .n = 32 * dk, .k = 16 },
};

/// A window's `rows` rows over `k` segments, ragged and each one to `window` rows; false when they cannot be.
pub fn split(rows: usize, k: usize, out: *[kda.max_segs]usize) bool {
    if (k == 0 or k > kda.max_segs or rows < k or rows > k * window) return false;
    for (out[0..k]) |*s| s.* = 1;
    var left = rows - k;
    var pass: usize = 0;
    while (left > 0) : (pass += 1) for (out[0..k], 0..) |*s, j| {
        const more = @min(left, (j * 5 + pass * 3) % window, window - s.*);
        s.* += more;
        left -= more;
    };
    return true;
}

/// The fault planted in a built table of `n` segments; `w` the window's scratch from row 0.
pub fn plant(t: *kda.Segs, n: usize, fault: Fault, w: kda.Wide) void {
    switch (fault) {
        .none => {},
        .neighbour_state => {
            const first = t.s[0].state_in;
            for (t.s[0 .. n - 1], t.s[1..n]) |*s, next| s.state_in = next.state_in;
            t.s[n - 1].state_in = first;
        },
        .shared_scratch => for (t.s[0..n]) |*s| {
            s.q = w.q; // every segment's rows from row 0, as the launches one after another reused them
            s.y = w.y;
        },
    }
}

fn noSums(_: c.Shape) usize {
    return 0;
}

pub const Case = struct {
    rig: Rig,
    ks: *const kda.Kernels,
    fault: Fault,
    msg: [128]u8 = undefined,
    heads: usize = 0,
    segs: usize = 0,
    rows: [kda.max_segs]usize = undefined,
    total: usize = 0,
    ins: [8]u64 = @splat(0), // P, A, G, conv windows, states, conv weights, a_log and dt_bias, norm weight
    out: u64 = 0,
    base: u64 = 0,
    wide: kda.Wide = .{ .q = 0, .y = 0 },
    lone: kda.Wide = .{ .q = 0, .y = 0 },

    const vt = g.hooks(Case);
    const names = [_][]const u8{ "cuda GLM KDA segs: prep, step and out, every segment in one launch", "control: GLM KDA segs reading the next segment's state", "control: GLM KDA segs sharing one q and y scratch" };

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const s: c.Spec = .{
            .name = names[@backingInt(self.fault)],
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "tf_glm_kda_prep, _step and _out a segment, one segment after another",
            .nan = .{ .stated = "out rows, states and saves equal the per-segment launches' bytes, NaN for NaN" },
            .served = &served,
            .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 16 } },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "the per-segment launches' own arithmetic: every block runs its segment's code" },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    fn pw(self: *const Case) usize {
        return std.mem.alignForward(usize, 3 * self.heads * dk + 2 * dk + self.heads, 8); // TF_GLM_KDA_PAD's rows
    }

    fn stateBytes(self: *const Case) usize {
        return self.heads * kda.dv * dk * 4;
    }

    /// The output's parts: out rows, every segment's state, then saves k, v, decay and beta, each from row 0.
    fn parts(self: *const Case) [7]usize {
        const n = self.heads * dk;
        const r = self.total;
        const sizes = [6]usize{ r * n * 2, self.segs * self.stateBytes(), r * n * 4, r * n * 2, r * n * 4, r * self.heads * 4 };
        var at: [7]usize = undefined;
        at[0] = 0;
        for (sizes, 0..) |s, i| at[i + 1] = at[i] + s;
        return at;
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % dk != 0 or shape.n == 0 or !split(shape.rows, shape.k, &self.rows)) return false;
        self.rig.reset();
        self.heads = shape.n / dk;
        self.segs = shape.k;
        self.total = shape.rows;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6b6461);
        const rng = prng.random();
        const n = self.heads * dk;
        const ch = 3 * n;
        const counts = [8]usize{ self.total * self.pw(), self.total * n, self.total * n, self.segs * taps * ch, self.segs * self.stateBytes() / 4, ch * 4, self.heads + n, kda.dv };
        const scales = [8]f32{ 1, 2, 2, 1, 0.5, 0.5, 1, 1 };
        for (counts, scales, 0..) |len, scale, i| {
            const bf = i < 4 or i == 7;
            const bytes = try a.alloc(u8, len * @as(usize, if (bf) 2 else 4));
            defer a.free(bytes);
            for (0..len) |j| {
                var v = (rng.float(f32) * 2 - 1) * scale;
                if (i == 6 and j < self.heads) v = -1 + rng.float(f32) * 1.5; // a_log
                if (i == 7) v = 0.5 + rng.float(f32); // the norm's weight
                if (fill == .nonfinite) v = planted(i, j, self.heads, v);
                if (bf) std.mem.writeInt(u16, bytes[j * 2 ..][0..2], c.toBf16(v), .little) else std.mem.writeInt(u32, bytes[j * 4 ..][0..4], @bitCast(v), .little);
            }
            self.ins[i] = try self.rig.input(bytes);
        }
        const at = self.parts();
        self.out = try self.rig.output(at[6]);
        self.base = try self.rig.scratch(at[6]);
        var most: usize = 0;
        for (self.rows[0..self.segs]) |r| most = @max(most, r);
        self.wide = .{ .q = try self.rig.scratch(self.total * n * 4), .y = try self.rig.scratch(self.total * n * 2) };
        self.lone = .{ .q = try self.rig.scratch(most * n * 4), .y = try self.rig.scratch(most * n * 2) };
        return true;
    }

    /// NaN in row 0's q channel 5, +Inf in row 1's decay, NaN in row 2's gate, -Inf in segment 1's state.
    fn planted(i: usize, j: usize, heads: usize, v: f32) f32 {
        const n = heads * dk;
        return switch (i) {
            0 => if (j == 5) std.math.nan(f32) else v,
            1 => if (j == n + 3) std.math.inf(f32) else v,
            2 => if (j == 2 * n + 7) std.math.nan(f32) else v,
            4 => if (j == n * kda.dv + 9) -std.math.inf(f32) else v,
            else => v,
        };
    }

    /// Segment i's chain and saves as the engine passes them (its rows from row0; saves in `to`'s parts).
    fn chainOf(self: *const Case, i: usize, row0: usize, to: u64) struct { kda.Chain, kda.Saves } {
        const n = self.heads * dk;
        const at = self.parts();
        const ch: kda.Chain = .{ .p = self.ins[0] + row0 * self.pw() * 2, .p_stride = self.pw(), .b_off = 3 * n + 2 * dk, .a = self.ins[1] + row0 * n * 2, .a_stride = n, .g = self.ins[2] + row0 * n * 2, .g_stride = n, .conv_state = self.ins[3] + i * taps * 3 * n * 2, .conv_w = self.ins[5], .state_in = self.ins[4] + i * self.stateBytes(), .a_log = self.ins[6], .dt_bias = self.ins[6] + self.heads * 4, .norm_w = self.ins[7], .eps = 1e-6, .lower = -5, .heads = self.heads };
        const sv: kda.Saves = .{ .out = to + at[0] + row0 * n * 2, .state_out = to + at[1] + i * self.stateBytes(), .k = to + at[2] + row0 * n * 4, .v = to + at[3] + row0 * n * 2, .g = to + at[4] + row0 * n * 4, .b = to + at[5] + row0 * self.heads * 4 };
        return .{ ch, sv };
    }

    pub fn run(self: *Case) !void {
        var t: kda.Segs = undefined;
        var row0: usize = 0;
        for (self.rows[0..self.segs], 0..) |r, i| {
            const ch, const sv = self.chainOf(i, row0, self.out);
            t.s[i] = kda.seg(ch, sv, self.wide, row0, r);
            row0 += r;
        }
        plant(&t, self.segs, self.fault, self.wide);
        try kda.chainSegs(self.ks, self.rig.s, self.chainOf(0, 0, self.out)[0], &t, self.segs);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.parts()[6];
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        for (want, mag, try self.replaced()) |*w, *m, byte| {
            w.* = @floatFromInt(byte);
            m.* = @abs(w.*);
        }
    }

    /// Each segment through kda.chain (wide) in turn, its q and y rows reused from row 0, as before chainSegs.
    pub fn replaced(self: *Case) ![]const u8 {
        var row0: usize = 0;
        for (self.rows[0..self.segs], 0..) |r, i| {
            const ch, const sv = self.chainOf(i, row0, self.base);
            try kda.chain(self.ks, self.rig.s, ch, r, sv, self.lone);
            row0 += r;
        }
        return self.rig.bytes(self.base, self.count());
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from the per-segment launches'", .{i});
        return null;
    }
};

/// The segments' projection rows into their streams in one launch (kda.copySegs) against cuMemcpyDtoDAsync a segment.
pub const CopyCase = struct {
    rig: Rig,
    ks: *const kda.Kernels,
    short: bool, // the planted control: each copy one 16-byte unit short
    msg: [128]u8 = undefined,
    rows: [kda.max_segs]usize = undefined,
    segs: usize = 0,
    width: usize = 0,
    src: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    total: usize = 0,

    const vt = g.hooks(CopyCase);
    const gap = 16; // poisoned bytes between the destinations: a copy past its rows shows

    /// rows: the window's rows over k segments; n: a row's bf16 columns (TP 2's stack, TP 3's padded one).
    const shapes = [_]c.Shape{ .{ .rows = 32, .n = 12576, .k = 4 }, .{ .rows = 37, .n = 8728, .k = 7 }, .{ .rows = 128, .n = 12576, .k = 16 } };

    pub fn deinit(self: *CopyCase) void {
        self.rig.deinit();
    }

    pub fn kernel(self: *CopyCase) g.Kernel {
        const s: c.Spec = .{
            .name = if (self.short) "control: GLM KDA segs copies one unit short" else "cuda GLM KDA segs: every segment's projection rows in one copy",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "cuMemcpyDtoDAsync a segment",
            .nan = .{ .stated = "bytes are copied, NaN payloads included" },
            .served = &shapes,
            .tiles = .{ .{}, .{}, .{ .size = 16 } },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "a copy sums nothing" },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.short) .shapes else null };
    }

    fn dst(self: *const CopyCase, to: u64, row0: usize, i: usize) u64 {
        return to + row0 * self.width * 2 + i * gap;
    }

    pub fn prepare(self: *CopyCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % 8 != 0 or !split(shape.rows, shape.k, &self.rows)) return false;
        self.rig.reset();
        self.segs = shape.k;
        self.width = shape.n;
        const a = self.rig.gpu.gpa;
        const bytes = try a.alloc(u8, shape.rows * shape.n * 2);
        defer a.free(bytes);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x636f7079);
        prng.random().bytes(bytes);
        if (fill == .nonfinite) std.mem.writeInt(u16, bytes[10..12], 0x7fc1, .little);
        self.src = try self.rig.input(bytes);
        self.total = bytes.len + shape.k * gap;
        self.out = try self.rig.output(self.total);
        self.base = try self.rig.scratch(self.total);
        return true;
    }

    pub fn run(self: *CopyCase) !void {
        var t: kda.Copies = undefined;
        var row0: usize = 0;
        for (self.rows[0..self.segs], 0..) |r, i| {
            const n: i64 = @intCast(r * self.width * 2 / 16);
            t.c[i] = .{ .src = self.src + row0 * self.width * 2, .dst = self.dst(self.out, row0, i), .n = if (self.short) n - 1 else n };
            row0 += r;
        }
        try kda.copySegs(self.ks, self.rig.s, &t, self.segs);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *CopyCase) usize {
        return self.total;
    }

    pub fn reference(self: *CopyCase, want: []f64, mag: []f64) !void {
        for (want, mag, try self.replaced()) |*w, *m, byte| {
            w.* = @floatFromInt(byte);
            m.* = @abs(w.*);
        }
    }

    /// Each segment's rows by cuMemcpyDtoDAsync, over the output's poison, as the mixer copied them before.
    pub fn replaced(self: *CopyCase) ![]const u8 {
        const d = self.rig.gpu.d;
        const base: cuda.DeviceBuffer = .{ .d = d, .ptr = self.base, .len = self.total };
        try base.fill8(0xff, self.rig.s.handle);
        var row0: usize = 0;
        for (self.rows[0..self.segs], 0..) |r, i| {
            try d.check(d.api.cuMemcpyDtoDAsync_v2(self.dst(self.base, row0, i), self.src + row0 * self.width * 2, r * self.width * 2, self.rig.s.handle), "cuMemcpyDtoDAsync");
            row0 += r;
        }
        return self.rig.bytes(self.base, self.total);
    }

    pub fn check(self: *CopyCase, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from cuMemcpyDtoDAsync's", .{i});
        return null;
    }
};

pub const Group = struct {
    module: cuda.Module,
    ks: kda.Kernels,
    cases: [3]Case,
    copies: [2]CopyCase,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
        errdefer self.module.unload();
        self.ks = try kda.Kernels.load(self.module);
        for (&self.cases, std.enums.values(Fault)) |*x, fault| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = &self.ks, .fault = fault };
            try kernels.append(a, x.kernel());
        }
        for (&self.copies, [_]bool{ false, true }) |*x, short| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = &self.ks, .short = short };
            try kernels.append(a, x.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group, a: std.mem.Allocator) void {
        for (&self.cases) |*x| x.deinit();
        for (&self.copies) |*x| x.deinit();
        self.module.unload();
        a.destroy(self);
    }
};

test "GLM KDA segs contract: every served window splits into ragged segments of one to eight rows" {
    var rows: [kda.max_segs]usize = undefined;
    for (served) |s| {
        try std.testing.expect(split(s.rows, s.k, &rows));
        var sum: usize = 0;
        for (rows[0..s.k]) |r| {
            try std.testing.expect(r >= 1 and r <= window);
            sum += r;
        }
        try std.testing.expectEqual(s.rows, sum);
        if (s.rows > s.k and s.rows < s.k * window) try std.testing.expect(std.mem.min(usize, rows[0..s.k]) != std.mem.max(usize, rows[0..s.k]));
    }
    try std.testing.expect(!split(3, 4, &rows) and !split(33, 4, &rows) and !split(17, 17, &rows) and !split(1, 0, &rows));
}

test "GLM KDA segs contract: every served copy window splits and copies whole 16-byte units" {
    var rows: [kda.max_segs]usize = undefined;
    for (CopyCase.shapes) |s| {
        try std.testing.expect(split(s.rows, s.k, &rows));
        var row0: usize = 0;
        for (rows[0..s.k], 0..) |r, i| {
            try std.testing.expect(kda.copyFits(row0 * s.n * 2, row0 * s.n * 2 + i * CopyCase.gap, r * s.n * 2));
            row0 += r;
        }
    }
}

test "GLM KDA segs contract: each planted table fault moves what its segments read" {
    var t: kda.Segs = undefined;
    const w: kda.Wide = .{ .q = 1 << 20, .y = 1 << 21 };
    for (&t.s, 0..) |*s, i| s.* = .{ .p = 0, .a = 0, .g = 0, .cs = 0, .state_in = 100 + i, .state_out = 0, .q = w.q + i * 64, .k = 0, .v = 0, .gs = 0, .b = 0, .y = w.y + i * 32, .out = 0, .rows = 1 };
    const clean = t;
    plant(&t, 4, .none, w);
    try std.testing.expectEqualDeep(clean, t);
    plant(&t, 4, .neighbour_state, w);
    for (0..4) |i| try std.testing.expectEqual(clean.s[(i + 1) % 4].state_in, t.s[i].state_in);
    try std.testing.expectEqual(clean.s[4].state_in, t.s[4].state_in);
    t = clean;
    plant(&t, 4, .shared_scratch, w);
    for (t.s[0..4]) |s| try std.testing.expect(s.q == w.q and s.y == w.y);
    try std.testing.expect(clean.s[1].q != w.q);
}
