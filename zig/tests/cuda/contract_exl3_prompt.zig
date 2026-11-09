//! GLM prompt replacements: identical full-K sums, every 8-member specialization, untouched shared slots, bf16 slots.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_exl3_prompt_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

const Kind = enum { gu, slots, down, b16 }; // b16: the down kernel's slot rows rounded to bf16 (TF_GLM_EXL3_BF16_SLOTS)
const Fault = enum { none, last_member, slot_stride };

fn depth(s: c.Shape) usize {
    return s.k / 16 + 16 + 14;
}

const shapes = blk: {
    var out: [69]c.Shape = undefined;
    for (out[0..64], 1..) |*s, r| s.* = .{ .rows = r, .n = 256, .k = 128 };
    out[64..].* = .{
        .{ .rows = 65, .n = 256, .k = 256 },
        .{ .rows = 128, .n = 1024, .k = 4096 },
        .{ .rows = 128, .n = 4096, .k = 1024 },
        .{ .rows = 2048, .n = 256, .k = 128 },
        .{ .rows = 4096, .n = 256, .k = 128 },
    };
    break :blk out;
};

pub const Case = struct {
    rig: rig_.Rig,
    kind: Kind,
    shared: bool = false,
    act: i32 = 0,
    fault: Fault = .none,
    f: cuda.Function,
    old: cuda.Function,
    shape: c.Shape = undefined,
    input: [2]u64 = undefined,
    tp: [2]u64 = undefined,
    sv: u64 = 0,
    su: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    members: u64 = 0,
    bad_items: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    compact: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.compact.deinit(self.rig.gpu.gpa);
    }

    pub fn kernel(self: *Case) g.Kernel {
        const name = switch (self.fault) {
            .last_member => "control: GLM prompt last item member dropped",
            .slot_stride => "control: GLM prompt compact stride posing as routed slots",
            .none => switch (self.kind) {
                .slots => "cuda exl3 tf_exl3_prompt_down_slots direct routed slots",
                .down => "cuda exl3 tf_exl3_prompt_down_glm",
                .b16 => "cuda exl3 tf_exl3_prompt_down_glm_b16 bf16 slots",
                .gu => if (self.shared) "cuda exl3 tf_exl3_prompt_gu_glm shared rotation" else if (self.act == 0) "cuda exl3 tf_exl3_prompt_gu_glm separate bf16" else "cuda exl3 tf_exl3_prompt_gu_glm separate fp32",
            },
        };
        return .{ .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = switch (self.kind) {
                .gu => .f16,
                .b16 => .bf16,
                .slots, .down => .f32,
            },
            .ops = 8,
            .replaces = if (self.kind == .b16) "retained common prompt kernel scattered into routed slots, then __float2bfloat16_rn" else "retained common prompt kernel, compact down output scattered into routed slots",
            .nan = .{ .stated = if (self.kind == .b16) "the retained kernel's bytes rounded to bf16 on identical plants, NaN as 0x7fff" else "the retained kernel's bytes on identical NaN and infinity plants" },
            .served = &shapes,
            .tiles = .{ .{ .size = 64 }, .{ .size = 256, .step = 256, .min = 256 }, .{ .size = 128, .step = 128, .min = 128 } },
            .sums = .{ .exact = false, .bf16 = 0, .why = "balanced reduced-precision sums can retain constant terms; fp16 must overflow" },
        }, .ctx = self, .vt = &g.hooks(Case), .control = if (self.fault == .none) null else .shapes };
    }

    pub fn prepare(self: *Case, sh: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (sh.rows == 0 or sh.n % 256 != 0 or sh.k % 128 != 0) return false;
        self.rig.reset();
        self.shape = sh;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const xrows = if (self.kind == .gu and self.shared) (sh.rows + 7) / 8 else sh.rows;
        const x = try a.alloc(u16, xrows * sh.k);
        defer a.free(x);
        for (&self.input) |*ptr| {
            for (x) |*v| v.* = @bitCast(@as(f16, @floatCast(if (fill == .stress) ref.input else rng.float(f32) * 2 - 1)));
            if (fill == .nonfinite) {
                x[3] = 0x7e00;
                if (xrows > 1) x[sh.k + 5] = 0x7c00;
                if (xrows > 2) x[2 * sh.k + 9] = 0xfc00;
            }
            ptr.* = try self.rig.input(std.mem.sliceAsBytes(x));
        }
        if (self.shared) self.input[1] = self.input[0];
        const tw = try a.alloc(i16, sh.k / 16 * (sh.n / 16) * 64);
        defer a.free(tw);
        for (&self.tp) |*ptr| {
            for (tw) |*v| v.* = if (fill == .stress) -1 else rng.int(i16);
            const address = [1]u64{try self.rig.input(std.mem.sliceAsBytes(tw))};
            ptr.* = try self.rig.input(std.mem.sliceAsBytes(&address));
        }
        const scale = try a.alloc(u16, sh.n);
        defer a.free(scale);
        @memset(scale, @bitCast(@as(f16, @floatCast(if (self.kind == .gu and fill == .stress) ref.scale else 1))));
        self.sv = try self.rig.input(std.mem.sliceAsBytes(scale));
        @memset(scale, @bitCast(@as(f16, @floatCast(if (fill == .stress) ref.scale else 1))));
        self.su = try self.rig.input(std.mem.sliceAsBytes(scale));
        const members = try a.alloc(i32, sh.rows);
        defer a.free(members);
        for (members, 0..) |*v, i| v.* = @intCast(i);
        rng.shuffle(i32, members);
        self.members = try self.rig.input(std.mem.sliceAsBytes(members));
        const nitems = (sh.rows + 63) / 64;
        const items = try a.alloc(i32, nitems * 3);
        defer a.free(items);
        for (0..nitems) |i| {
            items[3 * i] = 0;
            items[3 * i + 1] = @intCast(i * 64);
            items[3 * i + 2] = @intCast(@min(64, sh.rows - i * 64));
        }
        self.items = try self.rig.input(std.mem.sliceAsBytes(items));
        items[items.len - 1] -= 1;
        self.bad_items = try self.rig.input(std.mem.sliceAsBytes(items));
        const counts = [1]i32{@intCast(nitems)};
        self.counts = try self.rig.input(std.mem.sliceAsBytes(&counts));
        self.out = try self.rig.output(self.count() * self.elemBytes());
        try self.rig.zero(self.out, self.count() * self.elemBytes());
        self.base = try self.rig.scratch(sh.rows * sh.n * self.baseBytes());
        return true;
    }

    fn elemBytes(self: *Case) usize {
        return if (self.kind == .gu or self.kind == .b16) 2 else 4;
    }

    /// The retained kernel's element: fp16 gate/up rows, fp32 down rows.
    fn baseBytes(self: *Case) usize {
        return if (self.kind == .gu) 2 else 4;
    }

    pub fn count(self: *Case) usize {
        const rows = if (self.kind == .gu) self.shape.rows else (self.shape.rows + 7) / 8 * 9;
        return rows * self.shape.n;
    }

    fn launch(self: *Case, f: cuda.Function, dst: u64, baseline: bool) !void {
        const sh = self.shape;
        var args: cuda.Args = .{};
        const item = if (!baseline and self.fault == .last_member) self.bad_items else self.items;
        if (self.kind == .gu) {
            inline for (.{ self.input[0], self.input[1], self.tp[0], self.tp[1], item, self.counts, self.members, self.sv, self.sv, self.su, dst }) |v| args.add(v);
            inline for (.{ sh.k, sh.n, if (self.shared) @as(usize, 8) else 1 }) |v| args.add(@as(i32, @intCast(v)));
            args.add(@as(f32, 0));
            args.add(self.act);
        } else {
            inline for (.{ self.input[0], self.tp[0], item, self.counts, self.members, self.sv, dst }) |v| args.add(v);
            inline for (.{ sh.k, sh.n, @as(usize, 8) }) |v| args.add(@as(i32, @intCast(v)));
            if (!baseline) args.add(@as(i32, if (self.fault == .slot_stride) 8 else 9));
        }
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.n / (if (self.kind == .gu) @as(usize, 128) else 256)), .y = @intCast((sh.rows + 63) / 64) }, .block = .{ .x = 256 } }, self.rig.s, &args);
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.f, self.out, false);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(self.old, self.base, true);
        const src = try self.rig.bytes(self.base, self.shape.rows * self.shape.n * self.baseBytes());
        if (self.kind == .gu) return src;
        try self.compact.resize(self.rig.gpu.gpa, self.count() * 4);
        @memset(self.compact.items, 0);
        const rowbytes = self.shape.n * 4;
        for (0..self.shape.rows) |p| {
            const at = (p / 8 * 9 + p % 8) * rowbytes;
            @memcpy(self.compact.items[at..][0..rowbytes], src[p * rowbytes ..][0..rowbytes]);
        }
        if (self.kind != .b16) return self.compact.items;
        const all = self.compact.items;
        for (0..self.count()) |i| { // in place: value i's two bytes land before value i + 1's four are read
            const v: f32 = @bitCast(std.mem.readInt(u32, all[i * 4 ..][0..4], .little));
            std.mem.writeInt(u16, all[i * 2 ..][0..2], if (std.math.isNan(v)) 0x7fff else c.toBf16(v), .little);
        }
        return all[0 .. self.count() * 2];
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        if (!std.mem.eql(u8, got, want)) return "differs from the retained prompt kernel";
        return null;
    }

    fn fillReference(self: *Case, out: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) !void {
        const sum = try ref.dot(self.rig.gpu.gpa, self.shape.k, acc, order);
        for (out, 0..) |*v, i| {
            const row = i / self.shape.n;
            const active = self.kind == .gu or (row % 9 < 8 and row / 9 * 8 + row % 9 < self.shape.rows);
            v.* = if (active) ref.output(sum, self.kind == .gu, self.act, i % self.shape.n) else 0;
            if (mag) |m| m[i] = @abs(v.*) * 4;
        }
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        try self.fillReference(want, mag, null, .sequential);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.fillReference(out, null, acc, order);
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    entries: [8]Case,

    pub fn create(a: std.mem.Allocator, gpu: check_.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        errdefer self.mod.unload();
        const p = try cuda.exl3_experts.prompt.Kernels.resolve(self.mod);
        for (&self.entries, 0..) |*x, i| {
            const gu = i < 3 or i == 5;
            x.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .kind = if (gu) .gu else if (i == 3) .slots else if (i == 7) .b16 else .down,
                .shared = i == 0,
                .act = if (i == 2) 1 else 0,
                .fault = if (i == 5) .last_member else if (i == 6) .slot_stride else .none,
                .f = if (gu) p.gateup_glm.? else if (i == 3) p.down_slots.? else if (i == 7) p.down_glm_b16.? else p.down_glm.?,
                .old = if (gu) p.gateup else p.down,
            };
        }
        errdefer for (&self.entries) |*x| x.deinit();
        for (&self.entries) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.entries) |*x| x.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};

test "GLM prompt contract: every hook compiles without embedded CUDA images" {
    var x: Case = .{ .rig = .{ .gpu = undefined, .s = undefined }, .kind = .gu, .f = undefined, .old = undefined };
    const kernel = x.kernel();
    std.mem.doNotOptimizeAway(kernel.vt);
    std.mem.doNotOptimizeAway(&Group.create);
    std.mem.doNotOptimizeAway(&Group.destroy);
    try std.testing.expectEqual(@as(usize, 69), kernel.spec.served.len);
    x.kind = .b16;
    const half = x.kernel();
    try std.testing.expectEqual(c.Elem.bf16, half.spec.out);
    try std.testing.expect(half.spec.replaces != null and half.vt.replaced != null);
}
