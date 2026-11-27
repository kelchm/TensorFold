//! GLM EXL3 prompt replacements (ahead-staged and TF_EXL3_PROMPT_GU=pc): identical sums, every block count, skip guard.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_exl3_prompt_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

const Kind = enum { gu, down };
const Fault = enum { none, last_member, wrong_skip, narrow_kernel };

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

/// pc's items: every size of 1 to 128 members (each consumer half's 1 to 8 blocks), several items, served widths.
const wide_shapes = blk: {
    var out: [134]c.Shape = undefined;
    for (out[0..128], 1..) |*s, r| s.* = .{ .rows = r, .n = 256, .k = 128 };
    out[128..].* = .{
        .{ .rows = 129, .n = 256, .k = 256 },
        .{ .rows = 256, .n = 1024, .k = 4096 },
        .{ .rows = 256, .n = 4096, .k = 1024 },
        .{ .rows = 1000, .n = 256, .k = 128 },
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
    geometry: cuda.exl3_experts.prompt.Geometry = cuda.exl3_experts.prompt.narrow,
    f: cuda.Function,
    old: cuda.Function,
    shape: c.Shape = undefined,
    input: [2]u64 = undefined,
    tp: [2]u64 = undefined,
    sv: u64 = 0,
    su: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    nitems_base: usize = 0,
    tiled: u64 = 0,
    tiled_counts: u64 = 0,
    nitems_tiled: usize = 0,
    bad_items: u64 = 0,
    members: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const wide = self.geometry.tile != cuda.exl3_experts.prompt.tile;
        const name = switch (self.fault) {
            .last_member => if (wide) "control: GLM prompt pc 128-member item's last member dropped" else "control: GLM prompt last item member dropped",
            .wrong_skip => "control: GLM prompt down skips the active expert",
            .narrow_kernel => "control: GLM prompt 128-member items on the 64-member down kernel",
            .none => if (wide) switch (self.kind) {
                .down => "cuda exl3 tf_exl3_prompt_down_glm_h halves of 128-member items",
                .gu => if (self.act == 0) "cuda exl3 tf_exl3_prompt_gu_glm_pc 128-member items, shared rotation, bf16" else "cuda exl3 tf_exl3_prompt_gu_glm_pc 128-member items, shared rotation, fp32",
            } else switch (self.kind) {
                .down => "cuda exl3 tf_exl3_prompt_down_glm",
                .gu => if (self.shared) "cuda exl3 tf_exl3_prompt_gu_glm shared rotation" else if (self.act == 0) "cuda exl3 tf_exl3_prompt_gu_glm separate bf16" else "cuda exl3 tf_exl3_prompt_gu_glm separate fp32",
            },
        };
        return .{ .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = if (self.kind == .gu) .f16 else .f32,
            .ops = 8,
            .replaces = "retained common prompt kernel with shared-slot skip",
            .nan = .{ .stated = "the retained kernel's bytes on identical NaN and infinity plants" },
            .served = if (wide) &wide_shapes else &shapes,
            .tiles = .{ .{ .size = self.geometry.tile }, .{ .size = 256, .step = 256, .min = 256 }, .{ .size = 128, .step = 128, .min = 128 } },
            .sums = .{ .exact = false, .bf16 = 0, .why = "balanced reduced-precision sums can retain constant terms; fp16 must overflow" },
        }, .ctx = self, .vt = &g.hooks(Case), .control = if (self.fault == .none) null else .shapes };
    }

    fn slotRows(self: *Case) usize {
        return (self.shape.rows + 7) / 8 * 9;
    }

    pub fn prepare(self: *Case, sh: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (sh.rows == 0 or sh.n % 256 != 0 or sh.k % 128 != 0) return false;
        self.rig.reset();
        self.shape = sh;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const xrows = if (self.kind == .gu and self.shared) (sh.rows + 7) / 8 else self.slotRows();
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
        // Routed slots (0..7 of each 9-slot token row) belong to expert 0; the 9th slots belong to skipped expert 1.
        const toks = (sh.rows + 7) / 8;
        const members = try a.alloc(i32, sh.rows + toks);
        defer a.free(members);
        for (0..sh.rows) |i| members[i] = @intCast((i / 8) * 9 + (i % 8));
        rng.shuffle(i32, members[0..sh.rows]);
        for (0..toks) |r| members[sh.rows + r] = @intCast(r * 9 + 8);
        self.members = try self.rig.input(std.mem.sliceAsBytes(members));
        for ([_]usize{ 64, self.geometry.tile }, 0..) |tile, at| {
            const active = (sh.rows + tile - 1) / tile;
            const skipped = (toks + tile - 1) / tile;
            const total = active + skipped;
            const items = try a.alloc(i32, total * 3);
            defer a.free(items);
            for (0..active) |i| {
                items[3 * i] = 0;
                items[3 * i + 1] = @intCast(i * tile);
                items[3 * i + 2] = @intCast(@min(tile, sh.rows - i * tile));
            }
            for (0..skipped) |i| {
                const idx = active + i;
                items[3 * idx] = 1;
                items[3 * idx + 1] = @intCast(sh.rows + i * tile);
                items[3 * idx + 2] = @intCast(@min(tile, toks - i * tile));
            }
            const counts = [1]i32{@intCast(total)};
            if (at == 0) {
                self.items = try self.rig.input(std.mem.sliceAsBytes(items));
                self.counts = try self.rig.input(std.mem.sliceAsBytes(&counts));
                self.nitems_base = total;
            } else {
                self.tiled = try self.rig.input(std.mem.sliceAsBytes(items));
                self.tiled_counts = try self.rig.input(std.mem.sliceAsBytes(&counts));
                self.nitems_tiled = total;
                items[3 * (active - 1) + 2] -= 1;
                self.bad_items = try self.rig.input(std.mem.sliceAsBytes(items));
            }
        }
        const bytes = self.count() * self.elemBytes();
        self.out = try self.rig.output(bytes);
        try self.rig.zero(self.out, bytes);
        self.base = try self.rig.scratch(bytes);
        try self.rig.zero(self.base, bytes);
        return true;
    }

    fn elemBytes(self: *Case) usize {
        return if (self.kind == .gu) 2 else 4;
    }

    pub fn count(self: *Case) usize {
        return self.slotRows() * self.shape.n;
    }

    fn launch(self: *Case, f: cuda.Function, dst: u64, baseline: bool) !void {
        const sh = self.shape;
        var args: cuda.Args = .{};
        const item = if (baseline) self.items else if (self.fault == .last_member) self.bad_items else self.tiled;
        const counts = if (baseline) self.counts else self.tiled_counts;
        const nitems = if (baseline) self.nitems_base else self.nitems_tiled;
        const geo = if (baseline) cuda.exl3_experts.prompt.narrow else self.geometry;
        const skip_id: i32 = if (!baseline and self.fault == .wrong_skip) 0 else 1;
        if (self.kind == .gu) {
            inline for (.{ self.input[0], self.input[1], self.tp[0], self.tp[1], item, counts, self.members, self.sv, self.sv, self.su, dst }) |v| args.add(v);
            inline for (.{ sh.k, sh.n, if (self.shared) @as(usize, 9) else 1 }) |v| args.add(@as(i32, @intCast(v)));
            args.add(@as(f32, 0));
            args.add(self.act);
            args.add(skip_id);
        } else {
            inline for (.{ self.input[0], self.tp[0], item, counts, self.members, self.sv, dst }) |v| args.add(v);
            inline for (.{ sh.k, sh.n }) |v| args.add(@as(i32, @intCast(v)));
            args.add(skip_id);
        }
        const gu = self.kind == .gu;
        try cuda.launch.launch(f, .{
            .grid = .{ .x = @intCast(sh.n / (if (gu) @as(usize, 128) else 256)), .y = @intCast(if (gu) nitems else nitems * geo.halves) },
            .block = .{ .x = if (gu) geo.threads else 256 },
            .shared = if (gu) geo.shared else 0,
        }, self.rig.s, &args);
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.f, self.out, false);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(self.old, self.base, true);
        return self.rig.bytes(self.base, self.count() * self.elemBytes());
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
            const active = row % 9 < 8 and (row / 9) * 8 + row % 9 < self.shape.rows;
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
    entries: [11]Case,

    pub fn create(a: std.mem.Allocator, gpu: check_.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        errdefer self.mod.unload();
        const p = try cuda.exl3_experts.prompt.Kernels.resolve(self.mod);
        // 0..5: 64-member ahead-staged (gu shared/sep bf16/fp32, down, last_member ctl, wrong_skip ctl)
        for (self.entries[0..6], 0..) |*x, i| {
            const gu = i < 3 or i == 4;
            x.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .kind = if (gu) .gu else .down,
                .shared = i == 0,
                .act = if (i == 2) 1 else 0,
                .fault = if (i == 4) .last_member else if (i == 5) .wrong_skip else .none,
                .f = if (gu) p.gateup_glm.? else p.down_glm.?,
                .old = if (gu) p.gateup else p.down,
            };
        }
        // 6..10: TF_EXL3_PROMPT_GU=pc (gu bf16, gu fp32, down_h, last_member ctl, narrow_kernel ctl)
        const pc = cuda.exl3_experts.prompt.pc;
        for (self.entries[6..], 0..) |*x, i| {
            const gu = i < 2 or i == 3;
            x.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .kind = if (gu) .gu else .down,
                .shared = gu,
                .act = if (i == 1) 1 else 0,
                .fault = if (i == 3) .last_member else if (i == 4) .narrow_kernel else .none,
                .geometry = if (i == 4) .{ .tile = pc.tile, .threads = 256 } else pc,
                .f = if (gu) p.gateup_glm_pc.? else if (i == 4) p.down_glm.? else p.down_glm_h.?,
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
    _ = ref;
    var x: Case = .{ .rig = .{ .gpu = undefined, .s = undefined }, .kind = .gu, .f = undefined, .old = undefined };
    const kernel = x.kernel();
    std.mem.doNotOptimizeAway(kernel.vt);
    std.mem.doNotOptimizeAway(&Group.create);
    std.mem.doNotOptimizeAway(&Group.destroy);
    try std.testing.expectEqual(@as(usize, 69), kernel.spec.served.len);
    x.geometry = cuda.exl3_experts.prompt.pc;
    const wide = x.kernel();
    try std.testing.expectEqual(@as(usize, 134), wide.spec.served.len);
    for (wide_shapes[0..128], 1..) |sh, r| try std.testing.expectEqual(r, sh.rows);
    x.kind = .down;
    try std.testing.expectEqualStrings("cuda exl3 tf_exl3_prompt_down_glm_h halves of 128-member items", x.kernel().spec.name);
}
