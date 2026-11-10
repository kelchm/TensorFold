//! TF_GLM_SHARED_SLOT: the shared expert's down projection straight into its MoE slot, the parent's bytes put there.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const Rig = @import("contract_rig.zig").Rig;
const mm = @import("contract_glm_mm.zig");

const slots = 9; // top 8 and the shared expert, last
const served = [_]c.Shape{
    .{ .rows = 37, .n = 4096, .k = 1024 },
    .{ .rows = 768, .n = 4096, .k = 1024 },
    .{ .rows = 2048, .n = 4096, .k = 1024 },
    .{ .rows = 129, .n = 4096, .k = 2048 },
    .{ .rows = 1, .n = 4096, .k = 2048 },
};

fn depth(shape: c.Shape) usize {
    return 64 + shape.k / 64;
}

/// short: rows a slot short of the slot stride; seventh: slot 7, a routed expert's.
pub const Fault = enum { none, short, seventh };

pub const Slot = struct {
    p: mm.Product, // x, the weights, the float64 sums; its own out is never bound
    rig: *Rig = undefined, // p's
    put: cuda.Function, // tf_glm_put_slot
    fault: Fault = .none,
    out: u64 = 0,
    sy: u64 = 0, // the parent's fp32 rows
    ey: u64 = 0, // put_slot's slot rows
    name: [120]u8 = undefined,
    parent: [64]u8 = undefined,

    pub fn deinit(self: *Slot) void {
        self.p.deinit();
    }

    fn symbol(self: *Slot) []const u8 {
        if (self.p.plain != null) return "glm_mm4_128s2f"; // TF_GLM_MM_FAST4's, its parent glm_mm4_128s2
        return switch (self.p.bits) {
            16 => "glm_mm16_rows",
            4 => "glm_mm4_128s2",
            6 => "glm_mm6_128s2",
            else => "glm_mm8_128s2",
        };
    }

    pub fn kernel(self: *Slot) !g.Kernel {
        self.rig = &self.p.rig;
        const lead = switch (self.fault) {
            .none => "cuda ",
            .short => "control: shared slot rows a slot short ",
            .seventh => "control: shared expert in slot 7 ",
        };
        const name = try std.fmt.bufPrint(&self.name, "{s}{s}_ld shared slot {s}", .{ lead, self.symbol(), if (self.p.wide) "fp32" else "bf16" });
        const parent = try std.fmt.bufPrint(&self.parent, "{s} fp32 then tf_glm_put_slot", .{if (self.p.plain != null) "glm_mm4_128s2" else self.symbol()});
        return .{ .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = if (self.p.wide) .f32 else .bf16,
            .replaces = parent,
            .nan = .{ .stated = "the parent's fp32 rows as put_slot copies or rounds them (NaN as 0x7fff), other slots untouched" },
            .tiles = .{ .{ .size = 128 }, .{ .size = 64, .step = 8, .min = 8 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &served,
        }, .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(Slot); // a container constant: &g.hooks(Slot) in kernel() would point at its frame

    fn eb(self: *Slot) usize {
        return if (self.p.wide) 4 else 2;
    }

    pub fn prepare(self: *Slot, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.p.inputs(shape, fill, seed)) return false;
        const bytes = self.count() * self.eb();
        self.out = try self.p.rig.output(bytes);
        try self.p.rig.zero(self.out, bytes);
        self.sy = try self.p.rig.scratch(shape.rows * shape.n * 4);
        self.ey = try self.p.rig.scratch(bytes);
        return true;
    }

    fn kernels(self: *Slot, parent: bool) glm.mm.Kernels {
        var k = if (parent) (self.p.plain orelse self.p.kernels).* else self.p.kernels.*;
        k.prompt_fixed_k = false;
        return k;
    }

    pub fn run(self: *Slot) !void {
        const sh = self.p.d.?.shape;
        const k = self.kernels(false);
        const slot: usize = if (self.fault == .seventh) slots - 2 else slots - 1;
        const ldo = sh.n * @as(usize, if (self.fault == .short) slots - 1 else slots);
        if (!try glm.mm.matmulStrided(&k, self.p.rig.s, self.p.x, sh.k + 8, self.p.w, self.out + slot * sh.n * self.eb(), ldo, sh.rows, self.p.wide, true)) return error.NotStrided;
        try self.p.rig.s.synchronize();
    }

    /// The prompt call as fill makes it without the switch: the parent's fp32 rows, then tf_glm_put_slot into slot 8.
    pub fn replaced(self: *Slot) ![]const u8 {
        const sh = self.p.d.?.shape;
        const k = self.kernels(true);
        try glm.mm.matmul(&k, self.p.rig.s, self.p.x, sh.k + 8, self.p.w, self.sy, 0, sh.rows, true, true);
        try self.p.rig.zero(self.ey, self.count() * self.eb());
        var a: cuda.Args = .{};
        a.add(self.sy);
        a.add(self.ey);
        for ([_]usize{ sh.rows, sh.n, slots, slots - 1, @intFromBool(!self.p.wide) }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.put, .{ .grid = .{ .x = @intCast((sh.rows * sh.n + 255) / 256) }, .block = .{ .x = 256 } }, self.p.rig.s, &a);
        return self.p.rig.bytes(self.ey, self.count() * self.eb());
    }

    pub fn check(self: *Slot, _: g.Gate) !?[]const u8 {
        const got = try self.p.rig.outputBytes(); // the rig's output copy; replaced() reads into its host copy
        if (!std.mem.eql(u8, got, try self.replaced())) return "differs from the parent's rows put into the slot";
        return null;
    }

    pub fn count(self: *Slot) usize {
        return self.p.count() * slots;
    }

    /// The parent's float64 rows in slot 8 of each row's nine; the other slots 0.
    fn scatter(self: *Slot, compact: []const f64, out: []f64) void {
        const sh = self.p.d.?.shape;
        @memset(out, 0);
        for (0..sh.rows) |r| @memcpy(out[(r * slots + slots - 1) * sh.n ..][0..sh.n], compact[r * sh.n ..][0..sh.n]);
    }

    pub fn reference(self: *Slot, want: []f64, mag: []f64) !void {
        const a = self.p.rig.gpu.gpa;
        const n = self.p.count();
        const tmp = try a.alloc(f64, 2 * n);
        defer a.free(tmp);
        try self.p.reference(tmp[0..n], tmp[n..]);
        self.scatter(tmp[0..n], want);
        self.scatter(tmp[n..], mag);
    }

    pub fn simulate(self: *Slot, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.p.rig.gpu.gpa;
        const tmp = try a.alloc(f64, self.p.count());
        defer a.free(tmp);
        try self.p.simulate(acc, order, tmp);
        self.scatter(tmp, out);
    }
};

/// Every width in fp32 and bf16 slots, fast 4 bits, two controls: 4-bit bf16 rows a slot short, bf16 weights in slot 7.
pub fn cases(out: *[11]Slot, rig: Rig, kernels: *const glm.mm.Kernels, fast: *const glm.mm.Kernels, put: cuda.Function) void {
    const bits = [11]u8{ 16, 4, 6, 8, 16, 4, 6, 8, 4, 16, 4 };
    for (out, bits, 0..) |*x, b, i| x.* = .{
        .p = .{ .rig = rig, .kernels = if (i == 10) fast else kernels, .plain = if (i == 10) kernels else null, .bits = b, .wide = i >= 4 and i != 8 and i != 10 },
        .put = put,
        .fault = if (i == 8) .short else if (i == 9) .seventh else .none,
    };
}
