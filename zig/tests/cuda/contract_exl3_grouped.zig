//! The grouped EXL3 experts under the contract on CUDA: tf_exl3_grouped_* bit-equal to the loops they replace.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_exl3_grouped_ref.zig");
const decode = @import("contract_exl3_decode.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// One projection as Experts.run launches it (one matrix, no Grouped plan): shape.rows token rows, `slots` pairs each.
pub const Grouped = struct {
    rig: rig_.Rig,
    cb: ref.Codebook,
    spec: c.Spec,
    control: bool = false,
    f: cuda.Function, // the served kernel, or a control
    old: cuda.Function, // main's loop, or (GLM's decode kernels) the served tf_exl3_grouped_1
    nt: u32 = 8, // f's 16-column tiles a block (old's: 8)
    hv: [ref.experts]u16, // each expert's half-bits a value
    rows: usize = std.math.maxInt(usize), // the most rows f serves
    fix: ?ref.Fixture = null,
    in: [8]u64 = undefined, // x, tp, k2, uids, count, members; the trellises are inputs of their own
    used: usize = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Grouped) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *Grouped) void {
        if (self.fix) |*f| f.deinit();
        self.fix = null;
        self.rig.reset();
    }

    pub fn kernel(self: *Grouped) g.Kernel {
        return .{ .spec = self.spec, .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Grouped);

    /// The fixture's trellises, rows and picks on the device, with the grouping tf_exl3_group would write.
    pub fn prepare(self: *Grouped, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.Fixture.valid(shape) or shape.rows > self.rows) return false;
        self.drop();
        const a = self.rig.gpu.gpa;
        self.fix = try ref.Fixture.initWidths(a, self.cb, self.hv, shape, fill, seed);
        const f = &self.fix.?;
        var tp: [ref.experts]u64 = undefined;
        for (&tp, f.trellis) |*p, t| p.* = try self.rig.input(std.mem.sliceAsBytes(t));
        var k2: [ref.experts]i32 = undefined;
        for (&k2, self.hv) |*v, h| v.* = h;
        // distinct experts in id order, each one's members row * 32 + slot in row order, -1 after the last
        var uids: [ref.experts]i32 = undefined;
        const members = try a.alloc(i32, ref.experts * shape.rows);
        defer a.free(members);
        @memset(members, -1);
        self.used = 0;
        for (0..ref.experts) |e| {
            var m: usize = 0;
            for (f.pick, 0..) |pe, p| if (pe == e) {
                members[self.used * shape.rows + m] = @intCast(p / ref.slots * 32 + p % ref.slots);
                m += 1;
            };
            if (m == 0) continue;
            uids[self.used] = @intCast(e);
            self.used += 1;
        }
        const ucount = [1]i32{@intCast(self.used)};
        self.in = .{ try self.rig.input(std.mem.sliceAsBytes(f.x)), try self.rig.input(std.mem.sliceAsBytes(&tp)), try self.rig.input(std.mem.sliceAsBytes(&k2)), try self.rig.input(std.mem.sliceAsBytes(uids[0..self.used])), try self.rig.input(std.mem.sliceAsBytes(&ucount)), try self.rig.input(std.mem.sliceAsBytes(members)), 0, 0 };
        self.out = try self.rig.output(self.count() * 4);
        self.base = try self.rig.scratch(self.count() * 4);
        return true;
    }

    /// GroupedArgs in launch.zig add order: one matrix (x0 = x1), standard 0, no skip.
    fn launch(self: *Grouped, f: cuda.Function, nt: u32, z: u64) !void {
        const sh = self.fix.?.shape;
        const sk = self.fix.?.sk;
        var a: cuda.Args = .{};
        for ([_]u64{ self.in[0], self.in[0], self.in[1], self.in[1], self.in[2], self.in[2], self.in[3], self.in[4], self.in[5], z }) |p| a.add(p);
        for ([_]usize{ sh.k, sh.n, sh.rows * ref.slots, sk, sh.rows, ref.slots, ref.experts, 1, 0, 0 }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, -1));
        const mt = (sh.rows + 15) / 16;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.used), .y = @intCast(sh.n / (16 * nt)), .z = @intCast(sk * mt) }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    pub fn run(self: *Grouped) !void {
        try self.launch(self.f, self.nt, self.out);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Grouped) ![]const u8 {
        try self.launch(self.old, 8, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    /// Gate 1: the replaced loop's bytes, and some output non-finite: the planted values.
    pub fn check(self: *Grouped, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from the replaced loop's", .{ at, got.len });
        for (0..got.len / 4) |i| if (c.classOf(c.read(.f32, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *Grouped) usize {
        return self.fix.?.outputs();
    }

    pub fn reference(self: *Grouped, want: []f64, mag: []f64) !void {
        return self.fix.?.reference(want, mag);
    }

    pub fn simulate(self: *Grouped, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.fix.?.simulate(acc, order, out);
    }
};

/// The three codebooks' kernels and two planted faults (mul1) against main's loop, then GLM's decode.cases.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    entries: [5 + decode.cases.len]Grouped,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        errdefer self.mod.unload();
        const served = [3][:0]const u8{ "tf_exl3_grouped_0", "tf_exl3_grouped_1", "tf_exl3_grouped_2" };
        const olds = [3][:0]const u8{ "tf_ctl_exl3_grouped_old_0", "tf_ctl_exl3_grouped_old_1", "tf_ctl_exl3_grouped_old_2" };
        for (self.entries[0..5], 0..) |*x, i| {
            const cb: ref.Codebook = if (i < 3) @fromBackingInt(@intCast(i)) else .mul1;
            const fault: ref.Fault = if (i < 3) .none else if (i == 3) .skip else .stale;
            const f = if (i < 3) try self.mod.function(served[i]) else try controls.function(if (i == 3) "tf_ctl_exl3_grouped_skip" else "tf_ctl_exl3_grouped_stale");
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .cb = cb, .spec = ref.spec(cb, fault), .control = fault != .none, .f = f, .old = try controls.function(olds[@backingInt(cb)]), .hv = ref.halves(cb) };
        }
        for (self.entries[5..], decode.cases) |*x, case| {
            const f = if (case.control) try controls.function(case.symbol) else try self.mod.function(case.symbol);
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .cb = .mcg, .spec = decode.spec(case), .control = case.control, .f = f, .old = try self.mod.function("tf_exl3_grouped_1"), .nt = case.nt, .hv = decode.widths(case), .rows = case.rows };
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
