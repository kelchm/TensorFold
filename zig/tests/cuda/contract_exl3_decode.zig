//! GLM's EXL3 decode-window grouped kernels under the contract: each bit-equal to tf_exl3_grouped_1, its faults red.
const std = @import("std");
const core = @import("core");
const ref = @import("contract_exl3_grouped_ref.zig");

const c = core.contract.spec;

/// One registered case: a kernel of exl3_experts or a planted fault of contract_controls, against tf_exl3_grouped_1.
pub const Case = struct {
    symbol: [:0]const u8,
    name: []const u8,
    control: bool = false,
    nt: u32 = 4, // 16-column tiles a block
    four: bool = true, // every expert 4-bit: the one-width kernels trap on any other width
    rows: usize = 8, // the most rows of a window it serves
};

pub const cases = [_]Case{
    .{ .symbol = "tf_exl3_grouped_1_k8r8", .name = "cuda exl3 decode tf_exl3_grouped_1_k8r8 (4-bit mcg, 1-8 rows)" },
    .{ .symbol = "tf_exl3_grouped_1_nt4", .name = "cuda exl3 decode tf_exl3_grouped_1_nt4 (mcg, 4 tiles, 1-32 rows)", .four = false, .rows = 32 },
    .{ .symbol = "tf_ctl_exl3_decode_order", .name = "control: exl3 decode k8r8 sums its warps last to first", .control = true },
    .{ .symbol = "tf_exl3_grouped_1_k8r8s", .name = "cuda exl3 decode tf_exl3_grouped_1_k8r8s (4-bit mcg read once, 1-8 rows)" },
    .{ .symbol = "tf_exl3_grouped_1_k8r16s", .name = "cuda exl3 decode tf_exl3_grouped_1_k8r16s (4-bit mcg read once, 1-32 rows)", .rows = 32 },
    .{ .symbol = "tf_ctl_exl3_decode_shfl_src", .name = "control: exl3 decode read-once takes a last-tile word from the next lane", .control = true },
    .{ .symbol = "tf_ctl_exl3_decode_shfl_a", .name = "control: exl3 decode read-once gives the last tile the A fragment before", .control = true },
    .{ .symbol = "tf_exl3_grouped_1_k8r8p", .name = "cuda exl3 decode tf_exl3_grouped_1_k8r8p (k8r8s that waits, plain launch, 1-8 rows)" },
    .{ .symbol = "tf_exl3_grouped_1_k8r16p", .name = "cuda exl3 decode tf_exl3_grouped_1_k8r16p (k8r16s that waits, plain launch, 1-32 rows)", .rows = 32 },
};

/// rows, n, k: 3, 2, 1 k tiles a warp; GLM-5.3-Flash's gate/up and down on two ranks and one, at decode windows
const served_8 = [_]c.Shape{
    .{ .rows = 4, .n = 128, .k = 768 },   .{ .rows = 4, .n = 128, .k = 512 },   .{ .rows = 4, .n = 128, .k = 256 },
    .{ .rows = 8, .n = 1024, .k = 4096 }, .{ .rows = 5, .n = 4096, .k = 1024 }, .{ .rows = 1, .n = 2048, .k = 4096 },
    .{ .rows = 3, .n = 4096, .k = 2048 },
};
const served_32 = served_8 ++ [_]c.Shape{
    .{ .rows = 32, .n = 1024, .k = 4096 }, .{ .rows = 17, .n = 4096, .k = 1024 }, .{ .rows = 9, .n = 2048, .k = 4096 },
    .{ .rows = 16, .n = 4096, .k = 2048 },
};

/// The grouped spec at the case's windows, replacing the served tf_exl3_grouped_1.
pub fn spec(case: Case) c.Spec {
    var s = ref.spec(.mcg, .none);
    s.name = case.name;
    s.replaces = "tf_exl3_grouped_1 (the served kernel)";
    s.served = if (case.rows <= 8) &served_8 else &served_32;
    s.tiles[0] = .{ .size = if (case.rows <= 8) 8 else 16 };
    return s;
}

/// Each fixture expert's half-bits a value.
pub fn widths(case: Case) [ref.experts]u16 {
    return if (case.four) @splat(8) else ref.halves(.mcg);
}

test "exl3 decode contract: every served shape is a window the case serves, and the filter word names every case" {
    for (cases) |case| {
        const s = spec(case);
        try std.testing.expect(std.mem.indexOf(u8, s.name, "exl3 decode") != null);
        for (s.served) |x| try std.testing.expect(ref.Fixture.valid(x) and x.rows <= case.rows and (x.n / 16) % case.nt == 0);
    }
}

test "exl3 decode contract: gate 4's stress rows hold fp32 replays and make fp16 and bf16 sums miss at 4 bits" {
    const gates = core.contract.gates;
    const was = gates.quiet;
    gates.quiet = true;
    defer gates.quiet = was;
    for (cases) |case| {
        if (case.control) continue;
        var h: ref.HostSums = .{ .a = std.testing.allocator, .cb = .mcg, .hv = widths(case) };
        defer {
            if (h.f) |*f| f.deinit();
            std.testing.allocator.free(h.out);
        }
        const k: gates.Kernel = .{ .spec = spec(case), .ctx = &h, .vt = &ref.HostSums.vt };
        try std.testing.expectEqual(gates.Verdict.pass, try gates.sums(std.testing.allocator, k, std.hash.Wyhash.hash(0x6b63, k.spec.name)));
    }
}
