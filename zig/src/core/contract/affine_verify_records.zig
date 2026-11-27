//! The wide-row-tile affine launches: nvcc 13.0's registers on sm_80/86/89/120/121, every tile 128 threads a block.
const R = @import("registry.zig").CudaRecord;
const symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi5ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const symbol64 = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const symbol128 = "_ZN12tf_qmm_group12group_kernelILi64ELi128ELi64ELi1ELi4ELi3ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const symbol32_f32 = "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi5ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const symbol64_f32 = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
pub const records = [_]R{
    .{ .launch = .{ .name = "affine4 verify32 one CTA a tile", .symbol = symbol, .image = "qmm_group", .threads = 128, .dynamic = 32640, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 79, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify32 cluster of 8", .symbol = symbol, .image = "qmm_group", .threads = 128, .dynamic = 32640, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = .{ .{ .regs = 79, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify32 fp32 one CTA a tile", .symbol = symbol32_f32, .image = "qmm_group", .threads = 128, .dynamic = 32640, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 79, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify32 fp32 cluster of 8", .symbol = symbol32_f32, .image = "qmm_group", .threads = 128, .dynamic = 32640, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = .{ .{ .regs = 79, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify64 one CTA a tile", .symbol = symbol64, .image = "qmm_group", .threads = 128, .dynamic = 43008, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify64 cluster of 8", .symbol = symbol64, .image = "qmm_group", .threads = 128, .dynamic = 43008, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = .{ .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify64 fp32 one CTA a tile", .symbol = symbol64_f32, .image = "qmm_group", .threads = 128, .dynamic = 43008, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify64 fp32 cluster of 8", .symbol = symbol64_f32, .image = "qmm_group", .threads = 128, .dynamic = 43008, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = .{ .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify128 one CTA a tile", .symbol = symbol128, .image = "qmm_group", .threads = 128, .dynamic = 57600, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 240, .static = 0, .bound = 128 }, .{ .regs = 240, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "affine4 verify128 cluster of 8", .symbol = symbol128, .image = "qmm_group", .threads = 128, .dynamic = 57600, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 240, .static = 0, .bound = 128 }, .{ .regs = 240, .static = 0, .bound = 128 } } },
};

test "affine4 verify records (including FP32 wide tiles) fit every SM on nvcc 13.0 and 13.3" {
    const std = @import("std");
    const registry = @import("registry.zig");
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, false));
    for (records) |r| {
        if (std.mem.indexOf(u8, r.launch.name, "verify32 fp32") != null) {
            try std.testing.expectEqual(@as(u32, 92), registry.usageFor(r, .{ .major = 13, .minor = 0 })[4].regs);
            try std.testing.expectEqual(@as(u32, 72), registry.usageFor(r, .{ .major = 13, .minor = 3 })[4].regs);
        }
        if (std.mem.indexOf(u8, r.launch.name, "verify64 fp32") != null) {
            try std.testing.expectEqual(@as(u32, 124), registry.usageFor(r, .{ .major = 13, .minor = 0 })[4].regs);
            try std.testing.expectEqual(@as(u32, 125), registry.usageFor(r, .{ .major = 13, .minor = 3 })[4].regs);
        }
    }
}
