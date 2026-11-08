//! Independent constant-weight stress reference for the GLM prompt projections and their rounded epilogues.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const rows = core.contract.rows;
pub const hs: f64 = @as(f32, 0.08838834764831845);
pub const scale: f64 = 0x1p-16;
pub const input: f64 = 4096;
pub const weight: f64 = @as(f64, core.exl3_format.codebookValue(.mcg, 0xffff));

pub fn dot(a: std.mem.Allocator, k: usize, acc: ?c.Accum, order: rows.Order) !f64 {
    if (acc) |kind| {
        const terms = try a.alloc(f64, k);
        defer a.free(terms);
        @memset(terms, weight * input);
        return rows.sumIn(kind, order, terms);
    }
    return weight * input * @as(f64, @floatFromInt(k));
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Constant columns rotate to one nonzero column per 128; gate/up then rotates that activation back to every column.
pub fn output(sum: f64, gateup: bool, act_mode: i32, col: usize) f64 {
    const rotated = sum * 128 * hs;
    if (!gateup) return if (col % 128 == 0) rotated else 0;
    const g = if (act_mode == 0) bf(rotated * scale) else rotated * scale;
    var act = g / (1 + @exp(-g));
    if (act_mode == 0) act = bf(act);
    act *= g;
    if (act_mode == 0) act = bf(act);
    return act * scale * hs;
}

test "GLM prompt stress: fp16 overflows and bf16 misses after both epilogues in all accumulator orders" {
    const a = std.testing.allocator;
    const k = 4096;
    const exact = try dot(a, k, null, .sequential);
    for ([_]bool{ false, true }) |gu| for ([_]i32{ 0, 1 }) |act| {
        const want = output(exact, gu, act, 0);
        try std.testing.expect(std.math.isFinite(want) and want > 0);
        for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
            try std.testing.expectEqual(exact, try dot(a, k, .f32, order));
            const half = output(try dot(a, k, .f16, order), gu, act, 0);
            try std.testing.expect(!std.math.isFinite(half));
            const narrow = output(try dot(a, k, .bf16, order), gu, act, 0);
            const relative = @abs(narrow - want) / @abs(want);
            if (order != .pairwise) try std.testing.expect(relative > switch (order) {
                .sequential => @as(f64, 0.1),
                .strided => 0.01,
                .pairwise => 0.001,
            });
        }
    };
}
