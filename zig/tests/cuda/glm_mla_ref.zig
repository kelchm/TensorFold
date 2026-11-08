//! Independent FP64 head projections for MLA: the packed device arrays and the per-head dense references.
const std = @import("std");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
pub const Kind = enum {
    absorb,
    expand,
    pub fn n(kind: Kind) usize {
        return if (kind == .absorb) 512 else 256;
    }
    pub fn k(kind: Kind) usize {
        return if (kind == .absorb) 256 else 512;
    }
};

pub const Fixture = struct {
    a: std.mem.Allocator,
    kind: Kind,
    shape: c.Shape,
    dots: []Dense,
    x: []u16,
    w: []u16,

    pub fn init(a: std.mem.Allocator, kind: Kind, shape: c.Shape, fill: g.Fill, seed: u64) !Fixture {
        const n = kind.n();
        const k = kind.k();
        if (shape.n % n != 0 or shape.n == 0 or shape.k != k) return error.LatentShape;
        const heads = shape.n / n;
        const dots = try a.alloc(Dense, heads);
        errdefer a.free(dots);
        var made: usize = 0;
        errdefer for (dots[0..made]) |d| d.deinit();
        const x = try a.alloc(u16, shape.rows * heads * k);
        errdefer a.free(x);
        const w = try a.alloc(u16, heads * n * k);
        errdefer a.free(w);
        for (dots, 0..) |*d, h| {
            d.* = try Dense.init(a, .{ .rows = shape.rows, .n = n, .k = k }, 1);
            made += 1;
            d.fillX(fill, seed +% h, 16);
            var prng = std.Random.DefaultPrng.init(seed +% h +% 317);
            const rng = prng.random();
            if (fill == .stress) for (0..shape.rows) |r| for (0..k / 2) |i| {
                d.x[r * k + k / 2 + i] = c.toBf16(c.fromBf16(d.x[r * k + i]) + if (i == 0) @as(f32, 16) else 0);
            };
            for (0..n) |j| for (0..k) |i| {
                const v: f32 = if (fill == .stress) @as(f32, @floatFromInt(32 * (1 + ((i % (k / 2)) * 7 + j + h) % 6))) * (if (i < k / 2) @as(f32, 1) else -1) else rng.float(f32) / 4 - 0.125;
                const bits = c.toBf16(v);
                d.w[j * k + i] = c.fromBf16(bits);
                w[h * n * k + (if (kind == .absorb) i * n + j else j * k + i)] = bits;
            };
            for (0..shape.rows) |r| @memcpy(x[(r * heads + h) * k ..][0..k], d.x[r * k ..][0..k]);
        }
        return .{ .a = a, .kind = kind, .shape = shape, .dots = dots, .x = x, .w = w };
    }

    pub fn deinit(self: Fixture) void {
        for (self.dots) |d| d.deinit();
        self.a.free(self.dots);
        self.a.free(self.x);
        self.a.free(self.w);
    }
    pub fn count(self: Fixture) usize {
        return self.shape.rows * self.shape.n;
    }
    fn scatter(self: Fixture, h: usize, src: []const f64, dst: []f64) void {
        const n = self.kind.n();
        for (0..self.shape.rows) |r| @memcpy(dst[r * self.shape.n + h * n ..][0..n], src[r * n ..][0..n]);
    }
    pub fn reference(self: Fixture, want: []f64, mag: []f64) !void {
        const tmp = try self.a.alloc(f64, self.shape.rows * self.kind.n() * 2);
        defer self.a.free(tmp);
        const half = tmp.len / 2;
        for (self.dots, 0..) |d, h| {
            try d.reference(tmp[0..half], tmp[half..]);
            self.scatter(h, tmp[0..half], want);
            self.scatter(h, tmp[half..], mag);
        }
    }
    pub fn simulate(self: Fixture, acc: c.Accum, order: g.Order, out: []f64) !void {
        const tmp = try self.a.alloc(f64, self.shape.rows * self.kind.n());
        defer self.a.free(tmp);
        for (self.dots, 0..) |d, h| {
            try d.simulate(acc, order, tmp);
            for (tmp) |*v| v.* = c.fromBf16(c.toBf16(@floatCast(v.*)));
            self.scatter(h, tmp, out);
        }
    }
};

test "GLM MLA: each projection stress distinguishes fp16 and bf16 sums in every accumulator order" {
    const t = std.testing;
    for (std.enums.values(Kind)) |kind| {
        const fix = try Fixture.init(t.allocator, kind, .{ .rows = 1, .n = 2 * kind.n(), .k = kind.k() }, .stress, 17);
        defer fix.deinit();
        const all = try t.allocator.alloc(f64, fix.count() * 3);
        defer t.allocator.free(all);
        const want = all[0..fix.count()];
        const mag = all[fix.count()..][0..fix.count()];
        const got = all[2 * fix.count() ..];
        try fix.reference(want, mag);
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            try fix.simulate(.f32, order, got);
            for (got, want) |v, w| try t.expectEqual(c.fromBf16(c.toBf16(@floatCast(w))), v);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try fix.simulate(acc, order, got);
                var most: f64 = 0;
                for (got, want) |v, w| most = @max(most, if (std.math.isFinite(v)) @abs(v - w) / (@abs(w) * 0x1p-8 + 1e-30) else std.math.inf(f64));
                try t.expect(most >= 100);
            }
        }
    }
}
