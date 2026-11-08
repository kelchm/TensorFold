//! Packed backbone linears from cuda_tile.layerLinears and the KDA, DSA, MLP and shared-expert call sites.
const std = @import("std");
const glm = @import("glm_cuda");

pub const Group = enum { kda, mla, shared, dense };
pub const rows = [_]usize{ 2048, 4096, 8192 };

pub const Shape = struct {
    name: []const u8,
    group: Group,
    n: usize,
    k: usize,
    layers: usize,
    stride: usize = 0,
    offset: usize = 0,
    wide: bool = false,

    pub fn ldx(s: Shape) usize {
        return if (s.stride == 0) s.k else s.stride;
    }

    pub fn work(s: Shape, m: usize) f64 {
        return @as(f64, @floatFromInt(2 * m * s.n * s.k)) / 1e9;
    }
};

/// Share.of is the CUDA loader's partitioner; indexer heads and the two low-rank inputs stay replicated.
pub fn list() ![13]Shape {
    const c = glm.config.Config{};
    const share = try glm.shard.Share.of(&c, 0, 2);
    const width = glm.kda_block.rowWidth(share.kda.len());
    const kd = share.kda.len() * c.kda_dim;
    const nk = c.countKind(.kda);
    const nm = c.countKind(.mla);
    return .{
        .{ .name = "kda-qkv-fa-ga-b", .group = .kda, .n = width, .k = c.hidden, .layers = nk },
        .{ .name = "kda-fb", .group = .kda, .n = kd, .k = c.kda_dim, .layers = nk, .stride = width, .offset = 3 * kd },
        .{ .name = "kda-gb", .group = .kda, .n = kd, .k = c.kda_dim, .layers = nk, .stride = width, .offset = 3 * kd + c.kda_dim },
        .{ .name = "kda-o", .group = .kda, .n = c.hidden, .k = kd, .layers = nk, .wide = true },
        .{ .name = "mla-qa-kva", .group = .mla, .n = c.q_lora + c.kv_lora, .k = c.hidden, .layers = nm },
        .{ .name = "mla-qb", .group = .mla, .n = share.mla.len() * c.nope, .k = c.q_lora, .layers = nm },
        .{ .name = "mla-index-kw-weights", .group = .mla, .n = c.i_dim + c.i_heads, .k = c.hidden, .layers = nm },
        .{ .name = "mla-index-qb", .group = .mla, .n = c.i_heads * c.i_dim, .k = c.q_lora, .layers = nm },
        .{ .name = "mla-o", .group = .mla, .n = c.hidden, .k = share.mla.len() * c.v_dim, .layers = nm, .wide = true },
        .{ .name = "shared-gu", .group = .shared, .n = 2 * share.shared.len(), .k = c.hidden, .layers = c.layers - c.dense_layers },
        .{ .name = "shared-down", .group = .shared, .n = c.hidden, .k = share.shared.len(), .layers = c.layers - c.dense_layers, .wide = true },
        .{ .name = "dense-gu", .group = .dense, .n = 2 * share.dense.len(), .k = c.hidden, .layers = c.dense_layers },
        .{ .name = "dense-down", .group = .dense, .n = c.hidden, .k = share.dense.len(), .layers = c.dense_layers, .wide = true },
    };
}

pub fn print(shapes: []const Shape) void {
    for (shapes) |s| std.debug.print("GLM_PROMPT_MM shape={s} N={d} K={d} layers={d} calls_per_layer=1 ldx={d} x_offset={d} output={s}\n", .{ s.name, s.n, s.k, s.layers, s.ldx(), s.offset, if (s.wide) "fp32" else "bf16" });
}
