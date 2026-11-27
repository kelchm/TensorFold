//! GLM's MLA on the latent cache (DSA layers), BF16 row by row: 512 wide, or 576 with full GLM-5.3's roped key columns.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const glue = @import("cuda_glue.zig");

/// latent.CHUNK, KT, HB, HB_WIDE, PROMPT_ROWS, PROMPT_RB.
pub const chunk = 512;
pub const key_tile = 32;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
pub const prompt_rows = 64;
pub const prompt_rb = 64;
/// common/mla.cuh CHUNK_THREADS: the chunk kernels trap unless their eight warps launch together.
pub const chunk_threads = 256;

/// latent.head_block: a rank's 32 heads a program in prompt chunks (each key read once), 16 in decode windows.
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) heads_tile_wide else heads_tile;
}

/// The latent's width, and the roped key columns a full GLM-5.3 row carries past it (cuda.mla's).
pub const latent_width = 512;
pub const rope_width = cuda.mla.rope_width;
pub const roped_row = latent_width + rope_width;

/// headBlock for rows `row` wide: roped rows keep 16 heads, as wide_smem<32, 64>() is past a block's shared memory.
pub fn headBlockFor(rows: usize, row: usize) usize {
    return if (row == roped_row) heads_tile else headBlock(rows);
}

/// The roped key columns of a q and key row: none at 512 (GLM-5.3-Flash), 64 at 576 (full GLM-5.3).
fn ropeOf(row: usize) !usize {
    return switch (row) {
        latent_width => 0,
        roped_row => rope_width,
        else => error.LatentShape,
    };
}

/// The roped image a roped row's call takes.
fn roped(set: glue.Set) !*const cuda.mla.Mla {
    return set.mla orelse error.NoRopedKernels;
}

/// latent.chunks_for: 512-key chunks to cover `length` keys.
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// latent.absorb_q (BF16 kv_b): out [rows][heads][lw + rd] = q [rows][heads][d + rd] . wk, fp32; rd roped columns.
pub fn absorbQ(set: glue.Set, stream: cuda.Stream, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, rd: usize, lw: usize) !void {
    if (rd != 0) {
        if (rd != rope_width or d != cuda.mla.glm_nope) return error.LatentShape;
        return (try roped(set)).absorbQ(stream, q, d + rd, wk, out, lw + rd, rows, heads, d, lw);
    }
    if (d != 256 or lw != 512) return error.LatentShape;
    if (mmaRows(set, rows, .{ q, wk, out })) return headsMma(set.glue.latent_mma_kn, glue.latent_mma_smem[0], stream, q, heads * d, wk, out, heads * lw, rows, heads, lw, d);
    try rowBlocks(set.glue.absorb_q, stream, q, wk, out, rows, heads, lw / 32);
}

/// TF_GLM_ABSORB_MMA=1 on a prompt's aligned rows; decode windows retain the SIMT kernels.
fn mmaRows(set: glue.Set, rows: usize, ptrs: [3]u64) bool {
    if (!set.prompt or !set.absorb_mma or rows < prompt_rows) return false;
    for (ptrs) |p| if (p % 16 != 0) return false;
    return true;
}

fn headsMma(f: cuda.Function, smem: u32, stream: cuda.Stream, x: u64, xr: usize, w: u64, out: u64, orow: usize, rows: usize, heads: usize, n: usize, k: usize) !void {
    if (n % 128 != 0 or k % 32 != 0 or xr % 8 != 0 or orow % 8 != 0) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ x, @as(i64, @intCast(xr)), w, out, @as(i64, @intCast(orow)), @as(i64, @intCast(rows)), @as(c_int, @intCast(n)), @as(c_int, @intCast(k)) }) |v| a.add(v);
    try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast(n / 128), .y = cdiv(rows, 128), .z = @intCast(heads) }, .block = .{ .x = 256 }, .shared = smem }, stream, &a);
}

/// Where rope turns a row's 64 columns: from x + r x_row + h x_head into out + r o_row + h o_head (bf16 elements).
pub const RopeRows = cuda.mla.RopeRows;

/// The GPT-J pairs of `heads` heads a row at device i32 pos + r, fp32 frequencies `inv` [32] (no YaRN factor).
pub fn rope(set: glue.Set, stream: cuda.Stream, v: RopeRows, rows: usize, heads: usize, pos: u64, inv: u64) !void {
    if (inv == 0) return error.NoRopeFreqs;
    try (try roped(set)).rope(stream, v, rows, heads, pos, inv, 1);
}

/// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v (BF16 kv_b): out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw].
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (dv != 256 or lw != 512) return error.LatentShape;
    if (mmaRows(set, rows, .{ ol, wv, out })) return headsMma(set.glue.latent_mma_nk, glue.latent_mma_smem[1], stream, ol, heads * lw, wv, out, heads * dv, rows, heads, dv, lw);
    try rowBlocks(set.glue.expand_v, stream, ol, wv, out, rows, heads, dv / 16);
}

/// A latent cache holds BF16 rows, or E4M3 rows followed by one FP32 scale per row.
pub fn cacheBytes(capacity: usize, lw: usize, kv8: bool) !usize {
    if (kv8 and lw != 512) return error.LatentShape;
    const row = if (kv8) try std.math.add(usize, lw, 4) else try std.math.mul(usize, lw, 2);
    return std.math.mul(usize, capacity, row);
}

/// Page table `pt` (`ps`, `page_rows`), or 0 in order; `logical` keeps sparse lists as positions (paged E4M3).
pub const Table = struct { pt: u64 = 0, ps: u64 = 0, page_rows: u64 = 0, logical: bool = false };

/// BF16 rows through `t`, or E4M3 rows and their scales (scales nonzero), in order or on pages through `t`.
pub fn latentWriteFormat(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, scales: u64, t: Table, pos: u64, rows: usize, lw: usize) !void {
    if (scales == 0) return latentWrite(set, stream, lat, lat_stride, cache, t, pos, rows, lw);
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, scales, pos }) |v| a.add(v);
    if (t.pt != 0) {
        a.add(t.pt);
        a.add(@as(i64, @intCast(t.ps)));
    }
    const f = if (t.pt != 0) set.glue.lat_write_fp8_paged else set.glue.lat_write_fp8;
    try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, stream, &a);
}

/// latent.latent_write: rows of `lat` (stride `lat_stride`) into slots pos .. pos + rows - 1 (pos an i32 on device).
pub fn latentWrite(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, t: Table, pos: u64, rows: usize, lw: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, pos, @as(i64, @intCast(lw)), t.pt, @as(i64, @intCast(t.ps)) }) |v| a.add(v);
    try cuda_pdl.launch(set.glue.lat_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The chunk partials of one attention call: po fp32 [nch][rows][heads][512], pm and pl fp32 [nch][rows][heads].
pub const Partials = struct {
    po: u64,
    pm: u64,
    pl: u64,
    rows: usize = 0, // the nch x rows they hold (0: every call fits); a sparse call past them merges in the kernel
};

/// latent.attention: causal qa over the cache in `nch` 512-key chunks into out bf16; `dummy`: the merge's count i32.
pub fn attention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, t: Table, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    return attentionFormat(set, stream, qa, cache, 0, t, pos, p, out, dummy, rows, heads, lw, nch, scale, hb);
}

pub fn attentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    try chunks(set, stream, qa, cache, scales, t, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (keys through pos + r) and _sparse_chunks (row r's `counts` of `tokens`); `lw` 512 or 576.
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    const rope_cols = try ropeOf(lw);
    if (rope_cols != 0) {
        if (scales != 0) return error.LatentShape; // the E4M3 cache holds the latent alone
        if (hb != heads_tile) return error.HeadBlock;
        const on: cuda.mla.On = .{ .pt = t.pt, .ps = t.ps, .tokens = tokens, .counts = counts, .w = w, .sparse = sparse };
        return (try roped(set)).chunksOn(stream, qa, cache, pos, on, .{ .po = p.po, .pm = p.pm, .pl = p.pl }, rows, heads, nch, scale);
    }
    const kv8 = scales != 0;
    const paged8 = kv8 and t.pt != 0;
    // a paged sparse list's units are the kernel's: positions for E4M3 (Table.logical), the BF16 rows otherwise
    if (sparse and t.pt != 0 and t.logical != paged8) return error.TokenUnits;
    const f = switch (hb) {
        heads_tile => if (!kv8) set.glue.chunks_16 else if (paged8) set.glue.chunks_16_fp8_paged else set.glue.chunks_16_fp8,
        heads_tile_wide => if (!kv8) set.glue.chunks_32 else if (paged8) set.glue.chunks_32_fp8_paged else set.glue.chunks_32_fp8,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (kv8) a.add(scales);
    inline for (.{ pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    if (!kv8 or paged8) {
        a.add(t.pt);
        a.add(@as(i64, @intCast(t.ps)));
    }
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = chunk_threads }, .shared = shared }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, t: Table, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    return sparseAttentionFormat(set, stream, qa, cache, 0, t, tokens, counts, w, p, out, rows, heads, lw, scale);
}

pub fn sparseAttentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = chunksFor(w);
    const hb = if (!set.prompt and set.verify_tiles.wide(rows, heads, lw, w, scales, t.pt, t.logical)) heads_tile_wide else headBlockFor(rows, lw);
    if (fuses(p, nch, rows, lw)) return fused(set, stream, qa, cache, scales, t, tokens, counts, w, out, rows, heads, lw, nch, scale, hb);
    try chunks(set, stream, qa, cache, scales, t, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, hb, true);
    try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
}

/// Whether a sparse call of `rows` rows over `nch` chunks merges in its chunk kernel: its partials would not fit `p`.
pub fn fuses(p: Partials, nch: usize, rows: usize, lw: usize) bool {
    return lw == latent_width and p.rows != 0 and nch * rows > p.rows;
}

/// tf_glm_chunks_*_fused: block (row, hb heads) over the row's nch chunks, merged as tf_glm_merge merges them.
fn fused(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, tokens: u64, counts: u64, w: usize, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    if (lw != latent_width) return error.LatentShape;
    const kv8 = scales != 0;
    const paged8 = kv8 and t.pt != 0;
    if (t.pt != 0 and t.logical != paged8) return error.TokenUnits;
    const f = switch (hb) {
        heads_tile => if (!kv8) set.glue.chunks_16_fused else if (paged8) set.glue.chunks_16_fp8_paged_fused else set.glue.chunks_16_fp8_fused,
        heads_tile_wide => if (!kv8) set.glue.chunks_32_fused else if (paged8) set.glue.chunks_32_fp8_paged_fused else set.glue.chunks_32_fp8_fused,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (kv8) a.add(scales);
    inline for (.{ tokens, counts, out, @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intCast(nch)) }) |v| a.add(v);
    if (paged8) {
        a.add(t.pt);
        a.add(@as(i64, @intCast(t.ps)));
    }
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb) }, .block = .{ .x = chunk_threads }, .shared = shared }, stream, &a);
}

/// The partials and out hold the latent alone (the roped columns score, the values are the latent's).
fn merge(set: glue.Set, stream: cuda.Stream, p: Partials, out: u64, counts: u64, rows: usize, heads: usize, lw: usize, nch: usize, sparse: bool) !void {
    _ = try ropeOf(lw);
    var a: cuda.Args = .{};
    inline for (.{ p.po, p.pm, p.pl, out, counts, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    try cuda_pdl.launch(set.glue.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &a);
}

test "GLM MLA: a sparse call merges in its chunk kernel only past its partials' rows and only at lw 512" {
    const p: Partials = .{ .po = 1, .pm = 2, .pl = 3, .rows = 13 * 512 }; // a 4,096-row prompt buffer's dense calls
    try std.testing.expect(!fuses(p, chunksFor(2051), 1331, 512));
    try std.testing.expect(fuses(p, chunksFor(2051), 1332, 512));
    try std.testing.expect(fuses(p, chunksFor(2051), 4096, 512));
    try std.testing.expect(!fuses(p, chunksFor(2051), 4096, roped_row)); // roped full GLM-5.3 keeps cuda.mla
    try std.testing.expect(!fuses(.{ .po = 1, .pm = 2, .pl = 3 }, chunksFor(2051), 4096, 512)); // rows 0: all fit
}

test "latent.py's tiling rules" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(8));
    try std.testing.expectEqual(@as(usize, 32), headBlock(64));
    try std.testing.expectEqual(@as(usize, 3), chunksFor(1025));
    try std.testing.expectEqual(@as(usize, 1), chunksFor(512));
}
