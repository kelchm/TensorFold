//! Opt-in SM121 prompt projections for the non-roped GLM-5.3-Flash TP2 shape.
const cuda = @import("cuda");
const pdl = @import("cuda_pdl.zig");
pub const Kernels = struct {
    mod: cuda.Module,
    kn: cuda.Function,
    nk: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.glm_prompt_mma);
        errdefer mod.unload();
        const kn = try mod.function("tf_glm_latent_mma_kn");
        const nk = try mod.function("tf_glm_latent_mma_nk");
        try kn.allowDynamicShared(56832);
        try nk.allowDynamicShared(61440);
        return .{ .mod = mod, .kn = kn, .nk = nk };
    }

    pub fn eligible(_: *const Kernels, rows: usize, ptrs: [3]u64) bool {
        if (rows < 64) return false;
        for (ptrs) |p| if (p % 16 != 0) return false;
        return true;
    }

    pub fn project(m: *const Kernels, kind: enum { kn, nk }, stream: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, heads: usize) !void {
        const n: c_int = if (kind == .kn) 512 else 256;
        const k: c_int = if (kind == .kn) 256 else 512;
        const h: i64 = @intCast(heads);
        var args: cuda.Args = .{};
        inline for (.{ x, h * k, w, out, h * n, @as(i64, @intCast(rows)), n, k }) |v| args.add(v);
        try pdl.launch(if (kind == .kn) m.kn else m.nk, .{ .grid = .{ .x = @intCast(@divExact(n, 128)), .y = @intCast((rows + 127) / 128), .z = @intCast(heads) }, .block = .{ .x = 256 }, .shared = if (kind == .kn) 56832 else 61440 }, stream, &args);
    }
};
