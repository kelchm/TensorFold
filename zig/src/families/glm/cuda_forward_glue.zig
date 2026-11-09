//! A site's hyper-connection glue: hc_post, then the next site's hc_pre (TF_GLM_HC_FUSE: its partials in post).
const glue = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
const prof_mod = @import("cuda_prof.zig");
const fw = @import("cuda_forward.zig");
const Forward = fw.Forward;
const Rank = fw.Rank;
const Site = Forward.Site;

/// Whether hc_post of `rows` rows writes the next site's hc_pre partials: TF_GLM_HC_FUSE on a prompt call's rows.
pub fn fuses(f: *const Forward, rows: usize) bool {
    return f.hc_fuse and f.ranks[0].b.prefill and rows >= glue.hc_prompt_rows;
}

/// forward._pre: hc_pre of every row at a site into b.normed, xs, post and comb; `filled`: post wrote its partials.
pub fn pre(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, rows: usize, filled: bool) !void {
    const stamp = prof_mod.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    if (filled) return glue.hcFinish(f.set(), f.stream, b.x, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows);
    try glue.hcPre(f.set(), f.stream, b.x, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc);
}

/// hc_post of every row; with `next` (the next site's) the partials of its hc_pre too, when the rows fuse.
pub fn post(f: *const Forward, r: Rank, rows: usize, next: ?weights.Hc) !void {
    const stamp = prof_mod.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    if (next) |h| if (fuses(f, rows)) return glue.hcPostPartial(f.set(), f.stream, b.x, b.x, b.gathered, b.post, b.comb, h.fn_w, b.hcpart, rows, c.hidden, c.hc, r.w.share.world);
    try glue.hcPost(f.set(), f.stream, b.x, b.x, b.gathered, b.post, b.comb, rows, c.hidden, c.hc, r.w.share.world);
}

/// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
pub fn glueRows(f: *const Forward, r: Rank, j: usize, site: Site, lo: usize, hi: usize, g: u64) !void {
    const stamp = prof_mod.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    const S: usize = c.hc;
    const D: usize = c.hidden;
    const layers = r.w.layers;
    const x = b.x + lo * S * D * 2;
    const post_at = b.post + lo * S * 4;
    const comb_at = b.comb + lo * S * S * 4;
    const Next = struct { hc: weights.Hc, norm: glue.Norm };
    const next: ?Next = if (site == .mixer) .{ .hc = layers[j].ffn_hc, .norm = layers[j].post_norm } else if (j + 1 < layers.len) .{ .hc = layers[j + 1].attn_hc, .norm = layers[j + 1].in_norm } else null;
    const n = next orelse {
        try glue.hcPost(f.set(), f.stream, x, x, g, post_at, comb_at, hi - lo, D, S, r.w.share.world);
        return glue.streamMean(f.set(), f.stream, x, b.hidden + lo * D * 2, hi - lo, D, S);
    };
    const normed = b.normed + lo * D * 2;
    const xs = b.xs + lo * (D / 64) * 4;
    if (fuses(f, hi - lo)) {
        try glue.hcPostPartial(f.set(), f.stream, x, x, g, post_at, comb_at, n.hc.fn_w, b.hcpart, hi - lo, D, S, r.w.share.world);
        return glue.hcFinish(f.set(), f.stream, x, n.hc.base, n.hc.scale, n.norm, normed, xs, post_at, comb_at, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo);
    }
    try glue.hcPost(f.set(), f.stream, x, x, g, post_at, comb_at, hi - lo, D, S, r.w.share.world);
    try glue.hcPre(f.set(), f.stream, x, n.hc.fn_w, n.hc.base, n.hc.scale, n.norm, normed, xs, post_at, comb_at, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S);
}
