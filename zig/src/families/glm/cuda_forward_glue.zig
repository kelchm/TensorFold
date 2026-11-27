//! A site's hyper-connection glue: hc_post, then the next site's hc_pre (TF_GLM_HC_FUSE: its partials in post).
const glue = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
const prof_mod = @import("cuda_prof.zig");
const fw = @import("cuda_forward.zig");
const Forward = fw.Forward;
const Rank = fw.Rank;
const Site = Forward.Site;

/// Whether hc_post of `rows` rows writes the next site's hc_pre partials: TF_GLM_HC_FUSE on 4-stream prompt rows.
pub fn fuses(f: *const Forward, rows: usize) bool {
    const c = &f.ranks[0].w.config;
    return f.hc_fuse and f.ranks[0].b.prefill and !c.dsa and c.hc == 4 and c.hidden == 4096 and rows >= glue.hc_prompt_rows;
}

/// forward._pre: hc_pre of every row at a site into b.normed, xs, post and comb; `filled`: post wrote its partials.
pub fn pre(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, rows: usize, filled: bool) !void {
    const stamp = prof_mod.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    if (c.dsa) return glue.rmsnormBy(f.set(), f.stream, b.x, c.hidden, norm, b.normed, c.hidden, b.xs, c.eps, rows, c.hidden);
    if (filled) return glue.hcFinish(f.set(), f.stream, b.x, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows);
    try glue.hcPre(f.set(), f.stream, b.x, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc);
}

/// hc_post of every row; with `next` (the next site's) the partials of its hc_pre too, when the rows fuse.
pub fn post(f: *const Forward, r: Rank, rows: usize, next: ?weights.Hc) !void {
    const stamp = prof_mod.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    if (c.dsa) return glue.residualAdd(f.set(), f.stream, b.x, b.x, b.gathered, rows, c.hidden, r.w.share.world);
    if (next) |h| if (fuses(f, rows)) return glue.hcPostPartial(f.set(), f.stream, b.x, b.x, b.gathered, b.post, b.comb, h.fn_w, b.hcpart, rows, c.hidden, c.hc, r.w.share.world);
    try glue.hcPost(f.set(), f.stream, b.x, b.x, b.gathered, b.post, b.comb, rows, c.hidden, c.hc, r.w.share.world);
}

fn postRows(f: *const Forward, r: Rank, x: u64, post_at: u64, comb_at: u64, n: usize, D: usize, S: usize, g: u64, own: u64) !void {
    if (r.w.config.dsa) {
        if (own != 0) return glue.residualAddOwn(f.set(), f.stream, x, x, g, n, D, r.w.share.world, own, r.w.share.rank);
        return glue.residualAdd(f.set(), f.stream, x, x, g, n, D, r.w.share.world);
    }
    if (own != 0) return glue.hcPostOwn(f.set(), f.stream, x, x, g, post_at, comb_at, n, D, S, r.w.share.world, own, r.w.share.rank);
    return glue.hcPost(f.set(), f.stream, x, x, g, post_at, comb_at, n, D, S, r.w.share.world);
}

/// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
pub fn glueRows(f: *const Forward, r: Rank, j: usize, site: Site, lo: usize, hi: usize, g: u64, own: u64) !void {
    const stamp = prof_mod.span(f.prof, f.stream, .glue);
    defer stamp.end();
    const c = &r.w.config;
    const b = r.b;
    const S: usize = c.hc;
    const D: usize = c.hidden;
    const n = hi - lo;
    const layers = r.w.layers;
    const x = b.x + lo * S * D * 2;
    const post_at = b.post + lo * S * 4;
    const comb_at = b.comb + lo * S * S * 4;
    const Next = struct { hc: weights.Hc, norm: glue.Norm };
    const next: ?Next = if (site == .mixer) .{ .hc = layers[j].ffn_hc, .norm = layers[j].post_norm } else if (j + 1 < layers.len) .{ .hc = layers[j + 1].attn_hc, .norm = layers[j + 1].in_norm } else null;
    const nxt = next orelse {
        try postRows(f, r, x, post_at, comb_at, n, D, S, g, own);
        return glue.streamMean(f.set(), f.stream, x, b.hidden + lo * D * 2, n, D, S);
    };
    const normed = b.normed + lo * D * 2;
    const xs = b.xs + lo * (D / 64) * 4;
    if (c.dsa) {
        try postRows(f, r, x, post_at, comb_at, n, D, S, g, own);
        return glue.rmsnormBy(f.set(), f.stream, x, D, nxt.norm, normed, D, xs, c.eps, n, D);
    }
    if (fuses(f, n)) {
        if (own != 0)
            try glue.hcPostPartialOwn(f.set(), f.stream, x, x, g, post_at, comb_at, nxt.hc.fn_w, b.hcpart, n, D, S, r.w.share.world, own, r.w.share.rank)
        else
            try glue.hcPostPartial(f.set(), f.stream, x, x, g, post_at, comb_at, nxt.hc.fn_w, b.hcpart, n, D, S, r.w.share.world);
        return glue.hcFinish(f.set(), f.stream, x, nxt.hc.base, nxt.hc.scale, nxt.norm, normed, xs, post_at, comb_at, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, n);
    }
    try postRows(f, r, x, post_at, comb_at, n, D, S, g, own);
    try glue.hcPre(f.set(), f.stream, x, nxt.hc.fn_w, nxt.hc.base, nxt.hc.scale, nxt.norm, normed, xs, post_at, comb_at, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, n, D, S);
}
