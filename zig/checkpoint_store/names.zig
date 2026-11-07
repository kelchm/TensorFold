//! Reuses TensorFold rows_chain token/span/chunk naming; binds media content before each completed page.
const std = @import("std");
const glm = @import("glm");
const f = @import("format.zig");

pub const Input = struct {
    tokens: []const u32,
    spans: []const [2]u32,
    starts: []const u32,
};

/// GLM prompt kernels depend on planned chunk ends and canonical decode spans (native/glm_host.zig).
pub fn rules(metadata: *const glm.checkpoint_restore.Metadata) f.chain.Keying {
    return .{ .identity = 0x303639304d4c47, .block = glm.state.page_rows, .planned = true, .lookahead = @intFromBool(metadata.shape.flags & 2 != 0) };
}

fn validate(input: Input, metadata: *const glm.checkpoint_restore.Metadata) !void {
    try metadata.validate();
    const through = metadata.mark + rules(metadata).lookahead;
    if (input.tokens.len < through or input.tokens.len > f.max_pages * glm.state.page_rows + 1) return error.BlockIdentity;
    var end: u32 = 0;
    for (input.spans, 0..) |span, i| {
        if ((i != 0 and span[0] <= end) or span[1] <= span[0] or span[1] > metadata.mark) return error.BlockIdentity;
        end = span[1];
    }
    end = 0;
    for (input.starts, 0..) |start, i| {
        if ((i != 0 and start <= end) or start > metadata.mark) return error.BlockIdentity;
        end = start;
    }
    if (input.starts.len == 0 or input.starts[input.starts.len - 1] != metadata.mark) return error.BlockIdentity;
}

/// Binds the agreement to precisely the consumed tokens, arithmetic cuts, and head lookahead, never future tokens.
pub fn prefix(input: Input, metadata: *const glm.checkpoint_restore.Metadata) !f.Name {
    try validate(input, metadata);
    const through: usize = @intCast(metadata.mark + rules(metadata).lookahead);
    return f.chain.state(rules(metadata), input.tokens[0..through], input.spans, input.starts, @intCast(metadata.mark));
}

/// Inputs must come from the paused sequence owner; explicit empty spans assert that every row used prompt arithmetic.
pub fn build(input: Input, metadata: *const glm.checkpoint_restore.Metadata, out: []f.Name) !void {
    try validate(input, metadata);
    if (out.len > f.max_pages or out.len * glm.state.page_rows > metadata.mark) return error.BlockIdentity;
    f.chain.blocks(rules(metadata), input.tokens, input.spans, input.starts, @intCast(metadata.mark), 0, @intCast(out.len), out);
    for (out, 0..) |*name, i| {
        var h = f.Sha.init(.{});
        h.update("GLM rows media prefix 0960 v2");
        h.update(name);
        const through = (i + 1) * glm.state.page_rows + rules(metadata).lookahead;
        for (metadata.media.images[0..metadata.media.count]) |im| {
            if (im.at >= through) break;
            h.update(&metadata.media.processor);
            h.update(&f.wire.hash(im));
        }
        name.* = h.finalResult();
    }
}

/// Local geometry and TP identity separate files even when token rows happen to be identical.
pub fn context(a: f.epoch.Agreement, rank: u32, metadata: *const glm.checkpoint_restore.Metadata, parts: []const f.Part) f.Name {
    var h = f.Sha.init(.{});
    h.update("GLM local page geometry 0960 v2");
    inline for (.{ "model", "drafter", "arithmetic", "layout" }) |field| h.update(&@field(a.binding, field));
    h.update(&f.wire.hash(metadata.shape));
    h.update(&f.wire.hash([_]u32{ a.world, rank }));
    for (parts) |p| h.update(&f.wire.hash([_]u64{ p.span, p.block }));
    return h.finalResult();
}
