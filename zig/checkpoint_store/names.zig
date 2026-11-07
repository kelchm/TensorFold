//! Reuses TensorFold rows_chain token/span/chunk naming; binds media content before each completed page.
const std = @import("std");
const glm = @import("glm");
const f = @import("format.zig");

pub const Input = struct {
    tokens: []const u32,
    spans: []const [2]u32 = &.{},
    starts: []const u32 = &.{},
    planned: bool = false,
};

/// Inputs are the paused sequence's canonical tokens and prepared media, never a caller-chosen pathname.
pub fn build(input: Input, metadata: *const glm.checkpoint_restore.Metadata, out: []f.Name) !void {
    try metadata.validate();
    if (out.len > f.max_pages or out.len * glm.state.page_rows > metadata.mark or input.tokens.len < metadata.mark) return error.BlockIdentity;
    var end: u32 = 0;
    for (input.spans) |span| {
        if (span[0] < end or span[1] <= span[0] or span[1] > metadata.mark) return error.BlockIdentity;
        end = span[1];
    }
    end = 0;
    for (input.starts, 0..) |start, i| {
        if ((i != 0 and start <= end) or start > metadata.mark) return error.BlockIdentity;
        end = start;
    }
    f.chain.blocks(.{ .identity = 0x303539304d4c47, .block = glm.state.page_rows, .planned = input.planned }, input.tokens, input.spans, input.starts, @intCast(metadata.mark), 0, @intCast(out.len), out);
    for (out, 0..) |*name, i| {
        var h = f.Sha.init(.{});
        h.update("GLM rows media prefix 0950 v1");
        h.update(name);
        const through = (i + 1) * glm.state.page_rows;
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
    h.update("GLM local page geometry 0950 v1");
    inline for (.{ "model", "drafter", "arithmetic", "layout" }) |field| h.update(&@field(a.binding, field));
    h.update(&f.wire.hash(metadata.shape));
    h.update(&f.wire.hash([_]u32{ a.world, rank }));
    for (parts) |p| h.update(&f.wire.hash([_]u64{ p.span, p.block }));
    return h.finalResult();
}
