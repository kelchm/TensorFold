//! TF_GLM_REQUEST_STARTS: the explicit message-cut opt-in, refused with the shared KV store at startup.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const glm_env = @import("cuda_env.zig");
const env = @import("sys").env;
const t = std.testing;

const Saved = struct {
    const names = [_][:0]const u8{ "TF_GLM_REQUEST_STARTS", "TF_KV_SHARE" };
    values: [names.len]?[:0]u8 = @splat(null),
    fn take() !Saved {
        var s: Saved = .{};
        errdefer for (s.values) |v| if (v) |old| t.allocator.free(old);
        for (names, 0..) |n, i| if (@import("sys").getenv(n)) |v| {
            s.values[i] = try t.allocator.dupeSentinel(u8, v, 0);
        };
        return s;
    }
    fn restore(s: *Saved) void {
        for (names, s.values) |n, v| if (v) |old| {
            _ = env.setenv(n, old, 1);
            t.allocator.free(old);
        } else {
            _ = env.unsetenv(n);
        };
    }
};

fn put(name: [:0]const u8, value: ?[:0]const u8) !void {
    try t.expectEqual(@as(c_int, 0), if (value) |v| env.setenv(name, v, 1) else env.unsetenv(name));
}

test "glm request starts: shared KV refuses at startup before any rank opens, by name" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_GLM_REQUEST_STARTS", "1");
    try put("TF_KV_SHARE", "32");
    var ctx: cuda.Context = undefined;
    try t.expectError(error.GlmRequestStartsWithSharedKv, native.open(t.allocator, t.io, &ctx, "missing-request-starts-checkpoint", null, .{ .context = 8192, .drafts = false }));
    const reason = native.explain(null, error.GlmRequestStartsWithSharedKv) orelse return error.MissingNamedRefusal;
    try t.expect(std.mem.indexOf(u8, reason, "TF_GLM_REQUEST_STARTS=1") != null);
    try t.expect(std.mem.indexOf(u8, reason, "TF_KV_SHARE") != null);
}

test "glm request starts: --learn-conversations is allowed at startup alongside TF_GLM_REQUEST_STARTS=1" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_GLM_REQUEST_STARTS", "1");
    try put("TF_KV_SHARE", null);
    try glm_env.requestStartsAlone();
    var ctx: cuda.Context = undefined;
    const o: native.Options = .{ .context = 8192, .drafts = false, .learn_conversations = true };
    const opened = native.open(t.allocator, t.io, &ctx, "missing-request-starts-checkpoint", null, o);
    if (opened) |_| return error.ExpectedMissingCheckpoint else |err| try t.expect(err != error.GlmRequestStartsWithSharedKv);
    try put("TF_GLM_REQUEST_STARTS", null);
    try glm_env.requestStartsAlone();
}

test "glm request starts: valid values pass startup; invalid values fail startup by name; 1 and marks refuse shared KV" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_KV_SHARE", "32");
    for ([_]?[:0]const u8{ null, "", " ", "0" }) |v| {
        try put("TF_GLM_REQUEST_STARTS", v);
        try t.expectEqual(glm_env.RequestStarts.off, glm_env.requestStarts());
        try glm_env.startup();
        try glm_env.requestStartsAlone();
    }
    const sys_env = @import("sys").environment;
    for ([_][:0]const u8{ "MARKS", "on", "off", "true", "2", " marks", "marks " }) |v| {
        try put("TF_GLM_REQUEST_STARTS", v);
        try t.expectError(error.InvalidEnvironment, glm_env.startup());
        try t.expectError(error.InvalidEnvironment, glm_env.requestStartsAlone());
        var ctx: cuda.Context = undefined;
        try t.expectError(error.InvalidEnvironment, native.open(t.allocator, t.io, &ctx, "missing-request-starts-checkpoint", null, .{ .context = 8192, .drafts = false }));
        const why = (try sys_env.refusal(t.allocator, sys_env.Process{}, .{ .glm = true })) orelse return error.MissingEnvDiagnostic;
        defer t.allocator.free(why);
        try t.expect(std.mem.startsWith(u8, why, "TF_GLM_REQUEST_STARTS"));
    }
    for ([_][:0]const u8{ "1", "marks" }) |mode| {
        try put("TF_GLM_REQUEST_STARTS", mode);
        try t.expectEqual(if (std.mem.eql(u8, mode, "1")) glm_env.RequestStarts.all else glm_env.RequestStarts.marks, glm_env.requestStarts());
        try glm_env.startup();
        for ([_]?[:0]const u8{ null, "0", "00" }) |v| {
            try put("TF_KV_SHARE", v);
            try glm_env.requestStartsAlone();
        }
        for ([_][:0]const u8{ "1", "32", "00032" }) |v| {
            try put("TF_KV_SHARE", v);
            try t.expectError(error.GlmRequestStartsWithSharedKv, glm_env.requestStartsAlone());
        }
    }
    try put("TF_KV_SHARE", "bad");
    try t.expectError(error.BadKvShare, glm_env.requestStartsAlone());
}
