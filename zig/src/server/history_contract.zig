//! Preserve caller reasoning and refuse tool-result identities a sorting template would silently omit.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const strip = @import("reply_text.zig").pyStrip;
const Value = json.Value;

pub fn reasoningAlias(cx: *errors.Cx, item: *json.Object) std.mem.Allocator.Error!*json.Object {
    const value: Value = .{ .object = item };
    if (!std.mem.eql(u8, value.strField("role") orelse "", "assistant")) return item;
    if (strip(value.strField("reasoning_content") orelse "").len > 0) return item;
    const alias = value.strField("reasoning") orelse return item;
    if (strip(alias).len == 0) return item;
    const copy = try json.copyObject(cx.a, item);
    try copy.put(cx.a, "reasoning_content", .{ .string = alias });
    return copy;
}

fn idOf(value: Value) ?Value {
    const tool_id = value.get("tool_call_id");
    if (tool_id != null and tool_id.?.truthy()) return tool_id;
    return value.get("id");
}

pub fn toolResults(cx: *errors.Cx, messages: []const Value) errors.Refused!void {
    var preceding: ?[]const Value = null;
    for (messages, 0..) |message, index| {
        const role = message.strField("role") orelse "";
        if (!std.mem.eql(u8, role, "tool")) {
            preceding = null;
            if (std.mem.eql(u8, role, "assistant")) if (message.get("tool_calls")) |calls| {
                if (calls == .array and calls.array.len > 0) preceding = calls.array;
            };
            continue;
        }
        const value = idOf(message) orelse continue;
        if (value == .null or (value == .string and value.string.len == 0)) continue;
        if (value != .string) return cx.fail(.request, "messages[{d}].tool_call_id must be a string", .{index});
        // With no preceding call list, the template renders this block in its original order.
        const calls = preceding orelse continue;
        // A call without an ID disables sorting; leave that lossless fallback intact.
        const identified = for (calls) |call| {
            const id = idOf(call) orelse break false;
            if (!id.truthy()) break false;
        } else true;
        if (!identified) continue;
        for (calls) |call| {
            const id = idOf(call) orelse continue;
            if (id == .string and std.mem.eql(u8, id.string, value.string)) break;
        } else return cx.fail(.request, "messages[{d}].tool_call_id does not match the preceding assistant tool_calls", .{index});
    }
}
