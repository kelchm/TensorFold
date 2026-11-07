//! GLM's template may reorder tool outputs: request-owned occurrence labels travel with their media parts.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Tagged = struct {
    messages: Value,
    labels: []const []const u8,
    media: []const Value,
};

fn contains(value: Value, needle: []const u8) bool {
    return switch (value) {
        .string => |s| std.mem.indexOf(u8, s, needle) != null,
        .array => |items| for (items) |v| {
            if (contains(v, needle)) break true;
        } else false,
        .object => |o| blk: {
            var it = o.iterator();
            while (it.next()) |e| if (std.mem.indexOf(u8, e.key_ptr.*, needle) != null or contains(e.value_ptr.*, needle)) break :blk true;
            break :blk false;
        },
        else => false,
    };
}

/// Copy normalized content, keeping a distinct label next to each occurrence, including repeated URLs.
pub fn tag(cx: *Cx, messages: Value, image_type: []const u8) errors.Refused!Tagged {
    if (messages != .array) return .{ .messages = messages, .labels = &.{}, .media = &.{} };
    var serial: usize = 0;
    const prefix = while (true) : (serial += 1) {
        const candidate = try std.fmt.allocPrint(cx.a, "\x00tf-media-{d}:", .{serial});
        if (!contains(messages, candidate)) break candidate;
    };
    const list = try cx.a.dupe(Value, messages.array);
    var labels: std.ArrayList([]const u8) = .empty;
    var media: std.ArrayList(Value) = .empty;
    for (list) |*m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        var parts: std.ArrayList(Value) = .empty;
        for (content.array) |p| {
            const kind = p.strField("type") orelse "";
            if (std.mem.eql(u8, kind, "image_url") or std.mem.eql(u8, kind, image_type) or std.mem.eql(u8, kind, "video_url")) {
                const label = try std.fmt.allocPrint(cx.a, "{s}{d}\x00", .{ prefix, media.items.len });
                const text = try json.newObject(cx.a);
                try text.put(cx.a, "type", .{ .string = "text" });
                try text.put(cx.a, "text", .{ .string = label });
                try parts.append(cx.a, .{ .object = text });
                try labels.append(cx.a, label);
                try media.append(cx.a, p);
            }
            try parts.append(cx.a, p);
        }
        const copy = try json.copyObject(cx.a, m.object);
        try copy.put(cx.a, "content", .{ .array = parts.items });
        m.* = .{ .object = copy };
    }
    return .{ .messages = .{ .array = list }, .labels = labels.items, .media = media.items };
}

pub const Resolved = struct { rendered: model_text.Rendered, messages: Value };

/// Remove only the injected request ranges, retain text provenance, and extract actual rendered occurrence order.
pub fn resolve(cx: *Cx, tagged: Tagged, rendered: model_text.Rendered) errors.Refused!Resolved {
    const Hit = struct { at: usize, index: usize };
    const hits = try cx.a.alloc(Hit, tagged.labels.len);
    for (tagged.labels, 0..) |label, i| {
        const at = std.mem.indexOf(u8, rendered.text, label) orelse return cx.refuse("the chat template dropped a media occurrence");
        if (std.mem.indexOfPos(u8, rendered.text, at + label.len, label) != null) return cx.refuse("the chat template duplicated a media occurrence");
        const owned = for (rendered.spans) |span| {
            if (span[0] <= at and span[1] >= at + label.len) break true;
        } else false;
        if (!owned) return cx.refuse("the chat template lost media occurrence provenance");
        hits[i] = .{ .at = at, .index = i };
    }
    std.mem.sort(Hit, hits, {}, struct {
        fn less(_: void, a: Hit, b: Hit) bool {
            return a.at < b.at;
        }
    }.less);
    var text: std.ArrayList(u8) = .empty;
    var spans: std.ArrayList(model_text.Span) = .empty;
    const media = try cx.a.alloc(Value, hits.len);
    var done: usize = 0;
    for (hits, 0..) |hit, i| {
        try append(cx, rendered, done, hit.at, &text, &spans);
        done = hit.at + tagged.labels[hit.index].len;
        media[i] = tagged.media[hit.index];
    }
    try append(cx, rendered, done, rendered.text.len, &text, &spans);
    const message = try json.newObject(cx.a);
    try message.put(cx.a, "role", .{ .string = "user" });
    try message.put(cx.a, "content", .{ .array = media });
    const list = try cx.a.alloc(Value, 1);
    list[0] = .{ .object = message };
    return .{ .rendered = .{ .text = text.items, .spans = spans.items }, .messages = .{ .array = list } };
}

fn append(cx: *Cx, rendered: model_text.Rendered, lo: usize, hi: usize, text: *std.ArrayList(u8), spans: *std.ArrayList(model_text.Span)) !void {
    const offset = text.items.len;
    try text.appendSlice(cx.a, rendered.text[lo..hi]);
    for (rendered.spans) |span| {
        if (span[0] >= hi) break;
        if (span[1] > lo) try spans.append(cx.a, .{ offset + @max(lo, span[0]) - lo, offset + @min(hi, span[1]) - lo });
    }
}
