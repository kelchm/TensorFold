//! Completed replies' reasoning, addressed only by the exact call IDs delivered to their clients.
const std = @import("std");
const json = @import("json");
const strip = @import("reply_text.zig").pyStrip;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const max_calls = 128;

pub const Cache = struct {
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    by_id: std.StringHashMapUnmanaged(*Entry) = .empty,
    oldest: ?*Entry = null,
    newest: ?*Entry = null,
    count: usize = 0,
    bytes: usize = 0,
    limit: usize = 1024,
    // UTF-8 bytes, including IDs and entry storage: stricter than the Python cache's 32 Mi characters.
    max_bytes: usize = 32 << 20,

    const Entry = struct {
        previous: ?*Entry = null,
        next: ?*Entry = null,
        storage: []u8,
        ids: [][]const u8,
        reasoning: []const u8,
        size: usize,
    };

    /// Server shutdown joins its connections before destroying this cache.
    pub fn deinit(c: *Cache) void {
        while (c.oldest) |e| c.remove(e);
        c.by_id.deinit(c.gpa);
    }

    fn unlink(c: *Cache, e: *Entry) void {
        if (e.previous) |p| p.next = e.next else c.oldest = e.next;
        if (e.next) |n| n.previous = e.previous else c.newest = e.previous;
    }

    fn append(c: *Cache, e: *Entry) void {
        e.previous = c.newest;
        e.next = null;
        if (c.newest) |n| n.next = e else c.oldest = e;
        c.newest = e;
    }

    fn remove(c: *Cache, e: *Entry) void {
        c.unlink(e);
        for (e.ids) |id| _ = c.by_id.remove(id);
        c.count -= 1;
        c.bytes -= e.size;
        c.gpa.free(e.storage);
        c.gpa.free(e.ids);
        c.gpa.destroy(e);
    }

    /// Called only after successful delivery of a normally completed, parsed tool-calling reply.
    pub fn remember(c: *Cache, io: std.Io, call_ids: []const []const u8, reasoning: ?[]const u8) Allocator.Error!void {
        const text = reasoning orelse return;
        if (call_ids.len == 0 or call_ids.len > max_calls or strip(text).len == 0) return;
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        if (c.limit == 0 or text.len > c.max_bytes) return;
        var size = text.len;
        for (call_ids, 0..) |id, i| {
            if (id.len == 0 or id.len > 256 or id.len > c.max_bytes -| size) return;
            for (call_ids[0..i]) |earlier| if (std.mem.eql(u8, earlier, id)) return;
            size += id.len;
            // An ID must never select a different reply, even on an accidental collision.
            if (c.by_id.contains(id)) return;
        }
        const overhead = @sizeOf(Entry) + call_ids.len * @sizeOf([]const u8);
        if (overhead > c.max_bytes -| size) return;
        const e = try c.gpa.create(Entry);
        errdefer c.gpa.destroy(e);
        const storage = try c.gpa.alloc(u8, size);
        errdefer c.gpa.free(storage);
        const copied_ids = try c.gpa.alloc([]const u8, call_ids.len);
        errdefer c.gpa.free(copied_ids);
        // Bound index metadata as well: at most 1024 * 128 keys (capacity retained after eviction).
        const entry_limit: usize = @min(c.limit, 1024);
        const key_limit = entry_limit * @as(usize, max_calls);
        try c.by_id.ensureTotalCapacity(c.gpa, @intCast(@min(c.by_id.count() + call_ids.len, key_limit)));
        @memcpy(storage[0..text.len], text);
        var offset = text.len;
        for (call_ids, copied_ids) |id, *copy| {
            @memcpy(storage[offset..][0..id.len], id);
            copy.* = storage[offset..][0..id.len];
            offset += id.len;
        }
        e.* = .{ .storage = storage, .ids = copied_ids, .reasoning = storage[0..text.len], .size = size + overhead };
        while (c.count >= @min(c.limit, 1024) or c.bytes > c.max_bytes - e.size) c.remove(c.oldest.?);
        for (e.ids) |id| c.by_id.putAssumeCapacity(id, e);
        c.append(e);
        c.count += 1;
        c.bytes += e.size;
    }

    /// Copy under the lock: eviction cannot invalidate strings borrowed by an in-flight request.
    pub fn restore(c: *Cache, io: std.Io, a: Allocator, messages: Value) Allocator.Error!Value {
        if (messages != .array) return messages;
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        if (c.count == 0) return messages;
        var result: ?[]Value = null;
        for (messages.array, 0..) |m, i| {
            if (!std.mem.eql(u8, m.strField("role") orelse "", "assistant")) continue;
            if (strip(m.strField("reasoning_content") orelse "").len > 0 or
                strip(m.strField("reasoning") orelse "").len > 0 or
                std.mem.indexOf(u8, m.strField("content") orelse "", "</think>") != null) continue;
            const calls = m.get("tool_calls") orelse continue;
            if (calls != .array or calls.array.len == 0 or calls.array.len > max_calls) continue;
            var found: ?*Entry = null;
            const e = for (calls.array) |call| {
                const id = call.strField("id") orelse break null;
                const entry = c.by_id.get(id) orelse break null;
                if (found != null and found.? != entry) break null;
                found = entry;
            } else found;
            const entry = e orelse continue;
            const copy = try json.copyObject(a, m.object);
            try copy.put(a, "reasoning_content", .{ .string = try a.dupe(u8, entry.reasoning) });
            if (result == null) result = try a.dupe(Value, messages.array);
            result.?[i] = .{ .object = copy };
            c.unlink(entry);
            c.append(entry);
        }
        return if (result) |out| .{ .array = out } else messages;
    }
};
