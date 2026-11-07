//! Caller-owned transfer credits. DMA completion and file-write completion are independent lifetime gates.
const std = @import("std");
const capture = @import("checkpoint_capture.zig");

pub const Key = struct { slot: usize, generation: u64 };
const DMA = enum { pending, good, failed };
const IO = enum { waiting, pending, done, skipped };
pub const Slot = struct {
    generation: u64 = 0,
    chunk: ?capture.Chunk = null,
    dma: DMA = .pending,
    io: IO = .waiting,
};
pub const Ticket = struct { key: Key, chunk: capture.Chunk, target: []u8 };

/// Single-owner state machine; workers return tagged completions to that owner, including after cancellation.
pub const Ring = struct {
    source: capture.Source,
    storage: []u8,
    slots: []Slot,
    slot_bytes: usize,
    emitted: u64 = 0,
    written: u64 = 0,
    active: usize = 0,
    stopped: bool = false,
    closed: bool = false,

    /// Storage must be pinned for an asynchronous CUDA driver; its allocator outlives close().
    pub fn init(source: capture.Source, storage: []u8, slots: []Slot, slot_bytes: usize) !Ring {
        if (slots.len == 0 or slot_bytes == 0 or storage.len / slot_bytes < slots.len) return error.InvalidRing;
        try source.owner.borrow();
        @memset(slots, .{});
        return .{ .source = source, .storage = storage, .slots = slots, .slot_bytes = slot_bytes };
    }

    /// Success hands the driver one credit; even a failed submission must report when its work is drained.
    pub fn acquire(r: *Ring) !Ticket {
        if (r.closed) return error.Closed;
        if (r.stopped) return error.CheckpointStopped;
        if (r.emitted == r.source.len) return error.EndOfCheckpoint;
        const i = for (r.slots, 0..) |s, i| {
            if (s.chunk == null and s.generation < std.math.maxInt(u64)) break i;
        } else return error.CheckpointBackpressure;
        const chunk = (r.source.next(r.source.ctx, r.slot_bytes) catch |err| {
            r.cancel();
            return err;
        }) orelse {
            r.cancel();
            return error.ShortSource;
        };
        if (chunk.len == 0 or chunk.len > r.slot_bytes or chunk.offset != r.emitted or
            chunk.len > r.source.len - r.emitted)
        {
            r.cancel();
            return error.InvalidChunk;
        }
        const valid = switch (chunk.location) {
            .host => |bytes| bytes.len == chunk.len,
            .device => |address| address != 0 and address <= std.math.maxInt(u64) - chunk.len,
        };
        if (!valid) {
            r.cancel();
            return error.InvalidChunk;
        }
        const s = &r.slots[i];
        s.* = .{ .generation = s.generation + 1, .chunk = chunk };
        r.active += 1;
        r.emitted += chunk.len;
        return .{ .key = .{ .slot = i, .generation = s.generation }, .chunk = chunk, .target = r.storage[i * r.slot_bytes ..][0..chunk.len] };
    }

    fn slot(r: *Ring, key: Key) !*Slot {
        if (r.closed) return error.Closed;
        if (key.slot >= r.slots.len) return error.StaleCompletion;
        const s = &r.slots[key.slot];
        if (s.chunk == null or s.generation != key.generation) return error.StaleCompletion;
        return s;
    }

    pub fn inspect(r: *Ring, key: Key) !capture.Chunk {
        return (try r.slot(key)).chunk.?;
    }

    /// Called only after the event completes or the failing stream has drained; submission is not completion.
    pub fn dmaDone(r: *Ring, key: Key, success: bool) !void {
        const s = try r.slot(key);
        if (s.dma != .pending) return error.InvalidTransition;
        s.dma = if (success) .good else .failed;
        if (!success) r.cancel();
    }

    /// A sink can borrow the buffer only after DMA succeeded; its tagged completion ends that borrow.
    pub fn beginWrite(r: *Ring, key: Key) ![]const u8 {
        const s = try r.slot(key);
        if (r.stopped) return error.CheckpointStopped;
        if (s.dma != .good or s.io != .waiting) return error.InvalidTransition;
        s.io = .pending;
        return r.storage[key.slot * r.slot_bytes ..][0..s.chunk.?.len];
    }

    pub fn ioDone(r: *Ring, key: Key, success: bool) !void {
        const s = try r.slot(key);
        if (s.io != .pending) return error.InvalidTransition;
        s.io = .done;
        if (success) r.written += s.chunk.?.len else r.cancel();
    }

    /// Credits return only when both consumers are done. Cancellation never substitutes for either completion.
    pub fn retire(r: *Ring, key: Key) !void {
        const s = try r.slot(key);
        if (s.dma == .pending or (s.io != .done and s.io != .skipped)) return error.CheckpointBusy;
        s.chunk = null;
        r.active -= 1;
    }

    pub fn cancel(r: *Ring) void {
        if (r.closed) return;
        r.stopped = true;
        for (r.slots) |*s| if (s.chunk != null and s.io == .waiting) {
            s.io = .skipped;
        };
    }

    /// Ready for the rank writer's sealing step, not a claim of filesystem or all-rank durability.
    pub fn readyToSeal(r: *const Ring) bool {
        return !r.closed and !r.stopped and r.active == 0 and r.written == r.source.len;
    }

    pub fn close(r: *Ring) !void {
        if (r.closed) return error.Closed;
        if (r.active != 0) return error.CheckpointBusy;
        if (!r.stopped and !r.readyToSeal()) return error.CheckpointBusy;
        r.source.owner.unborrow();
        r.closed = true;
    }
};

test {
    _ = @import("checkpoint_stream_test.zig");
}
