//! A real device capsule freezer. The sequence stays paused through its completion fence.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("cuda_checkpoint.zig");

pub const Device = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    release_failure: ?*const fn (anyerror) noreturn = null,

    pub fn adapter(d: *Device) checkpoint.Freezer {
        return .{ .ctx = d, .freeze = freeze };
    }

    pub const descriptor_bytes = @sizeOf(Owned);

    const Owned = struct {
        gpa: std.mem.Allocator,
        buffer: cuda.DeviceBuffer,
        release_failure: ?*const fn (anyerror) noreturn,

        fn freeBuffer(o: *Owned) void {
            if (o.release_failure) |fail| o.buffer.freeChecked() catch |err| fail(err) else o.buffer.free();
        }

        fn release(ptr: *anyopaque) void {
            const o: *Owned = @ptrCast(@alignCast(ptr));
            o.freeBuffer();
            o.gpa.destroy(o);
        }
    };

    fn freeze(ptr: *anyopaque, mutable: *checkpoint.Mutable, bytes: usize) !checkpoint.Frozen {
        const d: *Device = @ptrCast(@alignCast(ptr));
        const o = try d.gpa.create(Owned);
        errdefer d.gpa.destroy(o);
        o.* = .{ .gpa = d.gpa, .buffer = try cuda.DeviceBuffer.alloc(d.d, bytes), .release_failure = d.release_failure };
        errdefer o.freeBuffer();
        // This defer runs before freeing the destination on any failed enqueue, event or stream operation.
        errdefer cuda.completion.drain(d.d, d.stream, "a failed checkpoint capsule freeze did not drain");
        while (try mutable.next(2 << 20)) |chunk| {
            try d.d.check(d.d.api.cuMemcpyDtoDAsync_v2(o.buffer.ptr + chunk.offset, chunk.location.device, chunk.len, d.stream.handle), "cuMemcpyDtoDAsync checkpoint capsule");
        }
        try d.stream.synchronize();
        return .{ .storage = .{ .device = o.buffer.ptr }, .len = bytes, .ctx = o, .release = Owned.release };
    }
};
