//! Optional NVML clocks matched by CUDA's UUID, including when CUDA_VISIBLE_DEVICES remaps device 0.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn print(gpu: check.Gpu) !void {
    var name: [128]u8 = undefined;
    std.debug.print("ROOFLINE device 0 ordinal name={s} sm={d} integrated={d}\n", .{ try gpu.ctx.name(&name), try gpu.ctx.capability(), try gpu.ctx.attribute(.integrated) });
    std.debug.print("ROOFLINE sm-count {d} SMs source=driver\n", .{try gpu.ctx.attribute(.multiprocessor_count)});
    std.debug.print("ROOFLINE l2-size {d} bytes source=driver\n", .{try gpu.ctx.attribute(.l2_cache_size)});
    std.debug.print("ROOFLINE driver-clock {d} kHz source=CU_DEVICE_ATTRIBUTE_CLOCK_RATE\n", .{try gpu.ctx.attribute(.clock_rate)});
    clocks(gpu);
}

fn clocks(gpu: check.Gpu) void {
    const Device = ?*opaque {};
    const Init = *const fn () callconv(.c) c_int;
    const Handle = *const fn ([*:0]const u8, *Device) callconv(.c) c_int;
    const Clock = *const fn (Device, c_int, *c_uint) callconv(.c) c_int;
    var lib = std.DynLib.open("libnvidia-ml.so.1") catch return unavailable("NVML library unavailable");
    defer lib.close();
    const init = lib.lookup(Init, "nvmlInit_v2") orelse return unavailable("nvmlInit_v2 missing");
    const shutdown = lib.lookup(Init, "nvmlShutdown") orelse return unavailable("nvmlShutdown missing");
    const handle = lib.lookup(Handle, "nvmlDeviceGetHandleByUUID") orelse return unavailable("NVML UUID lookup missing");
    const current = lib.lookup(Clock, "nvmlDeviceGetClockInfo") orelse return unavailable("NVML current clocks missing");
    const max = lib.lookup(Clock, "nvmlDeviceGetMaxClockInfo") orelse return unavailable("NVML max clocks missing");
    if (init() != 0) return unavailable("NVML initialization failed");
    defer _ = shutdown();
    const uuid = gpu.ctx.uuid() catch return unavailable("CUDA UUID unavailable");
    const hex = std.fmt.bytesToHex(uuid, .lower);
    var buf: [80]u8 = undefined;
    const id = std.fmt.bufPrint(&buf, "GPU-{s}-{s}-{s}-{s}-{s}\x00", .{ hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..32] }) catch unreachable;
    var dev: Device = null;
    if (handle(@ptrCast(id.ptr), &dev) != 0) return unavailable("NVML did not find CUDA device 0");
    for ([_]c_int{ 1, 2 }, [_][]const u8{ "sm", "memory" }) |kind, label| {
        var mhz: c_uint = 0;
        if (current(dev, kind, &mhz) == 0)
            std.debug.print("ROOFLINE clock-{s}-current {d} MHz source=NVML\n", .{ label, mhz })
        else
            std.debug.print("ROOFLINE clock-{s}-current 0 SKIP NVML query unsupported\n", .{label});
        if (max(dev, kind, &mhz) == 0)
            std.debug.print("ROOFLINE clock-{s}-max {d} MHz source=NVML\n", .{ label, mhz })
        else
            std.debug.print("ROOFLINE clock-{s}-max 0 SKIP NVML query unsupported\n", .{label});
    }
}

fn unavailable(reason: []const u8) void {
    std.debug.print("ROOFLINE clocks 0 SKIP {s}; driver-clock is not a live clock sample\n", .{reason});
}
