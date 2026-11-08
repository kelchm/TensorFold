//! Developer-session GPU identity, queried from the actual context device outside replay timing.
const std = @import("std");
const cuda = @import("cuda");
const Uuid = extern struct { bytes: [16]u8 };
const Query = *const fn (*Uuid, cuda.abi.Device) callconv(.c) cuda.abi.Result;

pub fn uuid(ctx: *const cuda.Context) ![40]u8 {
    var lib = ctx.d.lib;
    return read(ctx.d, ctx.device, try resolve(&lib));
}

fn resolve(lib: anytype) !Query {
    return lib.lookup(Query, "cuDeviceGetUuid_v2") orelse lib.lookup(Query, "cuDeviceGetUuid") orelse error.MissingSymbol;
}

fn read(d: anytype, device: cuda.abi.Device, query: Query) ![40]u8 {
    var value: Uuid = undefined;
    try d.check(query(&value, device), "cuDeviceGetUuid");
    const hex = std.fmt.bytesToHex(value.bytes, .lower);
    var text: [40]u8 = undefined;
    _ = try std.fmt.bufPrint(&text, "GPU-{s}-{s}-{s}-{s}-{s}", .{ hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..32] });
    return text;
}

test "replay device UUID prefers v2 falls back explicitly and checks actual device calls without a global API change" {
    const F = struct {
        v2: bool = true,
        v1: bool = true,
        var device: cuda.abi.Device = -1;
        fn good(value: *Uuid, ordinal: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            device = ordinal;
            for (&value.bytes, 0..) |*byte, i| byte.* = @intCast(i);
            return 0;
        }
        fn bad(_: *Uuid, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            return 1;
        }
        fn lookup(f: *@This(), comptime T: type, name: [:0]const u8) ?T {
            if (std.mem.eql(u8, name, "cuDeviceGetUuid_v2")) return if (f.v2) good else null;
            return if (f.v1) bad else null;
        }
        fn check(_: @This(), code: cuda.abi.Result, _: []const u8) !void {
            if (code != 0) return error.PlantedUuidFailure;
        }
    };
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Uuid));
    var lib: F = .{};
    const text = try read(lib, 7, try resolve(&lib));
    try std.testing.expectEqualStrings("GPU-00010203-0405-0607-0809-0a0b0c0d0e0f", &text);
    try std.testing.expectEqual(@as(cuda.abi.Device, 7), F.device);
    lib.v2 = false;
    try std.testing.expectError(error.PlantedUuidFailure, read(lib, 8, try resolve(&lib)));
    lib.v1 = false;
    try std.testing.expectError(error.MissingSymbol, resolve(&lib));
}
