//! Fixed-size, little-endian checkpoint records. Never persist ABI padding, pointers or enum ordinals.
const std = @import("std");

pub fn size(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .int => @sizeOf(T),
        .array => |a| a.len * size(a.child),
        .@"struct" => |s| blk: {
            var n: usize = 0;
            inline for (s.field_names) |name| n += size(@FieldType(T, name));
            break :blk n;
        },
        else => @compileError("checkpoint wire type must contain only integers, arrays and records"),
    };
}

pub fn encode(value: anytype, out: []u8) ![]const u8 {
    const n = comptime size(@TypeOf(value));
    if (out.len < n) return error.CheckpointMetadataLimit;
    put(value, out[0..n]);
    return out[0..n];
}

fn put(value: anytype, out: []u8) void {
    const T = @TypeOf(value);
    switch (@typeInfo(T)) {
        .int => std.mem.writeInt(T, out[0..@sizeOf(T)], value, .little),
        .array => |a| for (value, 0..) |v, i| put(v, out[i * size(a.child) ..][0..size(a.child)]),
        .@"struct" => |s| {
            var offset: usize = 0;
            inline for (s.field_names) |name| {
                const n = comptime size(@FieldType(T, name));
                put(@field(value, name), out[offset..][0..n]);
                offset += n;
            }
        },
        else => unreachable,
    }
}

pub fn decode(comptime T: type, bytes: []const u8) !T {
    if (bytes.len != size(T)) return error.CheckpointMetadataSize;
    return get(T, bytes);
}

fn get(comptime T: type, bytes: []const u8) T {
    switch (@typeInfo(T)) {
        .int => return std.mem.readInt(T, bytes[0..@sizeOf(T)], .little),
        .array => |a| {
            var result: T = undefined;
            for (&result, 0..) |*v, i| v.* = get(a.child, bytes[i * size(a.child) ..][0..size(a.child)]);
            return result;
        },
        .@"struct" => |s| {
            var result: T = undefined;
            var offset: usize = 0;
            inline for (s.field_names) |name| {
                const F = @FieldType(T, name);
                const n = comptime size(F);
                @field(result, name) = get(F, bytes[offset..][0..n]);
                offset += n;
            }
            return result;
        },
        else => unreachable,
    }
}

pub fn hash(value: anytype) [32]u8 {
    var bytes: [size(@TypeOf(value))]u8 = undefined;
    const data = encode(value, &bytes) catch unreachable;
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &result, .{});
    return result;
}
