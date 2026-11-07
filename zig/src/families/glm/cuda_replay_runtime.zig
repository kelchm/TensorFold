//! Bind effective runtime and transport without exposing environment values or the sharding treatment.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Sha = std.crypto.hash.sha2.Sha256;

fn relevant(name: []const u8) bool {
    for ([_][]const u8{ "TF_GLM_RANK", "TF_GLM_KV_SHARD", "TF_GLM_KV_SHARD_CARRY", "TF_GLM_KV_SHARD_TWINS", "TF_GLM_KV_SHARD_CHECK", "NCCL_COMM_ID" }) |skip|
        if (std.mem.eql(u8, name, skip)) return false;
    for ([_][]const u8{ "TF_", "NCCL_", "CUDA_", "CUBLAS_", "NVIDIA_", "LD_" }) |prefix|
        if (std.mem.startsWith(u8, name, prefix)) return true;
    return false;
}

fn less(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

pub fn settingsHash(gpa: std.mem.Allocator, settings: ranks.Settings, entries: []const []const u8) ![32]u8 {
    var normalized = settings;
    normalized.kv_shard = .{};
    const json = try std.json.Stringify.valueAlloc(gpa, normalized, .{});
    defer gpa.free(json);
    var selected: std.ArrayList([]const u8) = .empty;
    defer selected.deinit(gpa);
    for (entries) |entry| {
        const eq = std.mem.indexOfScalar(u8, entry, '=') orelse return error.ReplayEnvironment;
        if (relevant(entry[0..eq])) try selected.append(gpa, entry);
    }
    std.mem.sort([]const u8, selected.items, {}, less);
    var h = Sha.init(.{});
    h.update("glm-replay-runtime-v1\x00");
    h.update(json);
    h.update("\x00");
    for (selected.items) |entry| {
        h.update(entry);
        h.update("\x00");
    }
    return h.finalResult();
}

pub fn loaded(e: anytype) ![32]u8 {
    var entries: std.ArrayList([]const u8) = .empty;
    defer entries.deinit(e.gpa);
    var i: usize = 0;
    while (std.c.environ[i]) |entry| : (i += 1) try entries.append(e.gpa, std.mem.span(entry));
    const base = try settingsHash(e.gpa, e.settings, entries.items);
    const group = e.group.?;
    const actual = try std.json.Stringify.valueAlloc(e.gpa, .{
        .capacity = e.capacity,
        .world = e.share.world,
        .roce = group.fast != null,
        .p2p = group.p2p != null,
        .ce = e.ce != null,
        .side = e.side != null,
        .pair = e.pair != null,
        .moe_fuse = e.moe_fuse,
        .topk_sum = e.m.topk_sum,
        .shared_segs = e.m.shared_segs,
        .hc_split = e.m.hc_split,
        .shared_writes = e.m.shared_writes,
    }, .{});
    defer e.gpa.free(actual);
    var h = Sha.init(.{});
    h.update(&base);
    h.update(actual);
    return h.finalResult();
}

pub fn combine(all: [2][32]u8) [32]u8 {
    var h = Sha.init(.{});
    h.update("glm-replay-runtime-ranks-v1\x00");
    h.update(&all[0]);
    h.update(&all[1]);
    return h.finalResult();
}

test "replay runtime binding ignores treatment and env order but detects transport and execution changes" {
    const a = std.testing.allocator;
    const plain: ranks.Settings = .{};
    var sharded = plain;
    sharded.kv_shard.on = true;
    const first = try settingsHash(a, plain, &.{ "TF_GLM_RANK=0", "TF_GLM_KV_SHARD=0", "TF_ROCE=1", "TF_GLM_SHARED_SIDE=1" });
    const second = try settingsHash(a, sharded, &.{ "TF_GLM_SHARED_SIDE=1", "TF_ROCE=1", "TF_GLM_KV_SHARD=1", "TF_GLM_RANK=1" });
    try std.testing.expectEqual(first, second);
    const changed = try settingsHash(a, plain, &.{ "TF_ROCE=0", "TF_GLM_SHARED_SIDE=1" });
    try std.testing.expect(!std.mem.eql(u8, &first, &changed));
}
