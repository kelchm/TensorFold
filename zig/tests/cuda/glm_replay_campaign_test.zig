//! Host negative controls exercise the same manifest, identity and output checks used before CUDA.
const std = @import("std");
const campaign = @import("glm_replay_campaign.zig");
const meta = @import("glm_replay_metadata.zig");

fn control(cell: []const u8, block: []const u8, output: []const u8) campaign.Case {
    var c = std.mem.zeroes(campaign.Case);
    c.output = output;
    c.f.fixture_sha256 = cell;
    c.f.config_sha256 = "config";
    c.f.tokens_sha256 = "tokens";
    c.f.schedule_sha256 = "schedule";
    c.f.config.pool = .{ .h = 256, .page_tokens = 512, .configured_capacity = 1024, .prompt_twins = 1 };
    c.m = std.mem.zeroes(meta.Metadata);
    c.m.run_id = "run";
    c.m.pair_id = "pair";
    c.m.start_id = "start";
    c.m.cell_id = cell;
    c.m.block_id = block;
    c.m.fixture_sha256 = c.f.fixture_sha256;
    c.m.config_sha256 = c.f.config_sha256;
    inline for (@typeInfo(meta.Source).@"struct".field_names) |field| @field(c.m.source, field) = "same-source";
    return c;
}

test "campaign refuses implicit FP8 page mode before opening the driver" {
    try campaign.checkPages("on");
    for ([_]?[]const u8{ null, "auto", "off", "", "1" }) |value|
        try std.testing.expectError(error.ReplayNeedsPages, campaign.checkPages(value));
}

test "campaign descriptor parser refuses ambiguous schemas before any files or driver" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const good =
        \\{"schema":"glm-replay-native-campaign-v1","cases":[{"fixture":"f","metadata":"m","output":"o"}]}
    ;
    const m = try campaign.parse(a, good);
    try std.testing.expectEqualStrings("f", m.cases[0].fixture);
    for ([_][]const u8{
        "{}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[]}",
        "{\"schema\":\"wrong\",\"cases\":[{\"fixture\":\"f\",\"metadata\":\"m\",\"output\":\"o\"}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[{\"fixture\":\"f\",\"metadata\":\"m\"}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[{\"fixture\":\"f\",\"metadata\":\"m\",\"output\":0}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[{\"fixture\":\"f\",\"metadata\":\"m\",\"output\":\"o\",\"repeat\":5}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[{\"fixture\":\"f\",\"metadata\":\"m\",\"output\":\"o\",\"output\":\"o2\"}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[{\"fixture\":\"\",\"metadata\":\"m\",\"output\":\"o\"}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[{\"fixture\":\"f\",\"metadata\":\"m\",\"output\":\"o\\u0000x\"}]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"schema\":\"other\",\"cases\":[]}",
        "{\"schema\":\"glm-replay-native-campaign-v1\",\"cases\":[],\"repeat\":5}",
    }) |raw| {
        if (campaign.parse(a, raw)) |_| return error.NegativeControlPassed else |_| {}
    }
    const many = try a.alloc(campaign.Descriptor, campaign.max_cases + 1);
    @memset(many, m.cases[0]);
    const too_many = try std.json.Stringify.valueAlloc(a, campaign.Manifest{ .schema = m.schema, .cases = many }, .{});
    try std.testing.expectError(error.CampaignSchema, campaign.parse(a, too_many));
}

test "campaign checker accepts rotated repeated cells and binds exact order and identities" {
    const a = std.testing.allocator;
    var cases = [_]campaign.Case{
        control("anchor", "0", "anchor-0.jsonl"),
        control("wide", "0", "wide-0.jsonl"),
        control("anchor", "1", "anchor-1.jsonl"),
    };
    const ordered = try campaign.check(a, &cases);
    const original = cases;
    const reversed = [_]campaign.Case{ cases[2], cases[1], cases[0] };
    try std.testing.expect(!std.mem.eql(u8, &ordered, &try campaign.check(a, &reversed)));
    cases[0].output = "rank-local-other-path";
    try std.testing.expectEqual(ordered, try campaign.check(a, &cases));
    cases[2].m.block_id = "2";
    try std.testing.expect(!std.mem.eql(u8, &ordered, &try campaign.check(a, &cases)));
    cases[2].m.block_id = "1";
    cases[2].m.correctness = .{ .path = "gate.json", .sha256 = "changed-gate-hash" };
    try std.testing.expect(!std.mem.eql(u8, &ordered, &try campaign.check(a, &cases)));
    cases = original;
    for (&cases) |*c| c.m.start_id = "new-process-start";
    try std.testing.expect(!std.mem.eql(u8, &ordered, &try campaign.check(a, &cases)));
    cases = original;
    for (&cases) |*c| c.m.source.binary_sha256 = "new-build";
    try std.testing.expect(!std.mem.eql(u8, &ordered, &try campaign.check(a, &cases)));
    cases = original;
    cases[1].f.config_sha256 = "new-config";
    cases[1].m.config_sha256 = "new-config";
    try std.testing.expect(!std.mem.eql(u8, &ordered, &try campaign.check(a, &cases)));
}

test "campaign checker rejects mixed starts builds pools and duplicate block cell output identities" {
    const a = std.testing.allocator;
    const original = [_]campaign.Case{ control("solo", "0", "solo-0"), control("solo", "1", "solo-1") };
    inline for (.{ "run_id", "pair_id", "start_id" }) |field| {
        var cases = original;
        @field(cases[1].m, field) = "different";
        try std.testing.expectError(error.CampaignMixedStart, campaign.check(a, &cases));
    }
    var cases = original;
    cases[1].m.start_order = 1;
    try std.testing.expectError(error.CampaignMixedStart, campaign.check(a, &cases));
    inline for (@typeInfo(meta.Source).@"struct".field_names) |field| {
        cases = original;
        @field(cases[1].m.source, field) = "different";
        try std.testing.expectError(error.CampaignMixedSource, campaign.check(a, &cases));
    }
    inline for (.{ "h", "page_tokens", "configured_capacity", "prompt_twins" }) |field| {
        cases = original;
        @field(cases[1].f.config.pool, field) += 1;
        try std.testing.expectError(error.CampaignMixedPool, campaign.check(a, &cases));
    }
    cases = original;
    cases[1].m.block_id = "0";
    try std.testing.expectError(error.CampaignDuplicateBlock, campaign.check(a, &cases));
    cases = original;
    cases[1].output = "solo-0";
    try std.testing.expectError(error.CampaignDuplicateOutput, campaign.check(a, &cases));
    cases = original;
    cases[1].f.fixture_sha256 = "changed-fixture";
    cases[1].m.fixture_sha256 = "changed-fixture";
    try std.testing.expectError(error.CampaignChangedCell, campaign.check(a, &cases));
    cases = original;
    cases[1].m.fixture_sha256 = "wrong";
    try std.testing.expectError(error.MetadataFixture, campaign.check(a, &cases));
    cases = original;
    cases[1].m.block_id = "";
    try std.testing.expectError(error.CampaignIdentity, campaign.check(a, &cases));
    try std.testing.expectError(error.CampaignSchema, campaign.check(a, &.{}));
}

test "campaign exclusive output preflight resolves parent aliases and preserves existing and dangling files" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "result" });
    defer a.free(path);
    const canonical = try campaign.outputPath(a, io, path);
    defer a.free(canonical);
    const alias = try std.fs.path.join(a, &.{ dir, ".", "result" });
    defer a.free(alias);
    const canonical_alias = try campaign.outputPath(a, io, alias);
    defer a.free(canonical_alias);
    try std.testing.expectEqualStrings(canonical, canonical_alias);
    try tmp.dir.symLink(io, ".", "alias", .{ .is_directory = true });
    const linked = try std.fs.path.join(a, &.{ dir, "alias", "result" });
    defer a.free(linked);
    const canonical_linked = try campaign.outputPath(a, io, linked);
    defer a.free(canonical_linked);
    try std.testing.expectEqualStrings(canonical, canonical_linked);
    var cases = [_]campaign.Case{ control("solo", "0", canonical), control("solo", "1", canonical_linked) };
    try std.testing.expectError(error.CampaignDuplicateOutput, campaign.check(a, &cases));
    try tmp.dir.writeFile(io, .{ .sub_path = "result", .data = "preserved" });
    try std.testing.expectError(error.CampaignOutputExists, campaign.outputPath(a, io, path));
    const contents = try tmp.dir.readFileAlloc(io, "result", a, .limited(100));
    defer a.free(contents);
    try std.testing.expectEqualStrings("preserved", contents);
    try tmp.dir.deleteFile(io, "result");
    try tmp.dir.symLink(io, "missing-target", "result", .{});
    try std.testing.expectError(error.CampaignOutputExists, campaign.outputPath(a, io, path));
}

test "campaign metadata reader rejects numeric block identities before checking any artifacts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = control("solo", "block-7", "output");
    const raw = try std.json.Stringify.valueAlloc(a, c.m, .{});
    var value = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
    try value.object.put(a, "block_id", .{ .integer = 7 });
    try tmp.dir.writeFile(io, .{ .sub_path = "metadata.json", .data = try std.json.Stringify.valueAlloc(a, value, .{}) });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/metadata.json", .{tmp.sub_path});
    try std.testing.expectError(error.UnexpectedToken, meta.read(a, io, path, c.f, true, false));
}
