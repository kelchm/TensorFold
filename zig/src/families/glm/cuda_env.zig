//! Validate GLM's TF_GLM_* settings on every rank before its CUDA open reads ranks, NCCL or weights.
const std = @import("std");
const environment = @import("sys").environment;
const kp = @import("core").kv_prompts;

/// Every `.glm` row of the shared table, parsed exactly as its later reader parses it.
pub fn startup() !void {
    try environment.checkScope(.glm, environment.Process{});
}

/// TF_GLM_REQUEST_STARTS=1 cuts off the rows TF_KV_SHARE and --learn-conversations keep on: refused with either.
pub fn requestStartsAlone(learn_conversations: bool) !void {
    const on = if (@import("core").profile.get(.TF_GLM_REQUEST_STARTS)) |v| std.mem.eql(u8, v, "1") else false;
    if (on and learn_conversations) return error.GlmRequestStartsWithLearnConversations;
    if (on and try kp.sharing() > 0) return error.GlmRequestStartsWithSharedKv;
}

/// The draft tail is private to the external DFlash2 model and to copied snapshots.
pub fn draftSnapshotMode(enabled: bool, shared: usize, drafts: bool, external: bool) !void {
    if (!enabled) return;
    if (shared != 0) return error.GlmDraftSnapshotWithSharedKv;
    if (!drafts or !external) return error.GlmDraftSnapshotWithoutDrafter;
}

/// Rank-local P2 option checks, run inside startup.open's tp_watch boundary so peers hear refusals immediately.
pub fn validateRankOptions(set: @import("cuda_ranks.zig").Settings, shared: usize, drafts: bool, external: bool, drafter_bits: u8) !void {
    if (set.dflash_q4_down and (!external or !drafts or drafter_bits != 4)) {
        if (!@import("builtin").is_test) std.log.err("TF_GLM_DFLASH_Q4_DOWN=1 requires enabled DFlash2 and --drafter-bits 4", .{});
        return error.BadSetting;
    }
    if (set.dflash_only and (!external or !drafts)) {
        if (!@import("builtin").is_test) std.log.err("TF_GLM_DFLASH_ONLY=1 requires --drafter and enabled drafts", .{});
        return error.BadSetting;
    }
    try draftSnapshotMode(set.dflash_snap_context, shared, drafts, external);
}

test "startup switches: a draft snapshot requires an external drafter and copied KV only when enabled" {
    for ([_]usize{ 0, 1, 8 }) |shared| {
        for ([_]bool{ false, true }) |drafts| {
            for ([_]bool{ false, true }) |external| {
                try draftSnapshotMode(false, shared, drafts, external);
                const expected = draftSnapshotMode(true, shared, drafts, external);
                if (shared > 0) {
                    try std.testing.expectError(error.GlmDraftSnapshotWithSharedKv, expected);
                } else if (!drafts or !external) {
                    try std.testing.expectError(error.GlmDraftSnapshotWithoutDrafter, expected);
                } else try expected;
            }
        }
    }
}

/// TF_GLM_CONTEXT_CEILING: --context past config.json's window, as the attention has no rotary positions to stretch.
pub const ceiling_switch = "TF_GLM_CONTEXT_CEILING";

/// The window TF_GLM_CONTEXT_CEILING raises --context's ceiling to, read once the open has checked every .glm row.
pub fn raisedCeiling() !?u64 {
    try startup();
    return try environment.value(?u64, ceiling_switch, environment.Process.get(.{}, ceiling_switch), null);
}

test {
    _ = @import("cuda_env_test.zig");
}
