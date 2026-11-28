//! Shared GLM rank state descriptions, kept separate from engine operations.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const sampler = @import("cuda_sampler.zig");

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// One stream's window of a shared verify: its slot, its ids, and its drawn tokens.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32, bits: u64 = 0 };

/// A kept prompt state as the prompt cache holds it: the id every rank keeps it under, its position, its chunks below.
pub const Kept = struct {
    id: u64,
    at: u32,
    below: u64 = 0, // cuda_prompt_cuts.below of the pass that kept it
    executed: []u32 = &.{}, // marks mode: exact executed chunk starts strictly below `at` (owned by e.gpa)
    has_executed: bool = false, // distinguishes empty executed (&.{}) from untracked non-marks states
};

/// A stream's own sequence: state, sampling rule, MTP drafts; buffers are shared, one stream's round at a time.
pub const Seq = struct {
    st: fw.State,
    more: []fw.State = &.{}, // full GLM-5.3 at TP=1: its state for each other share the engine holds
    sampling: ?sampler.Sampling = null, // null: greedy
    in_head: usize = 0, // drafted rows the MTP caches hold past the kept ones (st.mtp_drafted)
    rows_at: u64 = 0, // its last verify's final-norm rows (bf16 [rows][hidden]), which the next draft reads
    cuts: std.ArrayList(u32) = .empty, // the request's chunk starts after 0 (TF_GLM_REQUEST_STARTS), every rank's copy
    resumed_starts: std.ArrayList(u32) = .empty, // marks mode: actual executed starts below resumed_at
    resumed_at: u32 = 0, // marks mode: position this pass resumed from (0: cold)
    track_executed: bool = false, // marks mode: record executed starts in Kept and Snap.below

    pub fn resetMeta(q: *Seq) void {
        q.cuts.clearRetainingCapacity();
        q.resumed_starts.clearRetainingCapacity();
        q.resumed_at = 0;
        q.track_executed = false;
    }

    pub fn deinitMeta(q: *Seq, gpa: std.mem.Allocator) void {
        q.cuts.deinit(gpa);
        q.resumed_starts.deinit(gpa);
    }

    pub fn setExecuted(q: *Seq, gpa: std.mem.Allocator, resumed: ?*const Kept) !void {
        q.resumed_starts.clearRetainingCapacity();
        if (resumed) |k| {
            try q.resumed_starts.appendSlice(gpa, k.executed);
            q.resumed_at = k.at;
        } else q.resumed_at = 0;
        q.track_executed = true;
    }
};

/// One rank's copy of a kept prompt state: the state arena's bytes and the host fields beside them.
pub const Snap = struct { lease: @import("cuda_snapshot_pages.zig").Lease, len: usize, pos: usize, mtp_len: usize, cur: []u1, bases: [2]usize, shared: usize = 0, packed_len: usize = 0, draft: @import("dflash_snapshot.zig").Saved = .{}, below: u64 = 0 };

/// Where a prompt pass resumes: a kept state (every rank holds `id`) at `at`, a cold pass's cut (cuda_prompt_cuts).
pub const Resume = struct { id: u64, at: usize };

/// Rank 0's marks in a prompt pass: `call` may keep the state at a chunk start; other ranks copy at that chunk's gate.
pub const Marks = struct {
    at: []const u32,
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, at: u32) void,
    interrupted: ?*const fn (ctx: *anyopaque, at: u32) void = null,
};

const ranks = @import("cuda_ranks.zig");
const images_mod = @import("cuda_images.zig");

/// Shared decode verify window capacity (rows across concurrent streams).
pub const shared_rows = 32;

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub const Options = struct {
    capacity: usize, // tokens a sequence holds
    mtp: bool, // load the MTP head (drafts)
    settings: ranks.Settings,
    vision: ?images_mod.Limits = null, // --vision's limits (null: off): every rank must hold the rows rank 0 sends
    drafter: bool = false, // DFlash2 loads after rank agreement
    drafter_dir: ?[]const u8 = null, // --drafter's folder: its digest enters rank agreement before weights load
    drafter_bits: u8 = 0, // --drafter-bits (4 or 0 when drafter is enabled): ranks agree on draft precision
    owner: ?*cuda.tp_group.Group = null, // the communicator's owner, which the backend aborts (null: the caller's own)
    room: ?Room = null, // remaining allocation budget, including the OS reserve and process limit
};
