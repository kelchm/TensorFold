//! Rank-local chunks: exclusive temporary, file fsync, no-replace link, directory fsync; no all-rank commit.
const std = @import("std");
const platform = @import("sys");
const transfer = @import("checkpoint_stream.zig");

pub const Identity = struct {
    block: [32]u8, // rows_chain.Name, supplied by the row store owning that module
    rank: u32,
    epoch: u64,

    /// The caller's existing rows/block identity, domain separated by the rank and checkpoint epoch.
    pub fn name(id: Identity) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("tensorfold checkpoint rank 1");
        h.update(&id.block);
        var words: [12]u8 = undefined;
        std.mem.writeInt(u32, words[0..4], id.rank, .little);
        std.mem.writeInt(u64, words[4..12], id.epoch, .little);
        h.update(&words);
        return h.finalResult();
    }
};

/// Publication hooks allow deterministic filesystem-failure tests without global fault switches.
pub const Publication = struct {
    ctx: ?*anyopaque = null,
    link: *const fn (?*anyopaque, [:0]const u8, [:0]const u8) anyerror!void = linkDefault,
    sync_dir: *const fn (?*anyopaque, c_int) anyerror!void = syncDefault,

    fn linkDefault(_: ?*anyopaque, from: [:0]const u8, to: [:0]const u8) !void {
        if (std.c.link(from, to) != 0) return error.CheckpointPublish;
    }
    fn syncDefault(_: ?*anyopaque, fd: c_int) !void {
        if (platform.c.fsync(fd) != 0) return error.CheckpointDirectorySync;
    }
};

pub const Local = struct { identity: Identity, bytes: u64 };
const Phase = enum { writing, sealed, published, failed, closed };

/// Instantiate with the existing row store's file module, preserving its single module owner.
pub fn Writer(comptime rows: type) type {
    return struct {
        const Self = @This();
        file: rows.Writer,
        dir_fd: c_int,
        identity: Identity,
        expected: u64,
        metadata_limit: usize,
        publication: Publication = .{},
        phase: Phase = .writing,
        linked: bool = false,
        bytes: u64 = 0,

        /// The directory already exists and belongs to this store. Rank files are invisible to the serving index.
        pub fn open(dir: [:0]const u8, identity: Identity, nonce: u64, payload_bytes: u64, metadata_limit: usize) !Self {
            if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.UnsupportedCheckpointFilesystem;
            const fd = platform.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
            if (fd < 0) return error.CheckpointDirectory;
            errdefer _ = platform.c.close(fd);
            const stem = std.fmt.bytesToHex(identity.name(), .lower);
            const file = try rows.Writer.openUnique(dir, &stem, ".rank", nonce);
            return .{ .file = file, .dir_fd = fd, .identity = identity, .expected = payload_bytes, .metadata_limit = metadata_limit };
        }

        pub fn append(w: *Self, offset: u64, bytes: []const u8) !void {
            if (w.phase != .writing) return error.CheckpointFileState;
            if (offset != w.file.payload) return error.CheckpointSinkBackpressure;
            if (bytes.len > w.expected - w.file.payload) return error.CheckpointFileSize;
            w.file.append(bytes) catch |err| {
                w.phase = .failed;
                return err;
            };
        }

        /// Sequential sink for ready ring slots; an out-of-order completion waits without borrowing its buffer.
        pub fn consume(w: *Self, ring: *transfer.Ring, key: transfer.Key) !void {
            const part = try ring.inspect(key);
            if (part.offset != w.file.payload) return error.CheckpointSinkBackpressure;
            const bytes = try ring.beginWrite(key);
            w.append(part.offset, bytes) catch |err| {
                try ring.ioDone(key, false);
                return err;
            };
            try ring.ioDone(key, true);
        }

        /// Every chunk is drained before metadata/trailer and file fsync; no complete-payload staging buffer exists.
        pub fn seal(w: *Self, ring: *const transfer.Ring, at: u32, metadata: []const []const u8) !void {
            if (w.phase != .writing) return error.CheckpointFileState;
            if (!ring.readyToSeal() or ring.written != w.expected or w.file.payload != w.expected) return error.CheckpointBusy;
            var n: usize = 0;
            for (metadata) |m| {
                if (m.len > w.metadata_limit - n) return error.CheckpointMetadataLimit;
                n += m.len;
            }
            w.bytes = w.file.seal(.{ .kind = @backingInt(rows.Kind.state), .payload = 0, .payload_hash = 0, .name = w.identity.name(), .at = at }, metadata, true) catch |err| {
                w.phase = .failed;
                return err;
            };
            w.phase = .sealed;
        }

        /// Hard-link publication never overwrites an existing epoch file; no receipt escapes a failed directory sync.
        pub fn publish(w: *Self) !Local {
            if (w.phase != .sealed) return error.CheckpointFileState;
            errdefer w.phase = .failed;
            try w.publication.link(w.publication.ctx, w.file.tmp[0..w.file.tmp_len :0], w.file.dest[0..w.file.dest_len :0]);
            w.linked = true;
            if (platform.c.unlink(w.file.tmp[0..w.file.tmp_len :0]) != 0) return error.CheckpointPublish;
            try w.publication.sync_dir(w.publication.ctx, w.dir_fd);
            w.phase = .published;
            return .{ .identity = w.identity, .bytes = w.bytes };
        }

        /// A failed writer still requires the ring to drain before capture ownership can end.
        pub fn close(w: *Self) void {
            if (w.phase == .closed) return;
            if (w.phase != .published) {
                w.file.abort();
                if (w.linked) _ = platform.c.unlink(w.file.dest[0..w.file.dest_len :0]);
            }
            _ = platform.c.close(w.dir_fd);
            w.phase = .closed;
        }
    };
}
