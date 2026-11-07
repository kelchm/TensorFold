//! All-rank checkpoint commit records. Rank files alone are never recovery candidates.
//! Single coordinator per epoch; store-owned directories and immutable published files are required.
const std = @import("std");
const sys = @import("sys");
const rank_file = @import("checkpoint_file.zig");
pub const wire = @import("checkpoint_wire.zig");
const Sha = std.crypto.hash.sha2.Sha256;
pub const max_ranks = 64;
pub const max_metadata = 8192;
pub const max_chunk = 2 << 20;

pub const Binding = struct {
    model: [32]u8,
    drafter: [32]u8,
    arithmetic: [32]u8,
    layout: [32]u8,
    media: [32]u8,
    prefix: [32]u8, // request/tokens identity supplied by the rows owner, not a pathname
};

pub const Agreement = struct {
    version: u32 = 1,
    world: u32,
    epoch: u64,
    mark: u64,
    binding: Binding,

    pub fn validate(a: Agreement) !void {
        inline for (.{ "model", "arithmetic", "layout", "media", "prefix" }) |field| {
            if (std.mem.allEqual(u8, &@field(a.binding, field), 0)) return error.CheckpointAgreement;
        }
        if (a.version != 1 or a.world == 0 or a.world > max_ranks or a.epoch == 0 or a.mark > std.math.maxInt(i32)) return error.CheckpointAgreement;
    }
    pub fn digest(a: Agreement) [32]u8 {
        return wire.hash(a);
    }
    /// One publication slot per epoch and binding, even if a stale coordinator proposes another mark.
    pub fn manifestName(a: Agreement) [32]u8 {
        var key = a;
        key.mark = 0;
        return wire.hash(key);
    }
    pub fn identity(a: Agreement, rank: u32) rank_file.Identity {
        return .{ .block = a.digest(), .rank = rank, .epoch = a.epoch };
    }
};

pub const Envelope = struct {
    magic: u64 = 0x31524b4346474c4d,
    version: u32 = 1,
    rank: u32,
    agreement: Agreement,
    payload: u64,
    detail_bytes: u32,
};

pub const Record = struct { bytes: u64 = 0, digest: [32]u8 = @splat(0) };
pub const Manifest = struct {
    magic: u64 = 0x31454b4346474c4d,
    version: u32 = 1,
    agreement: Agreement,
    ranks: [max_ranks]Record = @splat(.{}),
};

/// Checked between bounded I/O operations, including immediately before publication.
pub const Control = struct {
    ctx: ?*anyopaque = null,
    check_fn: *const fn (?*anyopaque) anyerror!void = pass,
    fn pass(_: ?*anyopaque) !void {}
    pub fn check(c: Control) !void {
        try c.check_fn(c.ctx);
    }
};

pub fn Store(comptime rows: type) type {
    return struct {
        const Self = @This();
        dir: [:0]const u8,
        validate_detail: *const fn (Envelope, []const u8) anyerror!void,
        control: Control = .{},
        io: rows.IO = .{},
        publication: rank_file.Publication = .{},

        /// Metadata and one caller-owned transfer slab are the only host staging. The slab must be pinned for CUDA.
        pub const Reader = struct {
            opened: rows.Opened,
            metadata: [max_metadata]u8 = undefined,
            meta_len: usize,
            envelope: Envelope,
            expected: ?Record,
            offset: u64 = 0,
            hash: std.hash.Wyhash = .init(0),
            sha: Sha = .init(.{}),
            control: Control,
            finished: bool = false,

            pub fn check(r: *const Reader) !void {
                try r.control.check();
            }
            pub fn durable(r: *const Reader) bool {
                return r.expected != null;
            }
            pub fn detail(r: *const Reader) []const u8 {
                return r.metadata[wire.size(Envelope)..r.meta_len];
            }
            pub fn close(r: *Reader) void {
                if (r.opened.fd >= 0) r.opened.close();
            }
            pub fn read(r: *Reader, into: []u8) !usize {
                try r.control.check();
                if (r.finished or into.len == 0 or into.len > max_chunk) return error.CheckpointReadState;
                const n: usize = @intCast(@min(into.len, r.envelope.payload - r.offset));
                if (n == 0) return 0;
                if (!rows.readUnaligned(r.opened.fd, into[0..n], r.offset)) return error.RowsRead;
                r.hash.update(into[0..n]);
                r.sha.update(into[0..n]);
                r.offset += n;
                return n;
            }
            pub fn finish(r: *Reader) !Record {
                try r.control.check();
                if (r.finished or r.offset != r.envelope.payload) return error.CheckpointReadState;
                if (r.hash.final() != r.opened.trailer.payload_hash) return error.RowsHash;
                r.sha.update(r.metadata[0..r.meta_len]);
                r.sha.update(std.mem.asBytes(&r.opened.trailer));
                const record: Record = .{ .bytes = r.envelope.payload + r.meta_len + @sizeOf(rows.Trailer), .digest = r.sha.finalResult() };
                if (r.expected) |expected| if (!std.meta.eql(record, expected)) return error.CheckpointRankHash;
                r.finished = true;
                return record;
            }
            fn verify(r: *Reader, slab: []u8) !Record {
                while (try r.read(slab) != 0) {}
                return r.finish();
            }
        };

        /// A proof returned only after every rank has been read and checked. Reopening a rank checks it again.
        /// Borrowed directory/control outlive this proof. A scheduler still owes the all-rank restore activation barrier.
        pub const Epoch = struct {
            manifest: Manifest,
            store: Self,
            pub fn openRank(e: *const Epoch, rank: u32) !Reader {
                if (rank >= e.manifest.agreement.world) return error.CheckpointRank;
                return e.store.openRank(e.manifest.agreement, rank, e.manifest.ranks[rank]);
            }
        };

        fn openRank(s: Self, agreement: Agreement, rank: u32, expected: ?Record) !Reader {
            try agreement.validate();
            if (rank >= agreement.world) return error.CheckpointRank;
            var path: [1200]u8 = undefined;
            const stem = std.fmt.bytesToHex(agreement.identity(rank).name(), .lower);
            var opened = try rows.open(try rows.path(&path, s.dir, &stem, ".rank", false), .state, agreement.identity(rank).name());
            errdefer opened.close();
            const tr = opened.trailer;
            if (tr.meta < wire.size(Envelope) or tr.meta > max_metadata or tr.at != agreement.mark or tr.flags != 0 or tr.block != 0 or tr.reserved != 0) return error.CheckpointMetadataSize;
            var r: Reader = .{ .opened = opened, .meta_len = @intCast(tr.meta), .envelope = undefined, .expected = expected, .control = s.control };
            if (!rows.readUnaligned(opened.fd, r.metadata[0..r.meta_len], tr.payload)) return error.RowsRead;
            if (std.hash.Wyhash.hash(1, r.metadata[0..r.meta_len]) != tr.meta_hash) return error.RowsHash;
            r.envelope = try wire.decode(Envelope, r.metadata[0..wire.size(Envelope)]);
            const e = r.envelope;
            if (e.magic != 0x31524b4346474c4d or e.version != 1 or e.rank != rank or !std.meta.eql(e.agreement, agreement) or e.payload != tr.payload or e.detail_bytes != r.meta_len - wire.size(Envelope)) return error.CheckpointRankIdentity;
            if (expected) |record| if (record.bytes != tr.payload + tr.meta + @sizeOf(rows.Trailer)) return error.CheckpointRankHash;
            try s.validate_detail(e, r.detail());
            return r;
        }

        pub const Coordinator = struct {
            store: Self,
            manifest: Manifest,
            seen: u64 = 0,
            phase: enum { collecting, failed, committed } = .collecting,

            pub fn init(store: Self, agreement: Agreement) !Coordinator {
                try agreement.validate();
                return .{ .store = store, .manifest = .{ .agreement = agreement } };
            }
            pub fn cancel(c: *Coordinator) void {
                if (c.phase == .collecting) c.phase = .failed;
            }
            pub fn accept(c: *Coordinator, local: rank_file.Local, slab: []u8) !void {
                if (c.phase != .collecting) return error.CheckpointEpochState;
                errdefer c.phase = .failed;
                try c.store.control.check();
                const a = c.manifest.agreement;
                const rank = local.identity.rank;
                if (rank >= a.world or !std.meta.eql(local.identity, a.identity(rank))) return error.CheckpointRankIdentity;
                const bit = @as(u64, 1) << @as(u6, @intCast(rank));
                if (c.seen & bit != 0) return error.CheckpointDuplicateRank;
                var r = try c.store.openRank(a, rank, null);
                defer r.close();
                const record = try r.verify(slab);
                if (local.bytes != record.bytes) return error.CheckpointRankHash;
                c.manifest.ranks[rank] = record;
                c.seen |= bit;
            }
            pub fn commit(c: *Coordinator, nonce: u64, slab: []u8) !Epoch {
                if (c.phase != .collecting) return error.CheckpointEpochState;
                errdefer c.phase = .failed;
                const a = c.manifest.agreement;
                const all = @as(u64, std.math.maxInt(u64)) >> @as(u6, @intCast(max_ranks - a.world));
                if (c.seen != all) return error.CheckpointMissingRank;
                // Repeat verification and sync *each* file before the directory barrier and manifest publication.
                for (0..a.world) |rank| {
                    var r = try c.store.openRank(a, @intCast(rank), c.manifest.ranks[rank]);
                    defer r.close();
                    _ = try r.verify(slab);
                    try c.store.io.sync(c.store.io.ctx, r.opened.fd);
                }
                const fd = sys.c.open(c.store.dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(sys.c.mode_t, 0));
                if (fd < 0) return error.CheckpointDirectory;
                defer _ = sys.c.close(fd);
                try c.store.publication.sync_dir(c.store.publication.ctx, fd);
                var data: [wire.size(Manifest)]u8 = undefined;
                const bytes = try wire.encode(c.manifest, &data);
                const stem = std.fmt.bytesToHex(a.manifestName(), .lower);
                var w = try rows.Writer.openUnique(c.store.dir, &stem, ".epoch", nonce);
                defer w.abort();
                w.io = c.store.io;
                try w.append(bytes);
                _ = try w.seal(.{ .kind = @backingInt(rows.Kind.state), .payload = 0, .payload_hash = 0, .name = a.digest(), .at = @intCast(a.mark) }, &.{}, true);
                try c.store.control.check();
                const dest = w.dest[0..w.dest_len :0];
                try c.store.publication.link(c.store.publication.ctx, w.tmp[0..w.tmp_len :0], dest);
                errdefer {
                    _ = sys.c.unlink(dest);
                    // An uncertain failed publication never removes a previous epoch. Attempt to persist rollback.
                    _ = sys.c.fsync(fd);
                }
                if (sys.c.unlink(w.tmp[0..w.tmp_len :0]) != 0) return error.CheckpointPublish;
                try c.store.control.check();
                try c.store.publication.sync_dir(c.store.publication.ctx, fd);
                c.phase = .committed;
                return .{ .manifest = c.manifest, .store = c.store };
            }
        };

        fn candidate(s: Self, path: [:0]const u8, filename: []const u8, binding: Binding, world: u32, slab: []u8) !Epoch {
            var f = try rows.open(path, .state, null);
            defer f.close();
            if (f.trailer.payload != wire.size(Manifest) or f.trailer.meta != 0 or f.trailer.flags != 0 or f.trailer.block != 0 or f.trailer.reserved != 0) return error.CheckpointManifest;
            var bytes: [wire.size(Manifest)]u8 = undefined;
            if (!rows.readUnaligned(f.fd, &bytes, 0)) return error.RowsRead;
            if (std.hash.Wyhash.hash(0, &bytes) != f.trailer.payload_hash or f.trailer.meta_hash != std.hash.Wyhash.hash(1, "")) return error.RowsHash;
            const m = try wire.decode(Manifest, &bytes);
            if (m.magic != 0x31454b4346474c4d or m.version != 1) return error.CheckpointManifest;
            try m.agreement.validate();
            const a = m.agreement;
            const stem = std.fmt.bytesToHex(a.manifestName(), .lower);
            if (filename.len != 70 or !std.mem.eql(u8, filename[0..64], &stem) or !std.meta.eql(a.binding, binding) or a.world != world or a.mark != f.trailer.at or !std.mem.eql(u8, &a.digest(), &f.trailer.name)) return error.CheckpointRankIdentity;
            for (m.ranks[a.world..]) |r| if (!std.meta.eql(r, Record{})) return error.CheckpointManifest;
            for (0..a.world) |rank| {
                var r = try s.openRank(a, @intCast(rank), m.ranks[rank]);
                defer r.close();
                _ = try r.verify(slab);
            }
            return .{ .manifest = m, .store = s };
        }

        /// Scan committed manifests, not rank receipts or temp files. Broken/newer epochs cannot mask older valid ones.
        /// Memory is O(max_ranks + max_metadata + chunk), independent of epoch count and payload size.
        pub fn recover(s: Self, binding: Binding, world: u32, slab: []u8) !?Epoch {
            if (slab.len == 0 or slab.len > max_chunk or world == 0 or world > max_ranks) return error.CheckpointReadState;
            const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
            defer _ = sys.c.closedir(dir);
            var best: ?Epoch = null;
            while (true) {
                try s.control.check();
                const entry = sys.c.readdir(dir) orelse break;
                const name = std.mem.sliceTo(&entry.name, 0);
                if (name.len != 70 or !std.mem.endsWith(u8, name, ".epoch")) continue;
                var path: [1200]u8 = undefined;
                const p = try std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ s.dir, name }, 0);
                const found = s.candidate(p, name, binding, world, slab) catch |err| {
                    // Cancellation is not a corrupt file and must not silently select an older state.
                    if (err == error.Cancelled) return err;
                    continue;
                };
                if (best == null or found.manifest.agreement.epoch > best.?.manifest.agreement.epoch) best = found;
            }
            return best;
        }
    };
}
