//! Synchronous bounded worker over the actual GLM Cursor and credit Ring. Failed DMA drains before unpinning.
const std = @import("std");
const core = @import("core");
const glm = @import("glm");
const cuda = @import("cuda");
const f = @import("format.zig");
const fs = @import("files.zig");
const names = @import("names.zig");
const local = @import("local.zig");
const retained = @import("retained.zig");
const cp = glm.checkpoint;
const cap = core.checkpoint_capture;
const stream = core.checkpoint_stream;

pub const Mode = enum { reference, experimental };
pub const Demand = struct { pages: usize, names_bytes: usize, maintenance_bytes: usize, slab_bytes: usize, disk_bytes: u64 };
pub const Admission = struct {
    ctx: *anyopaque,
    // Validates caller-owned admission; this callback must not acquire resources or mutate reservations.
    admit: *const fn (*anyopaque, Demand) anyerror!void,
};
pub const Transfer = struct {
    ctx: *anyopaque,
    copy: *const fn (*anyopaque, cap.Chunk, []u8) anyerror!void,
    drain: *const fn (*anyopaque) void,
};

/// The caller supplies a pinned slab admitted independently of the RAM prompt cache.
pub const Device = struct {
    stream: cuda.Stream,
    pub fn adapter(d: *Device) Transfer {
        return .{ .ctx = d, .copy = copy, .drain = drain };
    }
    fn copy(ptr: *anyopaque, chunk: cap.Chunk, target: []u8) !void {
        const d: *Device = @ptrCast(@alignCast(ptr));
        switch (chunk.location) {
            .host => |bytes| @memcpy(target, bytes),
            .device => |address| {
                try d.stream.d.check(d.stream.d.api.cuMemcpyDtoHAsync_v2(target.ptr, address, target.len, d.stream.handle), "checkpoint block DtoH");
                try d.stream.synchronize();
            },
        }
    }
    fn drain(ptr: *anyopaque) void {
        const d: *Device = @ptrCast(@alignCast(ptr));
        cuda.completion.drain(d.stream.d, d.stream, "checkpoint block transfer failed to drain");
    }
};

pub const Stats = struct {
    new_pages: u64 = 0,
    repaired_pages: u64 = 0,
    reused_pages: u64 = 0,
    immutable_written: u64 = 0,
    mutable_written: u64 = 0,
    transferred: u64 = 0,
    largest_transfer: usize = 0,
};
pub const Request = struct {
    gpa: std.mem.Allocator,
    saved: *cp.Capture,
    metadata: *const glm.checkpoint_restore.Metadata,
    agreement: f.epoch.Agreement,
    rank: u32,
    input: names.Input,
    slab: []u8,
    transfer: Transfer,
    admission: Admission,
    stats: *Stats,
};

/// Reference mode returns before touching files, capture, adapters or allocating memory.
pub fn save(mode: Mode, files: ?*fs.Files, request: ?Request) !?f.Receipt {
    if (mode == .reference) return null;
    const s = files orelse return error.BlockAdmission;
    const r = request orelse return error.BlockAdmission;
    try s.begin(null);
    defer s.transaction = false;
    errdefer s.recount() catch {
        s.used = s.quota;
    };
    if (r.slab.len < 4 or r.slab.len > f.epoch.max_chunk or r.saved.pages.closed) return error.BlockAdmission;
    if (!std.mem.eql(u8, &r.agreement.binding.prefix, &try names.prefix(r.input, r.metadata))) return error.BlockIdentity;
    var header = try headerOf(r);
    const pages = header.pages;
    const context = names.context(r.agreement, r.rank, r.metadata, header.parts[0..header.count]);
    try r.admission.admit(r.admission.ctx, .{ .pages = pages, .names_bytes = pages * (@sizeOf(f.Name) + @sizeOf(Page)), .maintenance_bytes = s.maintenance_bytes, .slab_bytes = r.slab.len, .disk_bytes = 0 });
    const row_names = try r.gpa.alloc(f.Name, pages);
    defer r.gpa.free(row_names);
    try names.build(r.input, r.metadata, row_names);
    for (row_names) |*name| name.* = f.pageName(context, name.*);
    var index = try retained.Index.init(s);
    defer index.close();
    try index.load(s, r.slab);
    const existing = try r.gpa.alloc(Page, pages);
    defer r.gpa.free(existing);
    const point_name = r.agreement.identity(r.rank).name();
    if (try s.rawExists(r.agreement.manifestName(), ".commit")) return error.BlockAlreadyExists;
    if (try s.rawExists(point_name, ".point") or try s.rawExists(point_name, ".cap")) {
        if (s.readers != 0) return error.BlockBusy;
        try s.remove(point_name, ".point");
        try s.remove(point_name, ".cap");
        try s.publication.sync_dir(s.publication.ctx, s.fd);
        try s.recount();
    }
    var page_bytes: u64 = 0;
    for (header.parts[0..header.count]) |p| page_bytes += p.span;
    var required: u64 = r.saved.frozen.len + f.wire.size(f.Chunks) + @sizeOf(f.rows.Trailer);
    required += @as(u64, pages) * f.wire.size(f.Object) + f.wire.size(f.Header) + @sizeOf(f.rows.Trailer);
    required += f.wire.size(f.epoch.Manifest) + @sizeOf(f.rows.Trailer);
    for (row_names, existing) |name, *page| {
        page.* = .{ .want = index.get(name, .block) };
        if (try s.exists(name, ".blk")) {
            if (page.want) |want| {
                const actual = fs.inspectObject(s, name, ".blk", r.slab) catch |err| failed: {
                    if (!fs.disk.invalid(err)) return err;
                    break :failed null;
                };
                page.reuse = if (actual) |a| std.meta.eql(a, want) else false;
                if (!page.reuse and try s.rawExists(name, ".heal")) return error.BlockRepairExhausted;
            } else {
                if (s.readers != 0) return error.BlockBusy;
                try s.remove(name, ".blk");
                try s.remove(name, ".heal");
            }
        }
        if (!page.reuse) {
            if (page.want != null and s.readers != 0) return error.BlockRepairBusy;
            required = try std.math.add(u64, required, page_bytes + f.wire.size(f.Chunks) + @sizeOf(f.rows.Trailer));
        }
    }
    try s.recount();
    try r.admission.admit(r.admission.ctx, .{ .pages = pages, .names_bytes = pages * (@sizeOf(f.Name) + @sizeOf(Page)), .maintenance_bytes = s.maintenance_bytes, .slab_bytes = r.slab.len, .disk_bytes = required });
    var reservation = try s.reserve(required);
    defer reservation.close();
    var capsule_published = false;
    errdefer if (capsule_published) {
        s.remove(point_name, ".cap") catch {};
        _ = @import("sys").c.fsync(s.fd);
    };
    var point = try s.writer(point_name, ".point");
    defer point.abort();
    for (row_names, existing, 0..) |name, state, page| {
        const object = if (state.reuse) reuse: {
            r.stats.reused_pages += 1;
            break :reuse state.want.?;
        } else new: {
            const object = try writeObject(s, &reservation, r, &header, name, page, state.want);
            if (state.want != null) r.stats.repaired_pages += 1 else r.stats.new_pages += 1;
            r.stats.immutable_written += page_bytes;
            break :new object;
        };
        var bytes: [f.wire.size(f.Object)]u8 = undefined;
        try point.append(try f.wire.encode(object, &bytes));
    }
    header.capsule = try writeObject(s, &reservation, r, &header, point_name, null, null);
    capsule_published = true;
    r.stats.mutable_written += r.saved.frozen.len;
    try header.validate();
    var meta: [f.wire.size(f.Header)]u8 = undefined;
    const metadata = try f.wire.encode(header, &meta);
    const bytes = try s.seal(&point, point_name, .state, @intCast(r.agreement.mark), metadata);
    var opened = try fs.disk.open(s, point.tmp[0..point.tmp_len :0], .state, point_name);
    defer opened.close();
    var read_meta: [f.wire.size(f.Header)]u8 = undefined;
    const actual_meta = try fs.metadata(s, &opened, &read_meta);
    if (!std.mem.eql(u8, metadata, actual_meta)) return error.BlockHash;
    const record = try fs.verify(s, &opened, actual_meta, r.slab);
    _ = try s.publishSealed(&reservation, &point, bytes);
    const receipt: f.Receipt = .{ .agreement = r.agreement, .rank = r.rank, .point = record };
    s.pending = receipt;
    return receipt;
}

const Page = struct { want: ?f.Object = null, reuse: bool = false };

fn headerOf(r: Request) !f.Header {
    const saved = r.saved;
    const m = r.metadata;
    if (saved.pages.ids.len > f.max_pages or saved.plan.count > f.max_parts or saved.plan.len != m.payload or saved.pos != m.mark or saved.mtp_len != m.mtp_len or saved.bases[0] != m.bases[0] or saved.bases[1] != m.bases[1] or saved.cur_len != m.cur_count or (saved.ring_from orelse -1) != m.ring_from) return error.BlockIdentity;
    for (saved.cur[0..saved.cur_len], m.cur[0..m.cur_count]) |a, b| if (a != b) return error.BlockIdentity;
    var h: f.Header = .{ .envelope = try m.envelope(r.agreement, r.rank), .metadata = m.*, .count = @intCast(saved.plan.count), .pages = @intCast(saved.pages.ids.len), .capsule = .{} };
    var block: u64 = 0;
    for (saved.plan.parts[0..saved.plan.count], 0..) |p, i| {
        h.parts[i] = .{ .len = p.len, .span = p.span, .immutable = saved.immutable[i], .capsule = saved.capsule_offsets[i], .block = block };
        block += p.span;
    }
    h.capsule.record.bytes = saved.frozen.len + f.wire.size(f.Chunks) + @sizeOf(f.rows.Trailer);
    try h.validate();
    return h;
}

fn writeObject(s: *fs.Files, reservation: *fs.Reservation, r: Request, header: *const f.Header, name: f.Name, page: ?usize, repair: ?f.Object) !f.Object {
    const ext = if (repair != null) ".heal" else if (page == null) ".cap" else ".blk";
    var w = try s.writer(name, ext);
    defer w.abort();
    var chunks: f.Chunks = .{ .count = header.count };
    for (header.parts[0..header.count], 0..) |part, i| {
        const len: usize = @intCast(if (page != null) part.span else part.len - part.immutable);
        if (len == 0) continue;
        var range: Range = .{ .cursor = .{ .capture = r.saved, .part = i, .done = if (page) |p| p * @as(usize, @intCast(part.span)) else @intCast(part.immutable) }, .remaining = len, .len = len };
        chunks.lengths[i] = len;
        chunks.hashes[i] = try pump(s, r, &range, &w);
    }
    var meta: [f.wire.size(f.Chunks)]u8 = undefined;
    const bytes = try s.seal(&w, name, .block, 0, try f.wire.encode(chunks, &meta));
    var opened = try fs.disk.open(s, w.tmp[0..w.tmp_len :0], .block, name);
    defer opened.close();
    const object = try fs.inspectOpened(s, &opened, name, r.slab);
    if (!std.mem.eql(u8, &object.metadata, &f.wire.hash(chunks))) return error.BlockHash;
    if (repair) |want| if (!std.meta.eql(object, want)) return error.BlockRepairMismatch;
    _ = try s.publishSealed(reservation, &w, bytes);
    return object;
}

const Range = struct {
    cursor: cp.Cursor,
    remaining: usize,
    len: usize,
    fn next(ptr: *anyopaque, limit: usize) !?cap.Chunk {
        const r: *Range = @ptrCast(@alignCast(ptr));
        if (r.remaining == 0) return null;
        const source = r.cursor.source();
        const chunk = (try source.next(source.ctx, @min(limit, r.remaining))) orelse return error.ShortSource;
        r.remaining -= chunk.len;
        return chunk;
    }
};

fn pump(s: *fs.Files, r: Request, range: *Range, writer: *f.rows.Writer) !f.Name {
    var slots: [1]stream.Slot = undefined;
    var ring = try stream.Ring.init(.{ .owner = &r.saved.pages, .ctx = range, .len = range.len, .next = Range.next }, r.slab, &slots, r.slab.len);
    defer {
        if (!ring.readyToSeal()) ring.cancel();
        ring.close() catch unreachable;
    }
    var sha = f.Sha.init(.{});
    while (ring.emitted < range.len) {
        try s.control.check();
        const ticket = try ring.acquire();
        r.transfer.copy(r.transfer.ctx, ticket.chunk, ticket.target) catch |err| {
            r.transfer.drain(r.transfer.ctx);
            try ring.dmaDone(ticket.key, false);
            try ring.retire(ticket.key);
            return err;
        };
        try ring.dmaDone(ticket.key, true);
        const bytes = try ring.beginWrite(ticket.key);
        writer.append(bytes) catch |err| {
            try ring.ioDone(ticket.key, false);
            try ring.retire(ticket.key);
            return err;
        };
        sha.update(bytes);
        r.stats.transferred += bytes.len;
        r.stats.largest_transfer = @max(r.stats.largest_transfer, bytes.len);
        try ring.ioDone(ticket.key, true);
        try ring.retire(ticket.key);
    }
    return sha.finalResult();
}
