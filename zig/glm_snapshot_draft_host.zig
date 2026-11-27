//! GLM DFlash2 640-row prompt context snapshot overlay and disk-snapshot compatibility host tests.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const t = std.testing;

test "GLM snapshots: copied draft KV keeps a wrapped initialized tail across slots on each rank" {
    const snap = glm.dflash.snapshot;
    for (0..2) |rank| {
        var d = rec.init();
        var f: Fixture = undefined;
        try f.init(&d, rank, 2, false);
        defer f.deinit();
        var dr: glm.dflash.Drafter = undefined;
        dr.d = &d;
        dr.gpa = t.allocator;
        dr.slots = .empty;
        defer {
            for (dr.slots.items) |*sl| if (sl.ring) |*r| r.free();
            dr.slots.deinit(t.allocator);
        }
        f.e.drafter = &dr;
        try dr.restart(0, 0);
        try dr.restart(1, 0);
        const src = dr.slots.items[0].ring.?;
        const dst = dr.slots.items[1].ring.?;
        const at = 2100;
        try f.mark(at, 0, 11);
        const target = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(target);
        const source = rec.bytes(src.ptr, src.len);
        for (0..snap.bands * snap.window) |i| {
            const byte: u8 = @intCast((i * 31 + rank + 1) % 239);
            @memset(source[i * snap.stride ..][0..snap.stride], byte);
        }
        const dest = rec.bytes(dst.ptr, dst.len);
        @memset(dest, 255);
        const valid = &dr.slots.items[0].prompt_history;
        valid.append(200, 100);
        valid.append(1350, 500); // a hole; only the new extent can be attributed to this prompt
        try t.expectEqual(@as(usize, 0), valid.rows(at));
        valid.append(1850, 250);
        try t.expectEqual(@as(usize, snap.cap), valid.rows(at));
        const baseline = f.e.stateBytes(at);
        f.e.settings.dflash_snap_context = true;
        f.e.drafter = null;
        try t.expectEqual(baseline, f.e.stateBytes(at));
        f.e.drafter = &dr;
        const budget = f.e.stateBytes(at);
        try t.expect(budget >= baseline + snap.bytes(snap.cap) - (2 << 20));
        try f.e.prepareSnapshots(budget);
        try f.e.keepAs(1);
        const saved = f.e.snaps.get(1).?;
        try t.expectEqual(@as(usize, at), saved.draft.end);
        try t.expectEqual(@as(usize, snap.cap), saved.draft.count);
        try t.expectEqual(snap.bytes(snap.cap), saved.packed_len - saved.len);
        try t.expectEqual(budget, saved.lease.ids.items.len * (2 << 20));
        const original_copies = rec.copies;
        f.e.slot = 1;
        try f.mark(3999, 0, 77);
        dr.slots.items[1].v_rows = 4;
        rec.fail_copy = rec.copies + original_copies; // failure in the last tail range
        try t.expectError(error.NotReady, f.e.restoreState(1));
        try t.expectEqual(@as(usize, 0), dr.slots.items[1].prompt_history.rows(at));
        try t.expectEqual(@as(usize, 0), dr.slots.items[1].v_rows);
        rec.fail_copy = null;
        try f.e.restoreState(1);
        try f.expectPrefix(target, at, 0);
        try t.expectEqual(@as(i64, at - snap.cap), dr.slots.items[1].from);
        try t.expectEqual(@as(usize, snap.cap), dr.slots.items[1].prompt_history.rows(at));
        for (0..snap.bands * snap.window) |i| {
            const slot = i % snap.window;
            const in_tail = slot >= at - snap.cap or slot < at % snap.window;
            const want: u8 = if (in_tail) @intCast((i * 31 + rank + 1) % 239) else 255;
            try t.expectEqual(want, dest[i * snap.stride]);
            try t.expectEqual(want, dest[(i + 1) * snap.stride - 1]);
        }
        f.e.snaps.getPtr(1).?.draft.end -= 1;
        @memset(dest, 253);
        try f.e.restoreState(1);
        try t.expectEqual(@as(i64, at), dr.slots.items[1].from);
        try t.expectEqual(@as(usize, 0), dr.slots.items[1].prompt_history.rows(at));
        try t.expectEqual(@as(u8, 253), dest[0]);
        f.e.dropState(1);
        try t.expectEqual(budget, f.e.snap_pages.?.spare());
        f.e.slot = 0;
        try f.mark(at, 0, 11);
        rec.fail_copy = rec.copies + original_copies;
        try t.expectError(error.NotReady, f.e.keepAs(2));
        try t.expect(!f.e.holds(2));
        try t.expectEqual(budget, f.e.snap_pages.?.spare());
        rec.fail_copy = null;
        try f.e.keepAs(2);
        try t.expectEqual(@as(usize, snap.cap), f.e.snaps.get(2).?.draft.count);
        f.e.dropState(2);
    }
}

// CPU stand-in for prompt ingestion; each ring row is identifiable by band and position.
const DraftPrompt = struct {
    dr: *glm.dflash.Drafter,
    taps_prefill: u64 = 0,
    tap_rows: usize = glm.dflash.MAX_CTX,

    pub fn slot(p: *@This(), i: u32) !*glm.dflash.Slot {
        return p.dr.slot(i);
    }

    fn value(pos: usize, band: usize) u8 {
        return @intCast((pos * 11 + 29 * band) % 239);
    }

    pub fn ingest(_: *@This(), _: void, runs: []const glm.dflash.Run) !void {
        const sn = glm.dflash.snapshot;
        for (runs) |r| {
            const start: usize = @intCast(r.pos0);
            for (start..start + r.n) |pos| {
                for (0..sn.bands) |band| {
                    const offset = (band * sn.window + pos % sn.window) * sn.stride;
                    @memset(rec.bytes(r.ring + offset, sn.stride), value(pos, band));
                }
            }
        }
    }

    fn expectRange(ring: cuda.DeviceBuffer, end: usize, count: usize) !void {
        const sn = glm.dflash.snapshot;
        for (0..sn.bands) |band| {
            for (0..sn.window) |row| {
                // Distance backwards from the last committed row, including that row.
                const distance = (end + sn.window - 1 - row) % sn.window;
                const byte = if (distance < count) value(end - 1 - distance, band) else 255;
                const data = rec.bytes(ring.ptr + (band * sn.window + row) * sn.stride, sn.stride);
                try t.expectEqual(byte, data[0]);
                try t.expectEqual(byte, data[data.len - 1]);
            }
        }
    }
};

test "GLM snapshots: admission ignores the bound slot and chained prompt keeps charge only realized rows" {
    const sn = glm.dflash.snapshot;
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    var dr: glm.dflash.Drafter = undefined;
    dr.d = &d;
    dr.gpa = t.allocator;
    dr.slots = .empty;
    defer {
        for (dr.slots.items) |*sl| if (sl.ring) |*r| r.free();
        dr.slots.deinit(t.allocator);
    }
    f.e.drafter = &dr;
    try dr.restart(0, 0);
    try dr.restart(1, 0);
    const src = dr.slots.items[0].ring.?;
    const dst = dr.slots.items[1].ring.?;
    var prompt: DraftPrompt = .{ .dr = &dr };
    f.e.settings.dflash_snap_context = true;
    try f.e.prepareSnapshots(3 * f.e.stateBytes(3000));

    for ([_]usize{ 63, 2048, 2049 }, [_]usize{ 17, 640, 411 }) |at, rows| {
        try dr.restart(0, at - rows);
        try dr.restart(1, 0);
        @memset(rec.bytes(src.ptr, src.len), 254);
        @memset(rec.bytes(dst.ptr, dst.len), 255);
        try glm.dflash.ingestPrompt(&prompt, {}, 0, at - rows, rows, at, 0);
        try f.mark(at, 0, 17);

        // A's initialized history, B's empty history, and an uncreated slot yield the same promise.
        f.e.slot = 0;
        const promised = f.e.stateBytes(at);
        for ([_]u32{ 1, 5 }) |other| {
            f.e.slot = other;
            try t.expectEqual(promised, f.e.stateBytes(at));
        }
        f.e.slot = 1;
        try f.e.keepAs(10);
        const empty = f.e.snaps.get(10).?;
        try t.expectEqual(@as(usize, 0), empty.draft.count);
        try t.expectEqual(try glm.snapshots.pages.rounded(empty.len), empty.lease.ids.items.len * glm.snapshots.pages.page);
        try t.expectEqual(try glm.snapshots.pages.rounded(empty.len + sn.bytes(@min(at, sn.cap))), promised);
        f.e.dropState(10);

        f.e.slot = 0;
        try f.e.keepAs(11);
        const first = f.e.snaps.get(11).?;
        try t.expectEqual(rows, first.draft.count);
        try t.expectEqual(try glm.snapshots.pages.rounded(first.len + sn.bytes(rows)), first.lease.ids.items.len * glm.snapshots.pages.page);
        f.e.slot = 1;
        try f.e.restoreState(11);
        try DraftPrompt.expectRange(dst, at, rows);
        try t.expectEqual(@as(i64, @intCast(at - rows)), dr.slots.items[1].from);
        try glm.dflash.ingestPrompt(&prompt, {}, 1, at, 5, at + 5, 0);
        try f.mark(at + 5, 0, 47);
        try f.e.keepAs(12);
        const extended = @min(sn.cap, rows + 5);
        try t.expectEqual(extended, f.e.snaps.get(12).?.draft.count);
        @memset(rec.bytes(src.ptr, src.len), 255);
        f.e.slot = 0;
        try f.e.restoreState(12);
        try DraftPrompt.expectRange(src, at + 5, extended);
        try t.expectEqual(@as(i64, @intCast(at + 5 - extended)), dr.slots.items[0].from);
        f.e.dropState(11);
        f.e.dropState(12);
    }

    // Disabling the switch or sharing KV keeps target-only copies and cold draft restarts.
    const at = 63;
    try f.mark(at, 0, 14);
    try dr.restart(0, 0);
    try glm.dflash.ingestPrompt(&prompt, {}, 0, 0, at, at, 0);
    f.e.slot = 0;
    f.e.settings.dflash_snap_context = false;
    const baseline = f.e.stateBytes(at);
    for ([_]bool{ false, true }) |shared| {
        f.e.settings.dflash_snap_context = shared;
        f.e.share_pages = shared;
        f.e.slot = 0;
        try t.expectEqual(baseline, f.e.stateBytes(at));
        try f.e.keepAs(20);
        const s = f.e.snaps.get(20).?;
        try t.expectEqual(s.len, s.packed_len);
        try t.expectEqual(@as(usize, 0), s.draft.count);
        try t.expectEqual(baseline, s.lease.ids.items.len * glm.snapshots.pages.page);
        f.e.slot = 1;
        @memset(rec.bytes(dst.ptr, dst.len), 255);
        try f.e.restoreState(20);
        try DraftPrompt.expectRange(dst, at, 0);
        try t.expectEqual(@as(i64, at), dr.slots.items[1].from);
        f.e.dropState(20);
    }
}

test "GLM snapshots: disk blankAs/seal/restore stays target-only and cold-restarts DFlash2 alongside draft-tail RAM keeps" {
    const HostCopy = struct {
        fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
            @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], rec.bytes(src, n));
            return 0;
        }
        fn htod(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
            @memcpy(rec.bytes(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
            return 0;
        }
        fn ctxDev(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
            out.* = 0;
            return 0;
        }
        fn devAttr(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            out.* = 0;
            return 0;
        }
    };
    const sn = glm.dflash.snapshot;
    var d = rec.init();
    d.api.cuMemcpyDtoH_v2 = HostCopy.dtoh;
    d.api.cuMemcpyHtoD_v2 = HostCopy.htod;
    d.api.cuCtxGetDevice = HostCopy.ctxDev;
    d.api.cuDeviceGetAttribute = HostCopy.devAttr;
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    var dr: glm.dflash.Drafter = undefined;
    dr.d = &d;
    dr.gpa = t.allocator;
    dr.slots = .empty;
    defer {
        for (dr.slots.items) |*sl| if (sl.ring) |*r| r.free();
        dr.slots.deinit(t.allocator);
    }
    f.e.drafter = &dr;
    f.e.settings.dflash_snap_context = true;
    try dr.restart(0, 0);
    try dr.restart(1, 0);
    const at: u32 = 2048;
    try f.e.prepareSnapshots(3 * f.e.stateBytes(at));
    var prompt: DraftPrompt = .{ .dr = &dr };
    try glm.dflash.ingestPrompt(&prompt, {}, 0, at - sn.cap, sn.cap, at, 0);
    try f.mark(at, 0, 31);
    const target = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
    defer t.allocator.free(target);
    try f.e.keepAs(1);
    const ram = f.e.snaps.get(1).?;
    try t.expectEqual(sn.cap, ram.draft.count);
    // Simulate disk round-trip: gather target bytes from the draft-tailed RAM lease into a blankAs/seal lease.
    const sh = try glm.disk_image.Shape.of(&f.seq.st, &f.e.c, f.e.capacity, false);
    try t.expectEqual(@as(u64, ram.len), glm.disk_image.leaseBytes(sh, at, 0));
    var segs: std.ArrayList(glm.disk_image.Seg) = .empty;
    defer segs.deinit(t.allocator);
    try glm.disk_image.segments(sh, at, 0, &segs, t.allocator);
    const img_buf = try t.allocator.alloc(u8, @intCast(glm.disk_image.imageBytes(sh, at)));
    defer t.allocator.free(img_buf);
    var words = (glm.disk_image.Words{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 0, .world = 1 }).encode();
    try (glm.disk_image.Copy{ .d = &d, .pool = &f.e.snap_pages.?, .lease = &ram.lease, .segs = segs.items, .words = &words }).run(0, img_buf, true);
    try glm.snapshots.blankAs(&f.e, 2, at);
    try (glm.disk_image.Copy{ .d = &d, .pool = &f.e.snap_pages.?, .lease = &f.e.snaps.getPtr(2).?.lease, .segs = segs.items, .words = &words }).run(0, img_buf, false);
    try glm.snapshots.seal(&f.e, 2, 0, .{ 0, 0 }, 0);
    const disk = f.e.snaps.get(2).?;
    try t.expectEqual(ram.len, disk.len);
    try t.expectEqual(disk.len, disk.packed_len);
    try t.expectEqual(@as(usize, 0), disk.draft.count);
    // Restore the disk-sealed state into slot 1: target bytes match and DFlash2 restarts cold at `at`.
    f.e.slot = 1;
    const dst = dr.slots.items[1].ring.?;
    @memset(rec.bytes(dst.ptr, dst.len), 255);
    try f.mark(3000, 0, 99);
    try f.e.restoreState(2);
    try f.expectPrefix(target, at, 0);
    try DraftPrompt.expectRange(dst, at, 0);
    try t.expectEqual(@as(i64, at), dr.slots.items[1].from);
    try t.expectEqual(@as(usize, 0), dr.slots.items[1].prompt_history.rows(at));
    f.e.dropState(1);
    f.e.dropState(2);
}

test "GLM snapshots: disk blankAs/seal/restore fits a pool sized strictly for targetBytes when draft context is on" {
    const HostCopy = struct {
        fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
            @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], rec.bytes(src, n));
            return 0;
        }
        fn htod(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
            @memcpy(rec.bytes(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
            return 0;
        }
        fn ctxDev(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
            out.* = 0;
            return 0;
        }
        fn devAttr(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            out.* = 0;
            return 0;
        }
    };
    var d = rec.init();
    d.api.cuMemcpyDtoH_v2 = HostCopy.dtoh;
    d.api.cuMemcpyHtoD_v2 = HostCopy.htod;
    d.api.cuCtxGetDevice = HostCopy.ctxDev;
    d.api.cuDeviceGetAttribute = HostCopy.devAttr;
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    var dr: glm.dflash.Drafter = undefined;
    dr.d = &d;
    dr.gpa = t.allocator;
    dr.slots = .empty;
    defer {
        for (dr.slots.items) |*sl| if (sl.ring) |*r| r.free();
        dr.slots.deinit(t.allocator);
    }
    f.e.drafter = &dr;
    f.e.settings.dflash_snap_context = true;
    try dr.restart(0, 0);
    const at: u32 = 2048;
    const target_budget = glm.snapshots.targetBytes(&f.e, at);
    const ram_budget = f.e.stateBytes(at);
    try t.expect(target_budget < ram_budget);
    try f.e.prepareSnapshots(target_budget);
    const sh = try glm.disk_image.Shape.of(&f.seq.st, &f.e.c, f.e.capacity, false);
    var segs: std.ArrayList(glm.disk_image.Seg) = .empty;
    defer segs.deinit(t.allocator);
    try glm.disk_image.segments(sh, at, 0, &segs, t.allocator);
    const img_buf = try t.allocator.alloc(u8, @intCast(glm.disk_image.imageBytes(sh, at)));
    defer t.allocator.free(img_buf);
    @memset(img_buf, 0x3c);
    var words = (glm.disk_image.Words{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 0, .world = 1 }).encode();
    try glm.snapshots.blankAs(&f.e, 1, at);
    const leased = f.e.snaps.get(1).?.lease.ids.items.len * glm.snapshots.pages.page;
    try t.expectEqual(target_budget, leased);
    try (glm.disk_image.Copy{ .d = &d, .pool = &f.e.snap_pages.?, .lease = &f.e.snaps.getPtr(1).?.lease, .segs = segs.items, .words = &words }).run(0, img_buf, false);
    try glm.snapshots.seal(&f.e, 1, 0, .{ 0, 0 }, 0);
    try f.e.restoreState(1);
    try t.expectEqual(@as(i64, at), dr.slots.items[0].from);
    try t.expectEqual(@as(usize, 0), dr.slots.items[0].prompt_history.rows(at));
    f.e.dropState(1);
    try t.expectEqual(target_budget, f.e.snap_pages.?.spare());
}
