//! Optional FIFO admission before body allocation; a resident ticket lasts through every engine borrow.
const std = @import("std");

pub const Policy = struct {
    residents: u32 = 0,
    waiting: u32 = 32,
    wait_ms: u32 = 300_000,
    floor: u64 = 0,
    reserve: u64 = 0,

    pub fn render(p: Policy, w: *std.Io.Writer) !void {
        inline for (comptime std.meta.fieldNames(Policy)) |field|
            try w.print("tensorfold_prepare_policy_{s} {d}\n", .{ field, @field(p, field) });
    }

    pub fn fromEnv(env: ?*const std.process.Environ.Map) !Policy {
        const p: Policy = .{
            .residents = @intCast(try setting(env, "TF_PREPARE_RESIDENTS", 0, 128)),
            .waiting = @intCast(try setting(env, "TF_PREPARE_WAITING", 32, 128)),
            .wait_ms = @intCast(try setting(env, "TF_PREPARE_WAIT_MS", 300_000, 3_600_000)),
            .floor = (try setting(env, "TF_PREPARE_FLOOR_MIB", 0, 1 << 30)) << 20,
            .reserve = (try setting(env, "TF_PREPARE_RESERVE_MIB", 0, 1 << 30)) << 20,
        };
        if (p.wait_ms == 0 or (p.floor > 0 and (p.reserve == 0 or p.residents == 0)))
            return error.BadPreparePolicy;
        if (p.reserve > 0 and p.floor == 0) return error.BadPreparePolicy;
        return p;
    }
};

fn setting(env: ?*const std.process.Environ.Map, name: []const u8, default: u64, max: u64) !u64 {
    const e = env orelse return default;
    const raw = e.get(name) orelse return default;
    const n = std.fmt.parseInt(u64, std.mem.trim(u8, raw, " \t\r\n"), 10) catch return error.BadPreparePolicy;
    if (n > max) return error.BadPreparePolicy;
    return n;
}

pub const Snapshot = struct {
    residents: u32 = 0,
    waiting: u32 = 0,
    peak_residents: u32 = 0,
    peak_waiting: u32 = 0,
    admitted: u64 = 0,
    waited: u64 = 0,
    full: u64 = 0,
    cancelled: u64 = 0,
    timed_out: u64 = 0,
    drained: u64 = 0,
    telemetry_missing: u64 = 0,
    reserved_bytes: u64 = 0,
};
const Waiter = struct { next: ?*Waiter = null };
pub const Error = error{ Cancelled, Draining, PrepareQueueFull, PrepareTimeout };

pub const Gate = struct {
    mutex: std.Io.Mutex = .init,
    head: ?*Waiter = null,
    tail: ?*Waiter = null,
    state: Snapshot = .{},
    available: *const fn (std.Io) ?u64 = memAvailable,

    pub fn enter(g: *Gate, io: std.Io, policy: Policy, draining: *const std.atomic.Value(bool), gone: anytype) Error!bool {
        if (policy.residents == 0) return false;
        const until = std.Io.Clock.awake.now(io).addDuration(.fromMilliseconds(policy.wait_ms));
        var waiter: Waiter = .{};
        var queued = false;
        var missing = false;
        defer if (queued) {
            g.mutex.lockUncancelable(io);
            g.remove(&waiter);
            g.mutex.unlock(io);
        };
        while (true) {
            const cancelled = gone.check();
            const drain = draining.load(.seq_cst);
            g.mutex.lockUncancelable(io);
            if (cancelled or drain) {
                if (cancelled) g.state.cancelled += 1 else g.state.drained += 1;
                g.mutex.unlock(io);
                return if (cancelled) error.Cancelled else error.Draining;
            }
            const first = if (queued) g.head == &waiter else g.head == null;
            var fits = g.state.residents < policy.residents;
            if (fits and first and policy.floor > 0) {
                if (g.available(io)) |available| {
                    fits = available >= policy.floor +| g.state.reserved_bytes +| policy.reserve;
                } else {
                    fits = false;
                    if (!missing) g.state.telemetry_missing += 1;
                    missing = true;
                }
            }
            if (first and fits) {
                if (queued) g.remove(&waiter);
                queued = false;
                g.state.residents += 1;
                g.state.peak_residents = @max(g.state.peak_residents, g.state.residents);
                g.state.reserved_bytes += policy.reserve;
                g.state.admitted += 1;
                g.mutex.unlock(io);
                return true;
            }
            if (!queued) {
                if (g.state.waiting >= policy.waiting) {
                    g.state.full += 1;
                    g.mutex.unlock(io);
                    return error.PrepareQueueFull;
                }
                if (g.tail) |tail| tail.next = &waiter else g.head = &waiter;
                g.tail = &waiter;
                queued = true;
                g.state.waiting += 1;
                g.state.waited += 1;
                g.state.peak_waiting = @max(g.state.peak_waiting, g.state.waiting);
            }
            if (std.Io.Clock.awake.now(io).nanoseconds >= until.nanoseconds) {
                g.state.timed_out += 1;
                g.mutex.unlock(io);
                return error.PrepareTimeout;
            }
            g.mutex.unlock(io);
            std.Io.sleep(io, .fromMilliseconds(10), .awake) catch return error.Cancelled;
        }
    }

    fn remove(g: *Gate, waiter: *Waiter) void {
        var previous: ?*Waiter = null;
        var node = g.head;
        while (node) |n| : (node = n.next) {
            if (n == waiter) {
                if (previous) |p| p.next = n.next else g.head = n.next;
                if (g.tail == n) g.tail = previous;
                g.state.waiting -= 1;
                return;
            }
            previous = n;
        }
        unreachable;
    }

    pub fn leave(g: *Gate, io: std.Io, reserve: u64) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        std.debug.assert(g.state.residents > 0 and g.state.reserved_bytes >= reserve);
        g.state.residents -= 1;
        g.state.reserved_bytes -= reserve;
    }

    pub fn snapshot(g: *Gate, io: std.Io) Snapshot {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        return g.state;
    }

    pub fn render(g: *Gate, io: std.Io, w: *std.Io.Writer) !void {
        const s = g.snapshot(io);
        inline for (comptime std.meta.fieldNames(Snapshot)) |field|
            try w.print("tensorfold_prepare_{s} {d}\n", .{ field, @field(s, field) });
    }
};

/// Linux host headroom, not a GPU budget or a process RSS limit. Missing telemetry never opens a floor gate.
pub fn memAvailable(io: std.Io) ?u64 {
    var buf: [4096]u8 = undefined;
    return parseAvailable(std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null);
}

pub fn parseAvailable(text: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var it = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const n = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, it.next() orelse return null, "kB")) return null;
        return std.math.mul(u64, n, 1024) catch null;
    }
    return null;
}
