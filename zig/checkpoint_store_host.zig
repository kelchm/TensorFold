const std = @import("std");
const store = @import("checkpoint_store.zig");
test {
    _ = @import("checkpoint_store/store_test.zig");
    _ = @import("checkpoint_store/protocol_test.zig");
    _ = @import("checkpoint_store/integrity_test.zig");
}
test "block store: reference no-op" {
    try std.testing.expect(!store.serving_enabled);
    try std.testing.expectEqual(@as(?store.format.Receipt, null), try store.capture.save(.reference, null, null));
}
