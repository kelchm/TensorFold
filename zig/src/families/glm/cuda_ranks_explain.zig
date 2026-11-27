//! Human-readable explanations for GLM TP rank fingerprint mismatches.

pub fn explain(f: anytype) []const u8 {
    return switch (f) {
        .layout => "a different build of this check",
        .rank => "a rank number given twice or out of order",
        .world => "the ranks were started with different world sizes",
        .keep_seqs => "TF_GLM_KEEP_SEQS",
        .kv8 => "--kv-cache fp8",
        .pages => "TF_KV_PAGES (unset or auto pages from two ranks)",
        .prompt_fixed_k => "TF_GLM_PROMPT_K",
        .tiled => "TF_GLM_MM, TF_GLM_DECODE_MLA_WIDE or TF_GLM_DECODE_TILE_RANGES",
        .split => "TF_GLM_PREFILL_REDUCE",
        .kda_wide => "TF_GLM_KDA_DECODE_WIDE or TF_GLM_SHARED_KDA_WIDE",
        .kda_pad => "TF_GLM_KDA_PAD",
        .kv_live_half => "TF_GLM_KV_LIVE_HALF",
        .pieces => "TF_GLM_OVERLAP_PIECES",
        .pair => "TF_GLM_PROMPT_PAIR or TF_GLM_PAIR_DFLASH",
        .pair_experts => "TF_GLM_PAIR_EXPERTS / TF_GLM_EXL3_DECODE / TF_GLM_EXL3_N_MAJOR",
        .layers => "TF_GLM_LAYERS",
        .mtp => "--no-drafts, --drafter, TF_GLM_DFLASH_ONLY, TF_GLM_DFLASH_Q4_DOWN or TF_GLM_DFLASH_SNAP_CONTEXT differs across ranks",
        .vision => "--vision (or its image tokens) given to some ranks and not others",
        .video_tokens => "--vision-video-tokens",
        .max_images => "the image count limit",
        .max_videos => "the video count limit",
        .max_frames => "the frame limit",
        .roce_multi => "TF_GLM_ROCE_MULTI",
        .roce_fuse => "TF_GLM_ROCE_FUSE",
        .roce_wide => "TF_ROCE=1 at three ranks or more (the RoCE gather and the drafter split)",
        .draft_split => "TF_GLM_DRAFT_SPLIT=1 at three ranks or more (the drafter split)",
        .kv_host => "TF_GLM_KV_HOST or TF_GLM_KV_HOST_SHARED",
        .kv_host_gib => "TF_GLM_KV_HOST_GIB",
        .solo => "TF_GLM_CHUNK_SOLO",
        .l2pf => "TF_GLM_L2PF, _SHARED_FIRST, _MB, _CTAS or _PACE_GBPS",
        .bits => "TF_GLM_DENSE_BITS, TF_GLM_HEAD_BITS or TF_GLM_PACKED_PROMPT_S2",
        .numerics => "TF_GLM_ABSORB_MMA or TF_GLM_EXL3_BF16_SLOTS",
        .capacity => "the sequence capacity (the context)",
        .drafter_bits => "--drafter-bits",
        .heat_high => "TF_HEAT_HIGH",
        .heat_low => "TF_HEAT_LOW",
        .config => "config.json",
        .program => "the executable",
        .checkpoint => "checkpoint metadata or shard bytes",
        .drafter_checkpoint => "--drafter checkpoint metadata or shard bytes",
    };
}

/// Logs rank 0 startup warnings when opt-in prompt arithmetic switches are active.
pub fn logNumerics(n: anytype, rank: usize) void {
    if (rank != 0) return;
    if (n.absorb_mma) @import("std").log.warn("TF_GLM_ABSORB_MMA: prompt absorb/expand uses tensor cores with a different FP32 accumulation order", .{});
    if (n.exl3_bf16_slots) @import("std").log.warn("TF_GLM_EXL3_BF16_SLOTS: EXL3 prompt chunks round their MoE slot rows to bf16 before the combine", .{});
}
