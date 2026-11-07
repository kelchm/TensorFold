# Decide whether to run the full replay campaign

Finish and retain the host harness, frozen Python adapter, review fixes and verification receipt.
Booking the full three-pair native campaign plus Python context waits for two cheaper screens
and the user's decision. Host readiness does not itself trigger a hardware campaign.

## Existing production workload

The serving owner uses existing production Python metrics observations, logs and Pi session
metadata, without instrumentation, configuration or service changes. Report aggregates only;
do not retain conversation contents, prompts, tool outputs, session identifiers or per-turn traces
in the report. State the observation window, request population, coverage and metric semantics.
Deduplicate TP-rank copies of a request or counter.

- Concurrency: active-count distribution, time fraction with more than one request active, and
  fraction of arriving requests overlapping another request where timestamps support it. Distinguish
  executing, queued and merely connected requests. Specify each denominator.
- Tool-result sizes: quantiles and totals with units. Bytes and tokens are different quantities;
  label actual token counts and any bounded estimate separately.
- Reply lengths and context lengths: request counts, totals and available quantiles or histogram
  bins. Distinguish uncached input from total context and new tool input from replayed cache tail.

Cumulative counters alone do not recover past concurrency or quantiles. A sampled active gauge
can miss short overlaps. Describe limitations and mark unavailable statistics instead of assuming
serial traffic or treating a mean as a percentile. The user's Pi workload determines relevance;
the synthetic benchmark mix is not evidence of production frequency.

## Cheap served sharding screen

The serving owner coordinates one window using the existing rerun harness's one-stream and
four-stream decode sets. Use the same deployed native q8 image and recipe, changing only
`TF_GLM_KV_SHARD=0` versus `1`. Retain ordinary drafting and the existing prompt/output controls.
Record all attempts, exact identities/settings, owner activation and existing rate, TTFT,
round/yield and acceptance measurements.

Record both arms' actual page arguments, admitted logical capacity and physical allocation.
Automatic pool sizing can differ; report that difference explicitly. This served screen retains
generated-history, draft-policy and potential pool-capacity confounds. It is not the equal-capacity
fixed-work causal experiment. Do not silently retune the serving recipe to erase those differences.

Allow roughly half an hour plus restarts as a planning estimate. Follow the serving owner's
persistent SSH-master, health, retained-store and restore procedure. A missing master, failed start,
correctness failure or health/restore problem stops the screen. Restore the agreed Python service
and verify both ranks. This document grants no additional machine, driver or publication authority;
the source implementation thread remains host-only.

## Return a recommendation before the full campaign

Report both screens together with coverage, limitations and the practical question replay would
resolve. If sharding off barely moves the four-stream result and real concurrency is rare, the
full campaign likely has little workload payoff. In that case, investigating native's 4,096-token
cache-resume boundary versus Python's roughly 64-token behavior is likely the higher-value next
engine question. If the sharding effect or concurrent workload is material, explain what matched
fixed work would disambiguate. Do not invent a universal cutoff or infer a kernel cause from the
screen. The user decides whether to book the full campaign.

The completed rerun motivates this order: equal-precision one-stream decode was near parity,
while the four-stream and cold long-prefill gaps were smaller than the Pi tool-result TTFT gap.
Replay deliberately excludes cache-resume granularity. Production Python's q4 versus native's q8
is an execution-policy difference and does not establish an engine defect.
