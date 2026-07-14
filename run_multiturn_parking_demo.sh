#!/bin/bash
# ---------------------------------------------------------------------------
# DECODE-focused demo: multi-turn "context parking" (session pause/resume).
#
# A conversation's KV state is PARKED to the pooled deep tier while the user is
# idle between turns, then RELOADED to answer the next turn. With JBOF that
# reload is fast (~80 us); WITHOUT it the state falls to COLDSTORE (~2 ms) — so
# the tier shows up directly in RESUMED-TURN TTFT.
#
# It rides entirely on existing machinery: the agentic-session loader releases
# each turn `tool_duration_ns` after the previous finishes (the idle gap), and a
# turn's input is the accumulated conversation (a growing shared prefix) so the
# next turn RELOADS it (a prefill over the parked context => tier-timed) instead
# of recomputing. --session-metrics splits TTFT into cold first-turn vs warm
# resumed-turn so the JBOF benefit is legible.
#
# 8 GPUs / 4 CPUs (2 nodes x 1 tray, Qwen3-32B, TP=2 superchips). Tiers sized so
# an idle session's context is evicted NPU->CPU->JBOF (CPU holds < the working
# set; each tier >= one context so demotions land rather than drop).
#
# Run inside the container from the repo root:
#   docker exec servingsim_docker bash -lc \
#     "cd /app/LLMServingSim && bash run_multiturn_parking_demo.sh"
# Live dashboard (host): python3 serving/dashboard/serve.py -> http://localhost:8000
#
# VERIFIED on the fast 8B twin (configs/cluster/multiturn_8b_demo.json): reuse
# served 100% from the pooled tier; resumed-turn TTFT 16.0 ms (JBOF) vs 34.7 ms
# (COLDSTORE) = 2.2x faster resume; reload 16.8 ms vs 221 ms = 13x. First-turn
# TTFT identical (37.1 ms) -> the whole delta is the parked-KV reload tier.
# ---------------------------------------------------------------------------

# 1) multi-turn workload: 16 sessions x 3 turns, 512-tok context, +16 user & 16
#    decode per turn, 500 ms idle gap. (More sessions than the small NPU holds,
#    so idle ones park.) See workloads/generators/make_multiturn.py.
#    VERIFIED Qwen A/B (this config): reload 61 ms (JBOF) vs 778 ms (COLDSTORE) = 12.7x;
#    resumed-turn TTFT 763 vs 910 ms; first-turn 471 ms both (delta = reload tier only).
python3 workloads/generators/make_multiturn.py workloads/multiturn_qwen_trace.jsonl 16 3 512 16 16 500 40

COMMON="--dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing \
  --prefix-storage COLDSTORE --cpu-scope per_instance --tier-policy exclusive \
  --max-num-seqs 1 --session-metrics --log-interval 1 \
  --dataset workloads/multiturn_qwen_trace.jsonl"

# 2a) WITH JBOF  -> parked context reloads from JBOF (fast)
python -m serving --cluster-config configs/cluster/multiturn_qwen3_32b.json $COMMON \
  --dashboard --output outputs/multiturn_qwen_jbof.csv

# 2b) WITHOUT JBOF -> parked context falls to COLDSTORE (slow); compare resumed-turn TTFT
python -m serving --cluster-config configs/cluster/multiturn_qwen3_32b_nojbof.json $COMMON \
  --output outputs/multiturn_qwen_nojbof.csv
