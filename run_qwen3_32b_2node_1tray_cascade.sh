#!/bin/bash
# ---------------------------------------------------------------------------
# CASCADE-TUNED variant of the 2-node/1-tray Qwen3-32B example (8 GPUs, 4 CPUs).
# Same topology as run_qwen3_32b_2node_1tray.sh, but sized so the KV cache
# actually OVERFLOWS the NPU and spills NPU -> CPU -> JBOF -> COLDSTORE, and the
# reuse phase RELOADS from the deep tier (so the dashboard tiers visibly fill and
# you see real reload latency / hit rates).
#
# Why these numbers (Qwen3-32B, TP=2 => 128 KB/token per rank; weight 30.51 GiB/GPU):
#   * npu_mem 30.56  -> NPU KV = 30.56 - 30.51 = ~48 MB/GPU = ~1.5 contexts of 256 tok.
#   * --max-num-seqs 1 -> only ONE 32 MB context is in flight, so it always fits
#     (no batch-overflow crash), yet the moment the NEXT context is admitted the
#     retained one must be evicted -> demoted downward. (With more concurrency the
#     batch itself exceeds a KV this small and the run aborts.)
#   * cpu_mem 0.08 (86 MB) and jbof 0.15 (154 MB) each hold >= ONE context. This
#     matters: the exclusive victim-cache DROPS an evicted prefix that does not fit
#     the tier below (KV is recomputable) -- so a tier SMALLER than one prefix never
#     fills. A context costs 2x bytes in CPU/JBOF/COLDSTORE (256 KB/token: both TP
#     ranks) vs 128 KB/token in the NPU, so size the lower tiers accordingly.
#   * coldstore 2 GiB = the sink the bulk lands in.
#
# Workload = make_rack_fill: 16 distinct 256-token contexts (phase 1 fills+spills),
# then 16 reuses oldest-first (phase 2 reloads each from the deep tier it landed in).
#
# VERIFIED (2 nodes x 1 tray, exclusive, per_instance): NPU fills ~66%, spills to
# JBOF (~128 MB) and COLDSTORE (~1 GiB); phase-2 reuse => COLDSTORE reload ~200 ms,
# overall prefix-hit 49%. TTFT spikes to ~196 ms on the reloads.
#
# RUN (inside the container, from the repo root):
#   docker exec servingsim_docker bash -lc \
#     "cd /app/LLMServingSim && bash run_qwen3_32b_2node_1tray_cascade.sh"
# LIVE DASHBOARD (host):  python3 serving/dashboard/serve.py  -> http://localhost:8000
#   watch "KV SPILL MEMORY OVER SIM TIME" fill, and the drill-down "By tray".
# ---------------------------------------------------------------------------

# 1) (re)generate the fill-then-reuse workload (workloads/*.jsonl are gitignored)
python3 workloads/generators/make_rack_fill.py workloads/qwen_cascade_trace.jsonl 16 256 8 4 3000

# 2) run the cascade
python -m serving \
  --cluster-config configs/cluster/qwen3_32b_2node_1tray_cascade.json \
  --dtype float16 --block-size 16 \
  --enable-prefix-caching --enable-prefix-sharing \
  --prefix-storage COLDSTORE --cpu-scope per_instance --tier-policy exclusive \
  --max-num-seqs 1 --dashboard --log-interval 0.2 \
  --dataset workloads/qwen_cascade_trace.jsonl \
  --output outputs/qwen_cascade.csv
