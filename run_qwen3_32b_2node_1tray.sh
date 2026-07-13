#!/bin/bash
# ---------------------------------------------------------------------------
# Example: 2 nodes x 1 tray, running Qwen/Qwen3-32B.
#
#   tray      = 2 superchips = 4 GPUs + 2 CPUs
#   superchip = 1 Vera CPU (1.5 TB LPDDR5X) + 2 Rubin GPUs (288 GB HBM4)
#             = ONE TP=2 instance   (profiler ships Qwen3-32B fp16 tp1 AND tp2)
#
#   => 2 nodes x (1 tray) = 2 nodes x 2 superchip-instances
#      = 4 instances, 8 GPUs, 4 CPUs total.
#
# Config: configs/cluster/qwen3_32b_2node_1tray.json  (new de-duplicated form —
#   CPU declared once per superchip; the node aggregate is DERIVED, no node-level
#   cpu_mem). JBOF/COLDSTORE are declared once at the top level (pod-wide).
#
# Capacities are faithful Vera Rubin (288 GiB HBM / 1.5 TiB CPU-per-superchip /
# 128 TiB pod JBOF / 1 PiB COLDSTORE); bw/latency are the real figures. With the
# tiny example_trace nothing spills past the NPU under --tier-policy exclusive
# (correct victim-cache behavior). To watch the NPU->CPU->JBOF->COLDSTORE cascade,
# shrink the caps (see vera_rubin_tray_2n1g_test_8B_tinynpu.json) or use a larger
# workload.
#
# RUN (inside the container, from the repo root):
#   docker exec servingsim_docker bash -lc \
#     "cd /app/LLMServingSim && bash run_qwen3_32b_2node_1tray.sh"
#
# LIVE DASHBOARD (optional, on the HOST in a second terminal):
#   python3 serving/dashboard/serve.py          # http://localhost:8000
#   # open it, then use the drill-down "By tray" toggle -> 2 trays x (4 GPUs + 2 CPUs)
# ---------------------------------------------------------------------------

python -m serving \
  --cluster-config configs/cluster/qwen3_32b_2node_1tray.json \
  --dtype float16 --block-size 16 \
  --enable-prefix-caching --enable-prefix-sharing \
  --prefix-storage COLDSTORE --cpu-scope per_instance --tier-policy exclusive \
  --dashboard --log-interval 0.5 \
  --dataset workloads/example_trace.jsonl \
  --output outputs/qwen3_32b_2node_1tray.csv
