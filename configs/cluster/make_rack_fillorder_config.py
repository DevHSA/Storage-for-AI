"""Generate a Vera Rubin NVL72-rack config whose tiers are sized to FILL IN ORDER.

Unlike make_vera_rubin_config.py (which uses realistic multi-TB tier capacities
that a feasible workload can never fill), this uses an ASCENDING, deliberately
*small* capacity ladder so a tractable workload overflows the tiers in order:

    NPU (per-GPU HBM) -> CPU -> FLASH (local SSD) -> JBOF (JBOF) -> COLDSTORE

Only the NPU RadixCache is per-GPU; CPU/FLASH/JBOF/COLDSTORE are ONE shared pool
per node for all N GPUs (this mirrors JBOF being pooled behind BlueField-4). So a
growing unique working set first fills every GPU's NPU, then spills into the
small shared CPU pool, then FLASH, and lands in JBOF (with JBOF) or falls through
to COLDSTORE (without). Tier LATENCIES/BANDWIDTHS stay Vera-Rubin-faithful; only
the CAPACITIES are scaled down so the hierarchy is actually exercised.

mem_size is in GiB. NPU must exceed the ~14.96 GiB fp16 weight of Llama-3.1-8B.

Usage:
  python make_rack_fillorder_config.py <num_gpus> <jbof|nojbof> <out.json>
"""
import json
import sys

n   = int(sys.argv[1]) if len(sys.argv) > 1 else 72
jbof = (sys.argv[2] if len(sys.argv) > 2 else "jbof") == "jbof"
out = sys.argv[3] if len(sys.argv) > 3 else f"configs/cluster/rack_fillorder_{n}gpu_{'jbof' if jbof else 'nojbof'}.json"

# Per-GPU HBM4. 16 GiB leaves ~1.04 GiB for KV over the 14.96 GiB weight
# (~8,500 KV tokens/GPU) so the NPU saturates quickly.
gpu = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "hardware": "RTXPRO6000",
    "npu_mem": {"mem_size": 16, "mem_bw": 22000, "mem_latency": 10},   # Rubin HBM4 speed, capped size
    "pd_type": None,
    "tp_size": 1,
}

# Shared-pool capacities scale with the GPU count so the fill-order holds at any
# scale: the pool must grow with the working set. Per-GPU budget = the n=72 rack
# values / 72 (so n=72 reproduces the single-rack study exactly). For n=1152 this
# is the pod-wide pool (one JBOF cache shared by all 1,152 GPUs).
sc = n / 72.0
node = {
    "num_instances": n,
    # Shared pools (small per-GPU caps to force spillover). Latencies faithful:
    "cpu_mem":       {"mem_size": round(2 * sc, 2),    "mem_bw": 1000, "mem_latency": 120},                          # Vera LPDDR5X ~120 ns
    "flash_mem":     {"mem_size": round(6 * sc, 2),    "mem_bw": 14,   "mem_latency": 10000},                        # local NVMe Gen5 ~10 us
    "coldstore_mem": {"mem_size": round(2000 * sc, 2), "mem_bw": 4,    "mem_latency": 2000000, "link_bw": 16, "link_latency": 20000},  # network storage ~2 ms
}
if jbof:
    # JBOF / JBOFP: Ethernet-attached NVMe flash behind BlueField-4 (~80 us + 3 us RDMA
    # over a 100 GB/s Spectrum-X link). Sized to hold the whole working set so the
    # reused prefix survives here (served from JBOF) instead of falling to COLDSTORE.
    node["jbof_mem"] = {"mem_size": round(120 * sc, 2), "mem_bw": 64, "mem_latency": 80000, "link_bw": 100, "link_latency": 3000}

node["instances"] = [dict(gpu) for _ in range(n)]
cfg = {"num_nodes": 1, "link_bw": 100, "link_latency": 3000, "nodes": [node]}

with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {n} GPU rack, JBOF={'on' if jbof else 'off'}, "
      f"tiers NPU16/CPU2/FLASH6/{'JBOF120/' if jbof else ''}COLDSTORE2000 GiB")
