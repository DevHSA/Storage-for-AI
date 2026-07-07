"""Vera Rubin SuperPod config with PROPORTIONALLY scaled-down tier capacities.

Chain: NPU -> CPU -> JBOF -> COLDSTORE   (node-local FLASH deliberately SKIPPED;
NVIDIA de-emphasizes G3 local SSDs in favor of the pooled JBOF/JBOF G3.5 tier).

Faithfulness approach: keep every tier's BANDWIDTH and LATENCY at the real Vera
Rubin value (physical -- a smaller tier is not a faster tier), and scale ONLY the
CAPACITIES by one common factor so the real inter-tier ratios are preserved and a
tractable workload overflows NPU -> CPU -> JBOF *naturally* (no artificially tiny
tier). Real per-GPU ratios (KV-usable): NPU 273 GB : CPU 750 GB : JBOF 16 TB
= 1 : 2.75 : 58.6. Model weights (~15 GB) don't scale, so the smallest sensible
NPU is 16 GiB (=> ~1.04 GiB KV headroom); the others are scaled to match.

Real references (see run.sh / chat): NPU 288 GB HBM4 @ 22 TB/s (NVIDIA);
CPU Vera LPDDR5X 1.5 TB/socket, ~750 GB/GPU, 1.2 TB/s, ~120 ns; JBOF 16 TB/GPU,
pooled NVMe ~64 GB/s @ 80 us behind a 100 GB/s / ~2 us BlueField-4 Spectrum-X
link; COLDSTORE network storage ~4 GB/s @ ~2 ms (estimate).

Scoping (matches the code): NPU per-GPU; CPU per-RACK pool; JBOF + COLDSTORE are
ONE pod-wide shared pool (capacity = sum of per-rack contributions), reached via
one BlueField-4 channel per rack.

Usage:
  python make_pod_proportional_config.py <num_racks> <jbof|nojbof> <out.json> [gpus_per_rack=72]
"""
import json
import sys

racks = int(sys.argv[1]) if len(sys.argv) > 1 else 16
jbof   = (sys.argv[2] if len(sys.argv) > 2 else "jbof") == "jbof"
out   = sys.argv[3] if len(sys.argv) > 3 else f"configs/cluster/pod_prop_{racks}rack_{'jbof' if jbof else 'nojbof'}.json"
gpr   = int(sys.argv[4]) if len(sys.argv) > 4 else 72

# --- per-GPU scaled capacities (GiB), preserving real Vera Rubin ratios --------
NPU_GIB         = 16      # 288 GB real -> 16 GiB (leaves ~1.04 GiB KV over the ~15 GB weight)
CPU_PER_GPU     = 2.9     # 750 GB real x (1.04/273)
JBOF_PER_GPU    = 61      # 16 TB real x (1.04/273)
COLD_PER_GPU    = 610     # ~10x JBOF (durable backstop; effectively unbounded)

def r(x):
    return round(x, 2)

gpu = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "hardware": "RTXPRO6000",
    "npu_mem": {"mem_size": NPU_GIB, "mem_bw": 22000, "mem_latency": 10},   # Rubin HBM4 (22 TB/s); latency ~vestigial (NPU hits are free)
    "pd_type": None,
    "tp_size": 1,
}

def make_node():
    node = {
        "num_instances": gpr,
        # per-RACK Vera LPDDR5X pool (1.2 TB/s, ~120 ns):
        "cpu_mem":       {"mem_size": r(CPU_PER_GPU * gpr),  "mem_bw": 1000, "mem_latency": 120},
        # rack's contribution to the POD-WIDE COLDSTORE (network storage ~2 ms):
        "coldstore_mem": {"mem_size": r(COLD_PER_GPU * gpr), "mem_bw": 4,    "mem_latency": 2000000, "link_bw": 16, "link_latency": 20000},
    }
    if jbof:
        # rack's contribution to the POD-WIDE JBOF/JBOF pool: pooled NVMe ~64 GB/s
        # @ 80 us behind a 100 GB/s / ~2 us BlueField-4 Spectrum-X RDMA link.
        node["jbof_mem"] = {"mem_size": r(JBOF_PER_GPU * gpr), "mem_bw": 64, "mem_latency": 80000, "link_bw": 100, "link_latency": 2000}
    # NOTE: no "flash_mem" block -> FLASH tier skipped (chain = NPU->CPU->JBOF->COLDSTORE)
    node["instances"] = [dict(gpu) for _ in range(gpr)]
    return node

cfg = {"num_nodes": racks, "link_bw": 100, "link_latency": 2000, "nodes": [make_node() for _ in range(racks)]}

with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {racks} racks x {gpr} GPUs = {racks*gpr} GPUs, JBOF={'on' if jbof else 'off'}, FLASH skipped; "
      f"per-GPU NPU {NPU_GIB}/CPU {CPU_PER_GPU}/JBOF {JBOF_PER_GPU if jbof else 0}/COLD {COLD_PER_GPU} GiB; "
      f"per-rack CPU pool {r(CPU_PER_GPU*gpr)} GiB; pod JBOF {r(JBOF_PER_GPU*gpr*racks) if jbof else 0} GiB")
