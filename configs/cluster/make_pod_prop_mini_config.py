"""SuperPod config for the PROPORTIONAL natural-spillover study on the mini model.

Uses meta-llama/Llama-3.1-8B-mini4L (a COPY of Llama-3.1-8B with 4 layers -> ~3.58
GiB weight instead of ~15 GiB), so an 8 GiB NPU keeps a healthy ~4.4 GiB KV
headroom (55%) -- versus ~1 GiB (6%) for the full 8B, which crashed under reload
churn. Tier capacities preserve the real Vera Rubin ratio NPU_KV:CPU:JBOF ~=
1 : 2.75 : 58; only bandwidth/latency are real and unscaled. Chain:
NPU -> CPU -> JBOF -> COLDSTORE (FLASH skipped).

Scoping: NPU per-GPU; CPU per-RACK; JBOF + COLDSTORE pod-wide shared (per-rack
BlueField-4 channels).

Usage:
  python make_pod_prop_mini_config.py <num_racks> <jbof|nojbof> <out.json> [gpr=72] [npu_gib cpu_pg jbof_pg cold_pg]

Per-GPU caps default to NPU 8 / CPU 12 / JBOF 250 / COLD 2500 GiB (the ratio ~1:2.75:58
around ~4.4 GiB NPU KV headroom). They can be overridden to a SMALLER absolute scale
(same faithful ratio) so a multi-node run reaches JBOF with a tractable workload.
"""
import json
import sys

racks = int(sys.argv[1]) if len(sys.argv) > 1 else 16
jbof   = (sys.argv[2] if len(sys.argv) > 2 else "jbof") == "jbof"
out   = sys.argv[3] if len(sys.argv) > 3 else f"configs/cluster/pod_prop_mini_{racks}rack_{'jbof' if jbof else 'nojbof'}.json"
gpr   = int(sys.argv[4]) if len(sys.argv) > 4 else 72

# per-GPU caps (GiB). Defaults preserve the real Vera Rubin ratio NPU_KV:CPU:JBOF ~ 1:2.75:58.
NPU_GIB      = float(sys.argv[5]) if len(sys.argv) > 5 else 8
CPU_PER_GPU  = float(sys.argv[6]) if len(sys.argv) > 6 else 12
JBOF_PER_GPU = float(sys.argv[7]) if len(sys.argv) > 7 else 250
COLD_PER_GPU = float(sys.argv[8]) if len(sys.argv) > 8 else 2500

def r(x):
    return round(x, 2)

gpu = {
    "model_name": "meta-llama/Llama-3.1-8B-mini4L",
    "hardware": "RTXPRO6000",
    "npu_mem": {"mem_size": NPU_GIB, "mem_bw": 22000, "mem_latency": 10},   # Rubin HBM4
    "pd_type": None,
    "tp_size": 1,
}

def make_node():
    node = {
        "num_instances": gpr,
        "cpu_mem":       {"mem_size": r(CPU_PER_GPU * gpr),  "mem_bw": 1000, "mem_latency": 120},                          # Vera LPDDR5X
        "coldstore_mem": {"mem_size": r(COLD_PER_GPU * gpr), "mem_bw": 4,    "mem_latency": 2000000, "link_bw": 16, "link_latency": 20000},
    }
    if jbof:
        node["jbof_mem"] = {"mem_size": r(JBOF_PER_GPU * gpr), "mem_bw": 64, "mem_latency": 80000, "link_bw": 100, "link_latency": 2000}
    # no flash_mem -> FLASH skipped (chain NPU->CPU->JBOF->COLDSTORE)
    node["instances"] = [dict(gpu) for _ in range(gpr)]
    return node

cfg = {"num_nodes": racks, "link_bw": 100, "link_latency": 2000, "nodes": [make_node() for _ in range(racks)]}
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {racks}x{gpr}={racks*gpr} GPUs, mini4L model, JBOF={'on' if jbof else 'off'}, FLASH skipped; "
      f"per-GPU NPU {NPU_GIB}/CPU {CPU_PER_GPU}/JBOF {JBOF_PER_GPU if jbof else 0}/COLD {COLD_PER_GPU} GiB")
