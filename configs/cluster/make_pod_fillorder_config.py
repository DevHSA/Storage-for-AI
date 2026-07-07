"""Generate a Vera Rubin SuperPod config: <num_racks> nodes x <gpus_per_rack> GPUs.

Faithful tier scoping:
  * NPU (HBM4)        -> per-GPU
  * CPU (Vera LPDDR)  -> per-RACK   (each node's own pool)
  * FLASH (local SSD) -> per-RACK   (each node's own pool)
  * JBOF (JBOF)        -> POD-WIDE shared pool (sum of per-rack contributions)
  * COLDSTORE (net)   -> POD-WIDE shared pool
The per-node jbof_mem / coldstore_mem blocks are each rack's CONTRIBUTION to the
one shared pod-wide pool; serving/__main__.py sums them into a single RadixCache
common to all racks (a prefix cached by any rack is reusable by every rack).
ASTRA-Sim reaches that shared pool through one BlueField-4 channel per rack
(num-devices = num_racks), so the racks' reloads run on parallel channels.

Capacities are scaled DOWN (like the single-rack fill-order study) so a feasible
workload saturates NPU -> CPU -> FLASH in order before spilling into JBOF.
Latencies stay Vera-Rubin-faithful. mem_size is in GiB.

Usage:
  python make_pod_fillorder_config.py <num_racks> <jbof|nojbof> <out.json> [gpus_per_rack=72]
"""
import json
import sys

racks = int(sys.argv[1]) if len(sys.argv) > 1 else 16
jbof   = (sys.argv[2] if len(sys.argv) > 2 else "jbof") == "jbof"
out   = sys.argv[3] if len(sys.argv) > 3 else f"configs/cluster/pod_fillorder_{racks}rack_{'jbof' if jbof else 'nojbof'}.json"
gpr   = int(sys.argv[4]) if len(sys.argv) > 4 else 72   # GPUs per rack (NVL72)

gpu = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "hardware": "RTXPRO6000",
    "npu_mem": {"mem_size": 16, "mem_bw": 22000, "mem_latency": 10},   # Rubin HBM4 speed, capped size
    "pd_type": None,
    "tp_size": 1,
}

def make_node():
    node = {
        "num_instances": gpr,
        "cpu_mem":       {"mem_size": 2,    "mem_bw": 1000, "mem_latency": 120},                          # per-rack Vera LPDDR5X ~120 ns
        "flash_mem":     {"mem_size": 6,    "mem_bw": 14,   "mem_latency": 10000},                        # per-rack local NVMe Gen5 ~10 us
        "coldstore_mem": {"mem_size": 2000, "mem_bw": 4,    "mem_latency": 2000000, "link_bw": 16, "link_latency": 20000},  # rack's share of pod-wide network storage ~2 ms
    }
    if jbof:
        # Rack's contribution to the pod-wide JBOF pool (~80 us + 3 us RDMA over a
        # 100 GB/s Spectrum-X link per rack -> 16 parallel channels aggregate).
        node["jbof_mem"] = {"mem_size": 120, "mem_bw": 64, "mem_latency": 80000, "link_bw": 100, "link_latency": 3000}
    node["instances"] = [dict(gpu) for _ in range(gpr)]
    return node

cfg = {"num_nodes": racks, "link_bw": 100, "link_latency": 3000, "nodes": [make_node() for _ in range(racks)]}

with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {racks} racks x {gpr} GPUs = {racks*gpr} GPUs, JBOF={'on' if jbof else 'off'}; "
      f"per-rack CPU2/FLASH6 GiB; pod-wide JBOF={120*racks if jbof else 0} GiB, COLDSTORE={2000*racks} GiB")
