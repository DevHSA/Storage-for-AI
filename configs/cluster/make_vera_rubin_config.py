"""Generate a Vera Rubin-style cluster config with N GPU instances on one node.

All GPUs on the node share the pooled tiers (Vera LPDDR, JBOF/JBOF, COLDSTORE) —
this mirrors the real rack where JBOF is a pooled KV tier behind BlueField-4.
Per-tier access latencies are realistic (HBM ns, LPDDR ~100ns, JBOF flash ~80us,
network storage ~ms). Node-level pooled capacities scale with the GPU count.

Usage:
  python make_vera_rubin_config.py <num_gpus> <jbof|nojbof> <out.json>
"""
import json
import sys

n   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
jbof = (sys.argv[2] if len(sys.argv) > 2 else "jbof") == "jbof"
out = sys.argv[3] if len(sys.argv) > 3 else f"configs/cluster/vera_rubin_{n}gpu_{'jbof' if jbof else 'nojbof'}.json"

gpu = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "hardware": "RTXPRO6000",
    "npu_mem": {"mem_size": 288, "mem_bw": 22000, "mem_latency": 10},   # Rubin HBM4
    "pd_type": None,
    "tp_size": 1,
}

node = {
    "num_instances": n,
    # Vera LPDDR5X: 1.5 TB per Vera CPU, 2 GPUs per Vera -> ~768 GB/GPU pooled.
    "cpu_mem": {"mem_size": 768 * n, "mem_bw": 1000, "mem_latency": 120},
}
if jbof:
    # JBOF / JBOFP: up to 16 TB/GPU of Ethernet-attached NVMe flash, pooled.
    node["jbof_mem"] = {"mem_size": 16000 * n, "mem_bw": 64, "mem_latency": 80000,
                        "link_bw": 100, "link_latency": 3000}
# G4 network storage (large, shared).
node["coldstore_mem"] = {"mem_size": 131072, "mem_bw": 4, "mem_latency": 2000000,
                         "link_bw": 16, "link_latency": 20000}
node["instances"] = [dict(gpu) for _ in range(n)]

cfg = {"num_nodes": 1, "link_bw": 100, "link_latency": 3000, "nodes": [node]}

with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {n} Rubin GPU instances, JBOF={'on' if jbof else 'off'}")
