"""Generate a JBOF (context-memory) reuse demo for the Vera Rubin configs.

Models a small, frequently-reused context (e.g. a shared system prompt / cached
snippet) that lives in the pooled KV tier. Each request = the shared prefix A
plus a short unique tail. A is kept shorter than the NPU block size so it is NOT
cached in GPU HBM (NPU) or the CPU pool page — it only lives in the deep pooled
tiers (JBOF/JBOF, else COLDSTORE). So every reuse reloads A from the pooled tier:

  WITH JBOF  -> A served from JBOF (fast, ~us)
  WITHOUT   -> A served from COLDSTORE / network storage (slow, ~ms)

This isolates the JBOF-vs-traditional-storage reload cost that NVIDIA advertises,
at a scale that is actually runnable (realistic HBM/LPDDR/JBOF capacities can't be
filled by a feasible workload, so we exploit page granularity instead).

Arrival modes:
  - "stagger" (default): request i arrives at i*stagger_ms (system idles between
    requests -> shows per-access reload latency / TTFT, not throughput).
  - "burst": request 0 (the primer that caches A) at t=0; ALL reuses arrive
    together at t=stagger_ms. The queue fills, so the pooled-tier reload sits on
    the critical path -> the JBOF bandwidth advantage shows up in total time /
    throughput as well.

Usage:
  python make_jbof_demo.py <out.jsonl> [n_reuse] [a_len] [tail_len] [out_len] [stagger_ms] [mode]
"""
import json
import sys

out       = sys.argv[1] if len(sys.argv) > 1 else "workloads/jbof_demo_trace.jsonl"
n_reuse   = int(sys.argv[2]) if len(sys.argv) > 2 else 40
a_len     = int(sys.argv[3]) if len(sys.argv) > 3 else 8      # < block_size (16) -> skips NPU
tail_len  = int(sys.argv[4]) if len(sys.argv) > 4 else 4
out_len   = int(sys.argv[5]) if len(sys.argv) > 5 else 4
stagger_ms = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0
mode      = sys.argv[7] if len(sys.argv) > 7 else "stagger"

A = list(range(1, a_len + 1))   # the shared reused context (lives only in the pooled tier)

with open(out, "w") as f:
    for i in range(n_reuse):
        base = 1_000_000 + i * 1000
        tail = [base + j for j in range(tail_len)]
        input_ids = A + tail
        if mode == "burst":
            # primer at 0, then all reuses arrive together (saturating burst)
            arrival = 0 if i == 0 else int(stagger_ms * 1_000_000)
        else:
            arrival = int(i * stagger_ms * 1_000_000)
        rec = {
            "input_toks": len(input_ids),
            "output_toks": out_len,
            "arrival_time_ns": arrival,
            "input_tok_ids": input_ids,
            "output_tok_ids": [base + 500_000 + j for j in range(out_len)],
        }
        f.write(json.dumps(rec) + "\n")

print(f"wrote {n_reuse} requests to {out} "
      f"(mode={mode}; shared context A={a_len} toks + unique tail={tail_len}; "
      f"first computes A, rest reload A from the pooled tier)")
