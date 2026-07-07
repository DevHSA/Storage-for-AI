"""Generate a phased trace that forces cross-tier spillover of a prefix.

Pattern:
  1. Request for prefix A          -> A is written through to every tier.
  2. `n_filler` distinct prefixes  -> fill the small tiers, evicting A out of
                                      NPU/CPU/FLASH/JBOF (retained only in the
                                      large bottom tier).
  3. Request for prefix A again    -> hit served from the deepest tier that
                                      still holds A (demonstrates the Python-
                                      timed deep-tier reload latency).

Usage:
  python make_tier_demo.py <out.jsonl> [n_filler] [prefix_len] [suffix_len] [out_len] [stagger_ms] [a_len]

`a_len` (default = prefix_len) sets the length of the REUSED prefix A separately
from the filler prefix length, so A can be small (small reload -> latency in the
tier's access-latency scale) while fillers are big enough to evict A downward.
"""
import json
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "workloads/tier_demo_trace.jsonl"
n_filler = int(sys.argv[2]) if len(sys.argv) > 2 else 20
prefix_len = int(sys.argv[3]) if len(sys.argv) > 3 else 320
suffix_len = int(sys.argv[4]) if len(sys.argv) > 4 else 32
out_len = int(sys.argv[5]) if len(sys.argv) > 5 else 8
stagger_ms = float(sys.argv[6]) if len(sys.argv) > 6 else 60.0
a_len = int(sys.argv[7]) if len(sys.argv) > 7 else prefix_len

A = list(range(1, a_len + 1))                 # the reused prefix
rows = []

def rec(idx, prefix, tag):
    base = 5_000_000 + idx * 100_000 + (0 if tag == "A" else 50_000)
    suffix = [base + i for i in range(suffix_len)]
    input_ids = prefix + suffix
    return {
        "input_toks": len(input_ids),
        "output_toks": out_len,
        "arrival_time_ns": int(idx * stagger_ms * 1_000_000),
        "input_tok_ids": input_ids,
        "output_tok_ids": [base + 900_000 + i for i in range(out_len)],
    }

idx = 0
rows.append(rec(idx, A, "A")); idx += 1                       # 1) cache A everywhere
for k in range(n_filler):                                      # 2) evict A from small tiers
    P = list(range(2_000_000 + k * 10_000, 2_000_000 + k * 10_000 + prefix_len))
    rows.append(rec(idx, P, "F")); idx += 1
rows.append(rec(idx, A, "A"))                                  # 3) reuse A -> deep-tier hit

with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(rows)} requests to {out} "
      f"(A[{a_len} toks] + {n_filler} fillers[{prefix_len} toks] + A-reuse)")
