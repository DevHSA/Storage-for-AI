"""Generate a tiny prefix-cache demonstration trace.

Every request shares one long common prefix (so later requests hit the prefix
cache) followed by a per-request unique suffix. Staggered arrivals let the first
request populate the cache before the rest arrive.

Usage:
    python make_shared_prefix.py <out.jsonl> [n_reqs] [prefix_len] [suffix_len] [out_len] [stagger_ms]
"""
import json
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "workloads/shared_prefix_trace.jsonl"
n_reqs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
prefix_len = int(sys.argv[3]) if len(sys.argv) > 3 else 320
suffix_len = int(sys.argv[4]) if len(sys.argv) > 4 else 64
out_len = int(sys.argv[5]) if len(sys.argv) > 5 else 16
stagger_ms = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0

shared_prefix = list(range(1, prefix_len + 1))

with open(out, "w") as f:
    for r in range(n_reqs):
        base = 1_000_000 + r * 10_000
        suffix = [base + i for i in range(suffix_len)]
        input_ids = shared_prefix + suffix
        output_ids = [base + 500_000 + i for i in range(out_len)]
        rec = {
            "input_toks": len(input_ids),
            "output_toks": out_len,
            "arrival_time_ns": int(r * stagger_ms * 1_000_000),
            "input_tok_ids": input_ids,
            "output_tok_ids": output_ids,
        }
        f.write(json.dumps(rec) + "\n")

print(f"wrote {n_reqs} requests to {out} "
      f"(shared prefix {prefix_len} toks, unique suffix {suffix_len}, output {out_len})")
