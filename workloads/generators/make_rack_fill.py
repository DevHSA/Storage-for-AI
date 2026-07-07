"""Generate a 'fill-the-tiers-in-order then reuse' workload for the JBOF study.

Two phases:
  PHASE 1 (fill):  n_ctx DISTINCT long contexts, each `ctx_len` unique tokens.
                   Computing them fills the hierarchy in order — every GPU's NPU
                   first, then the small shared CPU pool, then FLASH, landing the
                   bulk in JBOF (with JBOF) or COLDSTORE (without). This is what
                   'fill lower tiers before higher tiers' means.
  PHASE 2 (reuse): reuse each context (OLDEST first) as `Ci + short unique tail`.
                   By phase 2, an old Ci has been evicted from NPU/CPU/FLASH by
                   later fillers but survives in the big deep tier, so each reuse
                   RELOADS Ci from JBOF (JBOF) vs COLDSTORE (no JBOF) — the whole
                   with/without-JBOF comparison, paid ~n_ctx times (saturated).

All requests share arrival semantics of a saturated burst; phase 2 arrives a
gap after phase 1 so the fill completes (and old contexts get evicted) first.
Reuse order = fill order (oldest first) so every reuse misses the shallow tiers.

JSONL record schema matches the loader: input_toks, output_toks, arrival_time_ns,
input_tok_ids, output_tok_ids.

Usage:
  python make_rack_fill.py <out.jsonl> [n_ctx] [ctx_len] [tail_len] [out_len] [phase2_gap_ms]
"""
import json
import sys

out       = sys.argv[1] if len(sys.argv) > 1 else "workloads/rack_fill_trace.jsonl"
n_ctx     = int(sys.argv[2]) if len(sys.argv) > 2 else 1440
ctx_len   = int(sys.argv[3]) if len(sys.argv) > 3 else 512     # >=256 so contexts occupy NPU + CPU page
tail_len  = int(sys.argv[4]) if len(sys.argv) > 4 else 8
out_len   = int(sys.argv[5]) if len(sys.argv) > 5 else 4
gap_ms    = float(sys.argv[6]) if len(sys.argv) > 6 else 3000.0

STRIDE = ctx_len + tail_len + 16          # disjoint token-id block per context (guarantees unique KV)
gap_ns = int(gap_ms * 1_000_000)

def ctx_ids(i):
    base = 1 + i * STRIDE
    return list(range(base, base + ctx_len))

def tail_ids(i):
    base = 1 + i * STRIDE + ctx_len
    return list(range(base, base + tail_len))

def out_ids(i, phase):
    base = 500_000_000 + phase * 250_000_000 + i * 32
    return list(range(base, base + out_len))

with open(out, "w") as f:
    # PHASE 1 — fill: compute each distinct context once (arrival 0 = saturating burst).
    for i in range(n_ctx):
        ids = ctx_ids(i)
        f.write(json.dumps({
            "input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": 0,
            "input_tok_ids": ids, "output_tok_ids": out_ids(i, 0),
        }) + "\n")
    # PHASE 2 — reuse each context (oldest first) -> reload from the deepest tier it survives in.
    for i in range(n_ctx):
        ids = ctx_ids(i) + tail_ids(i)
        f.write(json.dumps({
            "input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": gap_ns,
            "input_tok_ids": ids, "output_tok_ids": out_ids(i, 1),
        }) + "\n")

print(f"wrote {2*n_ctx} requests to {out}: phase1={n_ctx} distinct {ctx_len}-tok fills @t=0, "
      f"phase2={n_ctx} reuses (+{tail_len}-tok tail) @t={gap_ms}ms. "
      f"Total unique KV ~= {n_ctx*ctx_len} tokens.")
