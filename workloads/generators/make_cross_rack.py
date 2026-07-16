"""Cross-rack (cross-tray) context-reuse workload for the JBOF demo.

Goal: prove that context PUSHED to the pod-wide JBOF by ONE tray is reusable by
ANOTHER tray, lowering that tray's TTFT. Run under:
    --tier-policy exclusive   (victim cache: evicted context cascades NPU->CPU->JBOF)
    --request-routing-policy CUSTOM   (honors the per-request target_instance below)
with >= 2 instances (tray 0 = producer, tray 1 = consumer).

PHASE 1  (producer, tray 0, arrival 0):
    n_ctx DISTINCT `ctx_len`-token contexts are computed on instance 0. In
    exclusive mode each is evicted NPU->CPU->JBOF as later ones arrive, so by the
    end tray 0's contexts live in the pod-wide JBOF. `flush` extra dummy contexts
    (also on tray 0) push the LAST real contexts out of tray 0's small NPU/CPU
    into JBOF too, so every context is reload-able from the shared pool.

PHASE 2  (consumer, tray 1, arrival +gap):
    instance 1 reuses each context (oldest first) as `ctx + short tail`. Tray 1
    never computed them and cannot see tray 0's (per-node) NPU/CPU, so the reload
    comes from the pod-wide deep tier:
        - WITH JBOF   -> reload from JBOF      (fast)  -> low  TTFT
        - WITHOUT JBOF -> reload from COLDSTORE (slow) -> high TTFT
    The phase-2 TTFT difference IS the cross-rack JBOF benefit.

JSONL schema: input_toks, output_toks, arrival_time_ns, target_instance,
              input_tok_ids, output_tok_ids.

Usage:
  python make_cross_rack.py <out.jsonl> [n_ctx] [ctx_len] [tail_len] [out_len] [flush] [gap_ms]
"""
import json
import sys

out      = sys.argv[1] if len(sys.argv) > 1 else "workloads/cross_rack_demo.jsonl"
n_ctx    = int(sys.argv[2]) if len(sys.argv) > 2 else 24
ctx_len  = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
tail_len = int(sys.argv[4]) if len(sys.argv) > 4 else 8
out_len  = int(sys.argv[5]) if len(sys.argv) > 5 else 4
flush    = int(sys.argv[6]) if len(sys.argv) > 6 else 4
gap_ms   = float(sys.argv[7]) if len(sys.argv) > 7 else 8000.0

STRIDE = ctx_len + tail_len + 16          # disjoint token-id block per context
gap_ns = int(gap_ms * 1_000_000)

def ctx_ids(i):   base = 1 + i * STRIDE;              return list(range(base, base + ctx_len))
def tail_ids(i):  base = 1 + i * STRIDE + ctx_len;    return list(range(base, base + tail_len))
def flush_ids(k): base = 1 + (10_000 + k) * STRIDE;   return list(range(base, base + ctx_len))
def out_ids(i, ph): base = 500_000_000 + ph * 250_000_000 + i * 32; return list(range(base, base + out_len))

rows = []
# PHASE 1 — producer on tray 0 (instance 0)
for i in range(n_ctx):
    ids = ctx_ids(i)
    rows.append({"input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": 0,
                 "target_instance": 0, "input_tok_ids": ids, "output_tok_ids": out_ids(i, 0)})
# flush dummies on tray 0 — push the last real contexts out of tray 0's NPU/CPU into JBOF
for k in range(flush):
    ids = flush_ids(k)
    rows.append({"input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": 0,
                 "target_instance": 0, "input_tok_ids": ids, "output_tok_ids": out_ids(10_000 + k, 0)})
# PHASE 2 — consumer on tray 1 (instance 1) reuses tray 0's contexts, oldest first
for i in range(n_ctx):
    ids = ctx_ids(i) + tail_ids(i)
    rows.append({"input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": gap_ns,
                 "target_instance": 1, "input_tok_ids": ids, "output_tok_ids": out_ids(i, 1)})

with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(rows)} requests to {out}: "
      f"phase1={n_ctx} producers (+{flush} flush) on tray 0 @t=0, "
      f"phase2={n_ctx} cross-rack reuses on tray 1 @t={gap_ms}ms. "
      f"ctx_len={ctx_len}, tail={tail_len}. "
      f"phase-2 request indices = [{n_ctx + flush}, {2 * n_ctx + flush}). "
      f"Reuse reloads from JBOF (with) vs COLDSTORE (without).")
