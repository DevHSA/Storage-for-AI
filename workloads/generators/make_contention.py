"""Same-node reload-contention workload for the JBOF FIFO channel.

Goal: force MANY instances on the SAME node to reload the SAME contexts from the
pod-wide JBOF at the SAME sim-time, so they SERIALIZE on that node's single JBOF
channel (device_id = node_id). Single-process models this FIFO in one shared
AnalyticalMemory (contended). Mode B Stage 1 gives each instance its own process
(uncontended -> under-counts). Mode B Stage 2 re-imposes the FIFO in Python.

PHASE 1 (producer, instance 0, arrival 0):
    n_ctx DISTINCT ctx_len-token contexts computed on instance 0; exclusive-mode
    eviction cascades them NPU->CPU->JBOF; `flush` dummies push the last ones out
    of instance 0's small NPU/CPU into JBOF.

PHASE 2 (contended reuse, arrival +gap, all `n_inst` instances at once):
    every instance requests every context (ctx + short tail) at the SAME arrival,
    so all n_inst reloads of each context hit the ONE JBOF channel together.

JSONL schema: input_toks, output_toks, arrival_time_ns, target_instance,
              input_tok_ids, output_tok_ids.

Usage:
  python make_contention.py <out.jsonl> [n_ctx] [ctx_len] [n_inst] [tail_len] [out_len] [flush] [gap_ms]
"""
import json
import sys

out      = sys.argv[1] if len(sys.argv) > 1 else "workloads/contention_demo.jsonl"
n_ctx    = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ctx_len  = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
n_inst   = int(sys.argv[4]) if len(sys.argv) > 4 else 4
tail_len = int(sys.argv[5]) if len(sys.argv) > 5 else 8
out_len  = int(sys.argv[6]) if len(sys.argv) > 6 else 4
flush    = int(sys.argv[7]) if len(sys.argv) > 7 else 4
gap_ms   = float(sys.argv[8]) if len(sys.argv) > 8 else 8000.0

STRIDE = ctx_len + tail_len + 16
gap_ns = int(gap_ms * 1_000_000)

def ctx_ids(i):   base = 1 + i * STRIDE;              return list(range(base, base + ctx_len))
def tail_ids(i):  base = 1 + i * STRIDE + ctx_len;    return list(range(base, base + tail_len))
def flush_ids(k): base = 1 + (10_000 + k) * STRIDE;   return list(range(base, base + ctx_len))
def out_ids(tag): base = 500_000_000 + tag * 64;      return list(range(base, base + out_len))

rows = []
n_prod = n_inst * n_ctx        # producer makes one DISTINCT context per phase-2 reuse
# PHASE 1 — producer on instance 0
for i in range(n_prod):
    ids = ctx_ids(i)
    rows.append({"input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": 0,
                 "target_instance": 0, "input_tok_ids": ids, "output_tok_ids": out_ids(i)})
for k in range(flush):
    ids = flush_ids(k)
    rows.append({"input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": 0,
                 "target_instance": 0, "input_tok_ids": ids, "output_tok_ids": out_ids(10_000 + k)})
# PHASE 2 — each instance reloads its OWN DISTINCT contexts at the SAME arrival, so
# the reloads pile onto the ONE shared channel together (pure channel contention)
# WITHOUT fighting over the same context (which exclusive mode could only keep in
# one place). Instance j reuses contexts [j*n_ctx : (j+1)*n_ctx).
tag = 1
for j in range(n_inst):
    for k in range(n_ctx):
        i = j * n_ctx + k
        ids = ctx_ids(i) + tail_ids(i)
        rows.append({"input_toks": len(ids), "output_toks": out_len, "arrival_time_ns": gap_ns,
                     "target_instance": j, "input_tok_ids": ids, "output_tok_ids": out_ids(20_000 + tag)})
        tag += 1

with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

p2 = n_prod + flush
print(f"wrote {len(rows)} requests to {out}: "
      f"phase1={n_prod} DISTINCT producers (+{flush} flush) on instance 0 @t=0, "
      f"phase2={n_inst}x{n_ctx}={n_inst*n_ctx} simultaneous reloads (distinct per instance) across "
      f"{n_inst} instances @t={gap_ms}ms. ctx_len={ctx_len}, tail={tail_len}. "
      f"phase-2 indices = [{p2}, {len(rows)}).")
