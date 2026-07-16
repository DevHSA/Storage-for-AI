#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cross-rack (cross-tray) context-reuse demo for the pod-wide JBOF.
#
# Proves: context PUSHED to the shared JBOF by tray 0 is reusable by tray 1,
# lowering tray 1's TTFT. Runs the SAME workload twice under --tier-policy
# exclusive: once WITH a pod-wide JBOF, once WITHOUT (falls back to COLDSTORE),
# then prints the phase-2 (cross-rack reuse) TTFT for each so the drop is visible.
#
# Routing is CUSTOM so the producer is pinned to tray 0 and the reuser to tray 1
# (the workload carries a per-request target_instance).
#
# Usage (from the repo root, inside the container):
#   bash run_cross_rack_demo.sh [N_CTX] [CTX_LEN]
# ---------------------------------------------------------------------------
set -e
N_CTX="${1:-16}"          # distinct contexts produced on tray 0 and reused on tray 1
CTX_LEN="${2:-2048}"      # tokens per shared context
FLUSH=4                   # extra tray-0 fills to push the last contexts into JBOF
GAP_MS=8000              # phase 2 arrives after tray 0 has cascaded everything to JBOF

FLAGS="--dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing \
--tier-policy exclusive --cpu-scope per_node --request-routing-policy CUSTOM --max-num-seqs 1"

echo "### generating workload ($N_CTX contexts x $CTX_LEN tokens) ###"
python3 workloads/generators/make_cross_rack.py workloads/cross_rack_demo.jsonl \
        "$N_CTX" "$CTX_LEN" 8 4 "$FLUSH" "$GAP_MS"

echo; echo "### RUN 1 — WITH pod-wide JBOF (reuse reloads from JBOF) ###"
python -m serving --cluster-config configs/cluster/cross_rack_jbof.json $FLAGS \
       --prefix-storage JBOF \
       --dataset workloads/cross_rack_demo.jsonl --output outputs/cross_rack_jbof.csv \
   | grep -iE 'Tier |NPU |CPU |JBOF|COLDSTORE|Total prefix hit' || true

echo; echo "### RUN 2 — WITHOUT JBOF (reuse reloads from slow COLDSTORE) ###"
python -m serving --cluster-config configs/cluster/cross_rack_nojbof.json $FLAGS \
       --prefix-storage COLDSTORE \
       --dataset workloads/cross_rack_demo.jsonl --output outputs/cross_rack_nojbof.csv \
   | grep -iE 'Tier |NPU |CPU |JBOF|COLDSTORE|Total prefix hit' || true

echo; echo "### PHASE-2 (cross-rack reuse on tray 1) TTFT — the proof ###"
P2_START=$(( N_CTX + FLUSH ))
python3 - "$P2_START" <<'PY'
import csv, sys
p2_start = int(sys.argv[1])
def phase2(path):
    xs=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r['request id']) >= p2_start:
                xs.append((int(r['instance id']), float(r['TTFT'])/1e6))
    return xs
for label,path in [("WITH JBOF","outputs/cross_rack_jbof.csv"),
                   ("WITHOUT JBOF","outputs/cross_rack_nojbof.csv")]:
    xs=phase2(path); mean=sum(t for _,t in xs)/len(xs) if xs else 0
    first=xs[0][1] if xs else 0
    insts=sorted({i for i,_ in xs})
    print(f"  {label:<13}: n={len(xs)} on instance(s) {insts}  "
          f"first-reuse TTFT={first:7.2f} ms  mean TTFT={mean:7.2f} ms")
PY
echo "  (WITH-JBOF mean TTFT should be well below WITHOUT-JBOF — cross-rack JBOF reuse wins.)"
