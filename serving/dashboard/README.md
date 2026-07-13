# LLMServingSim — live dashboard

A browser dashboard that shows **live** simulation metrics — throughput,
request/token/batch counters (incl. decode tokens), per-tier memory occupancy
(pooled vs per-device, with the per-device split + total), prefix-cache hits,
and latency percentiles (TTFT / TBT / ITL / E2E) — updating every log interval.

## How it works

1. **Emitter** (in the sim): run with `--dashboard` and the simulator writes a
   full metrics snapshot to `outputs/dashboard/live.json` every log interval
   (and once at the end). Zero overhead when the flag is off.
   (`serving/core/dashboard.py`)
2. **Frontend**: two options read that JSON —
   - **`serve.py` + `index.html`** — a zero-dependency stdlib web server + a
     self-contained HTML/JS page. **Recommended** (no pip, no Docker changes).
   - **`app.py`** — a Streamlit app (needs `pip install streamlit`, Python ≤3.13).

## Quick start (zero-install HTML dashboard)

Two terminals, both from the repo root.

**Terminal A — the dashboard server (on the host):**
```bash
python3 serving/dashboard/serve.py            # http://localhost:8000
# options: --port 8000  --file outputs/dashboard/live.json
```

**Terminal B — a simulation with `--dashboard` (in the container):**
```bash
docker exec servingsim_docker bash -lc "cd /app/LLMServingSim && \
  python -m serving --cluster-config configs/cluster/vera_rubin_tray_small.json \
    --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing \
    --prefix-storage COLDSTORE --cpu-scope per_instance --tier-policy exclusive \
    --dashboard --log-interval 0.2 \
    --dataset workloads/example_trace.jsonl --output outputs/run.csv"
```

Open **http://localhost:8000** — it auto-refreshes ~2×/sec. Lower
`--log-interval` for denser live updates (more throughput-chart points).

## Streamlit alternative

If you have a Python with pip (≤3.13):
```bash
pip install streamlit
streamlit run serving/dashboard/app.py        # http://localhost:8501
```
Point at a non-default snapshot via the sidebar or `LLMSS_DASHBOARD_FILE`.

## Metrics shown

- **Header** — status (starting/running/done), model, topology (nodes ·
  instances · GPUs · TP), `--tier-policy` / `--cpu-scope` / `--prefix-storage`,
  sim clock, wall time, progress (finished / total requests).
- **Tiles** — requests finished/running/waiting; prompt / decode / total tokens;
  batches; req/s; decode tok/s (cumulative + live); total tok/s.
- **Throughput chart** — prompt vs decode tok/s over sim time.
- **Memory by tier** — per tier: scope badge (**pooled (pod-wide)** vs
  **per-node** / **per-instance**), total used/cap bar, and for per-device tiers
  the per-device split with a total.
- **Prefix cache** — overall hit ratio + per-tier hit tokens / hit % / reload ms.
- **Latency** — TTFT / TBT / ITL / E2E: mean / p50 / p90 / p99 / max.
- **NPU HBM** — per-instance weight + KV occupancy.
