# LLMServingSim — live dashboard

A browser dashboard that shows **live** simulation metrics — throughput,
request/token/batch counters (incl. decode tokens), per-tier memory occupancy
(explicit scope + per-device split + total), per-tier reload latency, a
per-instance drill-down, prefix-cache hits, latency percentiles (TTFT / TBT /
ITL / E2E), and **over-sim-time charts** (throughput, KV-spill memory, TTFT,
TBT) — updating every log interval. Light/dark theme + adjustable refresh rate.

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
  sim clock, wall time, progress (finished / total requests). Controls for the
  **refresh rate** (0.5 s … 10 s) and a **light/dark** toggle.
- **Tier specs panel** (collapsible) — per tier: scope, mem bw/latency, and (for
  the deep tiers) the BlueField link bw/latency.
- **Tiles** — requests finished/running/waiting; prompt / decode / total tokens;
  batches; req/s; decode & prompt tok/s (cumulative + **last / peak**, so they
  stay meaningful at `done` when the instantaneous rate reads 0).
- **Over-sim-time charts** — throughput (prompt vs decode tok/s); **KV-spill
  memory** (stacked area of CPU → deep tiers, so you watch the cascade fill);
  **TTFT** mean; **TBT** mean.
- **Memory by tier (now)** — the tiered **KV cache** (not total HBM). The **NPU**
  row is KV-cache only: its capacity = per-GPU HBM **minus resident model weights**
  (so a TP=2 Qwen3-32B on 62 GiB GPUs shows ~31.5 GiB KV, not 62); the weights are
  in the *NPU HBM* table. Each per-device entry is **one instance** (a TP=N
  superchip spanning N GPUs, shown as `inst k · N×GPU`). Per tier: explicit scope
  badge — NPU **per-GPU**,
  CPU **per-instance / per-node** (by `--cpu-scope`), FLASH **per-node**,
  JBOF/COLDSTORE **pooled (pod-wide)** — total used/cap bar, and for per-device
  tiers the per-device split with a total.
- **Prefix cache** — overall hit ratio + per-tier hit tokens / hit % / reload ms,
  with a per-tier **reload-latency** bar chart (slow-tier cost at a glance).
- **Latency** — TTFT / TBT / ITL / E2E: mean / p50 / p90 / p99 / max.
- **Compute drill-down (by tray / by GPU)** — a toggle switches between:
  - **By tray** — a *tray* = up to 4 GPUs (2 TP=2 superchips) grouped within a
    node (dashboard-only rollup; the sim still runs the TP=N instances). Per tray:
    #GPUs, #CPUs (distinct host-memory pools it owns), HBM used/%, CPU KV/%,
    running / waiting requests, hit ratio.
  - **By GPU/instance** — per GPU/instance: running / waiting, NPU HBM %, KV
    bytes, prefix-cache hit ratio.
- **NPU HBM** — per-instance weight + KV occupancy.
