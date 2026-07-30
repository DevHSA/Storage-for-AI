# LLMServingSim — live dashboard

A browser dashboard that shows **live** simulation metrics — throughput,
request/token/batch counters (incl. decode tokens), per-tier memory occupancy
(explicit scope + per-device split + total), per-tier reload latency, a
per-instance drill-down, prefix-cache hits, latency percentiles (TTFT / TBT /
ITL / E2E), and **over-sim-time charts** (throughput, KV-spill memory, TTFT,
TBT) — updating every log interval. Light/dark theme + adjustable refresh rate.
The dark theme uses a near-black neutral background with a warm-orange accent,
steel-slate uppercase labels, thin light numbers (white value + orange unit), and
muted-teal monospace metadata; the KPI stat row is a single rounded container with
faint dividers. The **bars and tier-name labels** in the Request-statistics funnel
and the Memory-by-tier panels are all one uniform orange. Categorical tier colours
(NPU / CPU / FLASH / JBOF / COLDSTORE) are retained only in the **multi-series
charts** (KV-spill stacked area, throughput, etc.) so overlapping series stay
distinguishable.

There are **two dashboards**, chosen by a flag on `serve.py`:

- **V3 — run comparison** (`index_v3.html`, **default**): compares **two runs**
  side by side (see "Comparing two runs" below).
- **V2 — single run** (`index.html`, served with `--dashboardv2`): the detailed
  single-run view. It opens with an **at-a-glance summary** — the model banner, a
  full-width **simulation timeline** with milestone markers, then a **two-column
  hero**: four key charts on the left (average throughput, average TTFT, average
  request completion time, KV spill) and the **request-statistics** pane (progress
  + funnel) with three headline stat tiles (average throughput / TTFT / completion
  time) on the right. Detailed
  charts, tables, drill-downs, and the **Configuration &amp; sizing** panels (tier
  specs, KV-cache unit) follow below. `index_v1.html` is a frozen copy of the
  original single-column layout.

## How it works

1. **Emitter** (in the sim): run with `--dashboard` and the simulator writes a
   full metrics snapshot to `outputs/dashboard/live.json` every log interval
   (and once at the end). Zero overhead when the flag is off.
   (`serving/core/dashboard.py`)
2. **Frontend**: `serve.py` — a zero-dependency stdlib web server that serves the
   HTML page and the snapshot(s). **Recommended** (no pip, no Docker changes). It
   serves the **V3 comparison** page (`index_v3.html`) by default, or the
   **V2 single-run** page (`index.html`) with `--dashboardv2`. It exposes the run-1
   snapshot at `/api/metrics` (from `--file`) and the run-2 snapshot at
   `/api/metrics2` (from `--file2`). (`app.py` is a legacy Streamlit alternative,
   single-run only; needs `pip install streamlit`, Python ≤3.13.)

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
`--log-interval` for denser live updates (more throughput-chart points). Note the
default page is the **V3 comparison** view; add `--dashboardv2` to `serve.py` for
the single-run V2 view above.

## Comparing two runs (V3 — default)

V3 compares **two runs side by side** — the intended use is the same workload with
different memory tiers (e.g. **JBOF vs no-JBOF/COLDSTORE**). Each run is auto-named
by the memory tiers it enables (e.g. `NPU→CPU→JBOF` vs `NPU→CPU→COLDSTORE`).

Runs are **sequential** (not simultaneous): run sim 1 to `live.json`, run sim 2 to
`live2.json`, then open the comparison page (it reads both and updates live as each
completes).

**Terminal A — the comparison server (host):**
```bash
python3 serving/dashboard/serve.py \
  --file outputs/dashboard/live.json --file2 outputs/dashboard/live2.json
# open http://localhost:8000  (V3 is the default page)
```

**Terminal B — run 1, then run 2 (container)** — identical flags except the config
(tiers) and the `--dashboard-file`:
```bash
docker exec servingsim_docker bash -lc "cd /app/LLMServingSim && \
  python -m serving --cluster-config configs/cluster/pod_prop_mini_1gpu_nojbof.json \
    --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing \
    --prefix-storage COLDSTORE --cpu-scope per_instance --tier-policy exclusive \
    --dashboard --dashboard-file outputs/dashboard/live.json --log-interval 0.2 \
    --dataset workloads/example_trace.jsonl --output outputs/run1.csv"

docker exec servingsim_docker bash -lc "cd /app/LLMServingSim && \
  python -m serving --cluster-config configs/cluster/pod_prop_mini_1gpu_jbof.json \
    --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing \
    --prefix-storage JBOF --cpu-scope per_instance --tier-policy exclusive \
    --dashboard --dashboard-file outputs/dashboard/live2.json --log-interval 0.2 \
    --dataset workloads/example_trace.jsonl --output outputs/run2.csv"
```

V3 is laid out to fit the summary within one screen: a **lean full-width two-run
summary** (one box, Run 1 / Run 2 side by side, each named by its tiers — model,
workload, racks, instances, GPUs; both runs' wall-clock times show top-right); a
**compact two-track simulation timeline** (Run 1 white, Run 2 orange, with
bright dash milestone markers — hover any tick for the exact event time; the
end-of-track marker pulses while a run is in progress). The legend lists the
milestones in **chronological order** (earliest first across both runs): first
request arrival, first token generated, first request completed, then the tier
spills (OCTEON JBoF / COLDSTORE) and run complete. (V3 omits the "CPU DRAM spill"
and "all requests arrived" markers that V2 shows.) Then a **two-column
region** —
- **left:** the **performance plots** (two thin lines each, Run 1 white / Run 2
  orange, with a pulsing end-dot while a run is in progress) — average TTFT
  spans the left half (tall), with average throughput and
  average request completion time stacked in the right half. The plot column is a
  fixed height and the plots fill their cells, so toggling the completion-time plot
  never shifts the layout;
- **right:** **Key metrics** as centered tiles (2 columns Run 1 / Run 2 × 4 rows:
  requests finished / total, average throughput, average TTFT, average request
  completion time in seconds, values colour-coded per run) and, below it,
  **Improvement** as three tiles (average TTFT, average throughput, average request
  completion time) that stretch to the plots' height (Run 2 vs Run 1 % change per
  KPI with a "(higher/lower is better)" note, green = better).

A small **CT** toggle in the top control bar (dashed when in P90 mode) swaps the
**third metric** — shown in the completion-time **plot**, its **Key metrics** row,
and its **Improvement** tile — between **Average request completion time** (default)
and **P90 TTFT** (the 90th-percentile TTFT — a percentile, not a mean, hence no
"Average"). The layout is identical either way (same plots, tiles and rows); only
that one slot's label, value and plotted series change. The choice is remembered
across reloads.

> The P90-TTFT **plot curve** needs a run recorded after this change — the snapshot
> now logs `ttft_p90` per interval (`serving/core/dashboard.py`). **Key metrics** and
> **Improvement** use the summary `latency.ttft.p90`, which is present in every run,
> so those two update immediately even for older snapshots.

**Memory by tier** (separate per run) follows below. KV-spill and prefix-cache
panels are omitted in V3. To see meaningful differences, use a workload large enough to spill KV out of
the NPU so the JBOF-vs-COLDSTORE reload cost actually affects TTFT/E2E.

**Resilience:** each poll caches the last snapshot per run that carried real data. If a
poll fails, times out, or the server returns a `{status:"waiting"|"error"}` placeholder
(e.g. the file is momentarily missing, or the stdlib server is briefly starved while a
`--parallel-instances` run saturates the CPU), the dashboard **keeps showing the last
known state** instead of blanking — the footer shows a small "showing last known state —
reconnecting…" hint until a fresh snapshot arrives. A run only shows the empty "waiting"
state before it has ever produced data.

## Cluster config builder (`config_builder.html`)

A **drag-and-drop GUI** for authoring `configs/cluster/*.json` — no hand-editing.
With `serve.py` running, open **http://localhost:8000/config**; it also works
fully offline by opening `serving/dashboard/config_builder.html` directly.

- **Palette** (left): the common components — **Node / Rack, Instance, CPU DRAM,
  OCTEON JBoF, COLDSTORE** — with rarer ones (**FLASH**, **Node CPU pool**) tucked
  under a **more components** dropdown, and the ready-made **templates** under a
  collapsible **templates** toggle. **Click** to add (no scrolling) or **drag** onto
  the topology / a node / an instance; the page auto-scrolls while dragging.
- **Defaults**: a new **Instance** is Qwen3-32B · TP 2 · NPU 36.5 GiB
  (22000 GB/s, 360 ns) with a per-instance **CPU DRAM** of 4 GiB (12000 GB/s,
  113 ns); the run defaults to `--cpu-scope per_instance`. **OCTEON JBoF** defaults
  to 200 GiB (890 GB/s, 8 µs mem-lat; link 28800 GB/s, 1 µs) and **COLDSTORE** to
  200 GiB (890 GB/s, 5 ms mem-lat; link 250 GB/s, 7 µs).
- **Topology** occupies ~¾ of the page height (default) and is the primary surface —
  every component (node, instance, GPU, CPU DRAM / FLASH chip, OCTEON JBoF,
  COLDSTORE) is **clickable**; the selected one is outlined. **Drag the handle
  (⋯⋯⋯) under the topology to resize its height** (persisted).
- **A single inspector pane below the topology** shows the knobs for whatever you
  clicked — cluster root + run settings (default / breadcrumb "Cluster"), a node
  (with its add buttons + instance list), an instance (model / hardware / TP / NPU /
  advanced), or a memory tier (capacity / BW / latency, presets). Vera-Rubin and
  scaled presets are one click; templates seed a Qwen 2×2 or a Vera-Rubin tray.
- **Two ways to attach CPU DRAM**: the default **CPU DRAM** is **per-instance** —
  host DRAM dedicated to a single instance/superchip → per-instance `cpu_mem`,
  `--cpu-scope per_instance` (e.g. Vera Rubin's 1 CPU : 2 GPU superchip). A
  node-wide **Node CPU pool** (host DRAM shared by all the node's instances →
  node-level `cpu_mem`, `--cpu-scope per_node`) is available under *more components*.
- **Live validation** against the real parser rules (weights must fit the NPU, KV
  room ≥ one block, a node needs a CPU pool or per-instance CPU on all instances,
  TP∈{1,2} is profiled, prefix-storage vs declared tiers) plus a live
  weight / KV-per-token / smallest-block readout per instance.
- **Download** the JSON, **Copy** it, or (when served) **Save to
  configs/cluster/** to write it into the repo. The **run command** panel shows
  the matching CLI flags (which live outside the JSON).

## Streamlit alternative

If you have a Python with pip (≤3.13):
```bash
pip install streamlit
streamlit run serving/dashboard/app.py        # http://localhost:8501
```
Point at a non-default snapshot via the sidebar or `LLMSS_DASHBOARD_FILE`.

## Metrics shown

- **Header** — status (starting/running/done), model, **workload** (dataset
  filename), topology (nodes · instances · GPUs · TP), `--prefix-storage`,
  dtype / block, sim clock, and **system time elapsed** (wall-clock). Controls for
  the **refresh rate** (0.5 s … 10 s — live-reschedules the poll loop) and a
  **light/dark** toggle. (The finished / total progress bar lives at the top of
  the request-statistics pane, not the header.)
- **⚡ Average throughput card** — the **static** number to compare between runs:
  **total tok/s** (headline), decode (gen) tok/s, prompt (prefill) tok/s, req/s, plus
  **active tok/s** (average over intervals that produced tokens — idle-robust) and
  **peak tok/s**. `total_*` divide by the full sim time (makespan), so an idle gap
  dilutes them; `active` reflects the true serving rate for bursty/gapped workloads.
  (Same numbers print in the console `Throughput Results` block.)
- **Tier specs panel** (collapsible, in the **Configuration &amp; sizing** section
  at the bottom of the page) — per tier: scope, mem bw/latency, and (for the deep
  tiers) the BlueField link bw/latency.
- **Tiles** — requests finished/running/waiting; prompt / decode / total tokens;
  batches; req/s; decode & prompt tok/s (cumulative + **last / peak**, so they
  stay meaningful at `done` when the instantaneous rate reads 0).
- **Simulation timeline** (full-width bar under the banner) — a moving playhead
  sweeps a mostly-static axis seeded from the workload's last arrival time, with
  milestone markers: first request arrival, first token generated, first request
  completed, CPU / JBOF / COLDSTORE spill begins (only when a tier actually
  spills), all requests arrived, run complete. The end is bumped once (projected
  from the live decode rate + remaining decode tokens) if the playhead reaches it.
  Pass **`--timeline-seconds N`** to set an explicit initial ceiling (in seconds)
  for this axis instead of the last-arrival estimate — that value is the fixed
  axis end until the run breaches it (then it extends), and it **also locks every
  time-series plot's x-axis to the same ceiling**, so the timeline and all charts
  share one time axis. Display-only; it does not stop the simulation.
- **Request statistics** (right side of the hero) — the pane leads with the
  **finished / total** progress bar (`Requests finished X / Y`) at its top,
  followed by cumulative bars for the stages a request passes through: **arrived**
  (`arrival ≤ clock`) → **started**
  (prefill admitted, `queuing_delay ≥ 0`) → **decoding** (first token out,
  `ttft ≥ 0`). They nest (arrived ≥ started ≥ decoding); the gaps between bars are
  requests currently queued or prefilling. The denominator is the **total request
  count parsed from the workload file up front** (`est_total_requests` — one per
  line for flat schemas, one per `sub_request` for sessions; falls back to the
  live routed count), so the bars fill toward the true final total. (The old
  "finished" bar was dropped as redundant with the progress bar and the headline
  stat tiles.)
- **Hero summary stats** (right side, under the funnel) — three uniformly-styled
  headline tiles for quick comparison: **average throughput** (tok/s), **average
  TTFT** (ms), and **average request completion time** (ms, or seconds when large).
- **Hero compact panels** (right side, under the stats — split into two halves):
  a **🎯 Prefix cache** mini panel (overall hit ratio + per-tier hit-tokens / hit %,
  **without** the reload-latency bars or reload column) on the left, and a
  **🧠 Memory by tier** mini panel (per-tier **overall** used/cap bar only — no
  per-device split, to keep it short) on the right. These are compact copies; the
  **full** Prefix-cache (with reload-latency bars) and Memory-by-tier (with the
  per-device breakdown) panels still appear in the detail area below.
- **Over-sim-time charts** — the four **key** charts sit in the hero (left
  column), each one line: **Average throughput** (cumulative-average total tok/s =
  tokens so far ÷ elapsed sim time, converging to the whole-run average);
  **Average TTFT** (cumulative mean time-to-first-token over finished requests, ms);
  **Average request completion time** (cumulative mean end-to-end latency =
  last-token − arrival, ms); and **KV-spill memory** (stacked area of CPU → deep
  tiers, so you watch the cascade fill). Average TTFT and completion time are the
  same metrics as their headline stat
  tiles, sampled each Δ; both read 0 until the first request completes, then settle
  to the true average. (TBT and the per-tier percentiles remain in the Latency
  table below.) Each chart draws a full grid — horizontal value markers and
  vertical time gridlines with large-font axis labels — for easier reading.
- **Per-chart explainers** — every chart/table carries a collapsible
  *"ℹ️ How this is calculated"* block with two views: a **Technical** one (the
  exact formula) and an **In plain words** one (the intuition).
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
- **NPU HBM** — per-instance weight + KV occupancy, plus **KV/token** and
  **KV/block** (the smallest KV-cache allocation unit) per GPU.
- **KV-cache unit card** (in the **Configuration &amp; sizing** section at the
  bottom) — for the loaded model/config: KV bytes per token (per-GPU and
  full-context = ×tp_size) and one block = `block_size` tokens = the smallest KV
  unit. Also printed as a `KV Cache Unit` section in the console summary of every
  run.
