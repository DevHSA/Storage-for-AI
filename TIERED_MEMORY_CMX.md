# Tiered prefix-cache spillover & the CMX (Context Memory) study

This note explains the multi-tier KV-cache feature added to LLMServingSim, how it
works end-to-end, how it maps to the NVIDIA Vera Rubin memory hierarchy, and where
the **simulator's** performance bottleneck is when you scale.

> **Status (2026-07-05):** the deep tiers (FLASH / ICMS / COLDSTORE) are now
> **timed by ASTRA-Sim's analytical memory model** — each reload is a real
> `MEM_LOAD` node, and concurrent reloads against a shared pooled tier
> **contend**. This replaced the original design in which the deep tiers were
> Python-only (a precomputed latency injected as an all-`LOCAL` COMP node). See
> §6 for the evolution and why the rack-scale numbers changed.

---

## 1. What it is

Baseline LLMServingSim caches prefixes (RadixAttention) in GPU HBM (NPU) with an
optional single second tier (`--prefix-storage CPU|CXL`). This feature generalizes
that to an **arbitrary, config-driven spillover chain**:

```
NPU (HBM) -> CPU (host DRAM) -> FLASH -> ICMS -> COLDSTORE
```

- A tier is **active iff its block is present in the cluster config** (any subset).
  `--prefix-storage` sets the deepest tier to consider; absent tiers are skipped.
- Each tier declares `mem_size`, `mem_bw`, `mem_latency` (+ `link_bw`, `link_latency`
  for network-attached tiers). **Cache membership** (which prefix lives where) is
  managed in the Python wrapper (one RadixCache per tier). **Reload timing** is done
  by ASTRA-Sim: each tier is emitted into `memory_expansion.json` and every reload is
  a real `MEM_LOAD` node routed to that tier's memory model (see §3).

### Mapping to Vera Rubin NVL72
| Our tier | Vera Rubin component | Cap/GPU | Access latency |
|---|---|---|---|
| NPU | Rubin HBM4 | 288 GB | ~10 ns |
| CPU | Vera LPDDR5X | ~768 GB | ~80–120 ns |
| ICMS | **CMX / ICMSP** (Ethernet-attached pooled NVMe flash, behind BlueField-4) | ~16 TB | ~80 µs + ~3 µs RDMA |
| COLDSTORE | network storage (G4) | large | ~2 ms + ~20 µs |

CMX is architecturally our **ICMS** (pooled, separate-node tier). NVIDIA's "CMX vs.
traditional storage" claim is exactly our **with-ICMS vs. without-ICMS** comparison.

CXL (`--prefix-storage CXL`) remains a **standalone** legacy second tier, also
ASTRA-Sim-timed (via `cxl_mem`), but it is *not* a level inside the NPU→…→COLDSTORE
chain (`_TIER_CHAIN = ['CPU','FLASH','ICMS','COLDSTORE']`).

---

## 2. Reference example (small, tiers saturated)

Configs `configs/cluster/vera_rubin_small_cmx.json` and `..._nocmx.json` (1 GPU,
Vera Rubin latencies, small caps so the tiers fill), workload `workloads/vr_sat_trace.jsonl`
(reused 8-token context **A** + 20 filler prompts of 512 tokens + one reuse of A).

Run (see `serving/run.sh`, "Exp 0"):
```bash
python -m serving --cluster-config configs/cluster/vera_rubin_small_cmx.json \
  --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing \
  --prefix-storage COLDSTORE --dataset workloads/vr_sat_trace.jsonl --output outputs/vr_small_cmx.csv
# swap _cmx -> _nocmx for the comparison
```

Result: **NPU 96% full, CPU 75% full** (the fillers); the reused context is served
from **ICMS in ~0.11 ms (with CMX)** vs **COLDSTORE in ~2.35 ms (without, ~21×)**.
This is a **single GPU**, so there is no cross-NPU pool contention — the delta here is
purely the per-reload latency of the tier (§4 explains what changes at rack scale).

---

## 3. How the code works

```
__main__.py: parse CLI + cluster config
  -> config_builder.py: write ASTRA-Sim inputs (network.yml/system.json/memory_expansion.json).
       CPU->remote_mem, CXL->cxl_mem, and now FLASH/ICMS/COLDSTORE are ALSO written
       into memory_expansion.json so ASTRA-Sim times their reloads.
  -> build one Scheduler per instance + shared pooled tier caches (RadixCache/tier)
  -> launch ONE persistent ASTRA-Sim co-process
  -> main loop, per iteration, per instance:
       scheduler.schedule_with_prefix(): prefix_match -> token budget -> NPU evict -> Batch(+tier_loads)
       trace_generator.generate_trace(): Batch -> text trace .txt (tier reload = MEM_LOAD row)
       graph_generator: text trace -> Chakra .et protobuf (spawns chakra_converter)
       controller: signal ASTRA-Sim (stdin) -> it simulates -> returns cycle count (stdout)
       scheduler.add_done(): update requests; write-through prefixes into all tiers
  -> results (throughput / per-tier / TTFT) + per-request CSV + outputs/<name>.txt
```

### Caching mechanism (Python — unchanged)
RadixAttention prefix cache (SGLang-style radix tree), **one `RadixCache` per tier**:
- NPU page = `--block-size` (16); CPU pool page = 256; deep tiers (FLASH/ICMS/COLDSTORE) page = 1.
- **Inclusive write-through:** a computed prefix is inserted into NPU *and* every active
  pooled tier. Deeper tiers are larger, so they retain a prefix after shallower tiers
  evict it (LRU) — that is the spillover.
- **Lookup** probes NPU then each tier in order and assigns each prefix segment to the
  **shallowest tier that still holds it**. NPU hits cost 0; a pooled-tier hit is reloaded
  into NPU.
- Note: a context shorter than the NPU block (16) / CPU page (256) lives *only* in the
  token-granular pooled tiers — this is why very short requests show NPU/CPU = 0% and
  why the small example uses long *fillers* to fill NPU/CPU.

This layer decides **which tier serves a reload and how many bytes**; ASTRA-Sim then
**times** that reload (below).

### Intermediate file — the reload is a real `MEM_LOAD`
Per scheduled batch, a **text trace** `astra-sim/inputs/runs/<run_id>/trace/instance{i}_batch{b}.txt`
is written, then converted to a **`.et` protobuf** by the Chakra converter. Format:
```
COLOCATED   model_parallel_NPU_group: 1
293                                                  # layer count
Layername   comp_time  input_loc input_size  weight_loc weight_size  output_loc output_size  comm_type comm_size misc
embedding_1 4821  REMOTE:0 2080  LOCAL 1050673152  LOCAL 4259840  NONE 0 NONE   # first layer input from REMOTE (CPU)
icms_load_1 1     LOCAL 0  ICMS:0 1048576  LOCAL 0  NONE 0 NONE                 # <-- deep-tier reload: real MEM_LOAD to ICMS
...
kv_evict_0  0     LOCAL 0  REMOTE:0 23601152  LOCAL 0  NONE 0 NONE              # KV spill to CPU (MEM_STORE, timed by ASTRA-Sim)
sampler_292 24746 LOCAL 133386240 LOCAL 0  REMOTE:0 2080  NONE 0 NONE           # last layer output to REMOTE (CPU)
```
- `comp_time` = ns; locations use the enum `LOCAL(1)/REMOTE(2)/CXL(3)/STORAGE(4)/FLASH(5)/ICMS(6)/COLDSTORE(7)`.
- The **`<tier>_load` node** carries the reload in its **WEIGHT column**
  (`weight_loc=<TIER>:<device_id>`, `weight_size=<reload_bytes>`). The Chakra converter
  turns any middle-layer row whose `weight_loc != LOCAL` and `weight_size > 0` into a
  `MEM_LOAD` node and wires it as a data-dep parent of that row's COMP node, i.e. **on
  the request's critical path**. Its `comp_time` is a **negligible 1 ns anchor** whose only
  job is to force the converter to create that COMP node (it skips comp nodes with
  `comp_time == 0`) — the *real* reload latency is computed by ASTRA-Sim, so there is **no
  double-count**. It must be a **middle** node (the converter reads the first layer's
  `input_loc` / last layer's `output_loc` for its own boundary MEM_LOAD/MEM_STORE).

### How ASTRA-Sim times a reload (analytical memory model)
`config_builder.py` writes each active deep tier into `memory_expansion.json`, e.g.:
```json
"icms_mem":      { "memory-type": "MEMORY_POOL",             "mem-bw": 32, "mem-latency": 100000,
                   "num-devices": 1, "link-bw": 64, "link-latency": 1000 },
"coldstore_mem": { "memory-type": "MEMORY_POOL",             "mem-bw": 4,  "mem-latency": 3000000,
                   "num-devices": 1, "link-bw": 16, "link-latency": 10000 },
"flash_mem":     { "memory-type": "PER_NODE_MEMORY_EXPANSION","mem-bw": 8,  "mem-latency": 8000, "num-devices": <num_nodes> }
```
ASTRA-Sim's `AnalyticalMemory::get_mem_runtime` times each `MEM_LOAD` as
`mem_latency + bytes/mem_bw + (link_bw>0 ? link_latency + bytes/link_bw : 0)` — the
same closed form the Python metric uses, so an **uncontended** reload matches the old
model exactly (formula parity / regression-safe).

**Contention.** There is one shared `AnalyticalMemory` instance per tier (handed to
every NPU's `Sys`). It serializes requests FIFO per `device_id`
(`ongoing_transaction[device_id]`), where `device_id` is the `:N` suffix in the trace
token. The Python wrapper emits:
- **ICMS / COLDSTORE** as `MEMORY_POOL`, `num-devices = 1`, `device_id = 0` for all NPUs
  → every reload serializes on **one** queue == shared-bandwidth contention (the rack
  pool is the bottleneck).
- **FLASH** as `PER_NODE_MEMORY_EXPANSION`, `device_id = node_id` → contention only
  within a node.

Both are tunable per tier via optional config fields **`memory_type`** and
**`num_devices`** (e.g. set `"memory_type": "PER_NPU_MEMORY_EXPANSION"` to disable
contention, or `"num_devices": K` to model *K* parallel channels). This is the single
knob that sets where a pool sits between "infinite parallelism" and "one shared channel."

> Serialization is a first-order bandwidth-bottleneck model (each request runs at full
> `mem_bw`, one at a time) — same aggregate makespan as ideal proportional sharing, but
> not identical per-request finish times. It matches how CXL `MEMORY_POOL` already
> behaved. True proportional bandwidth sharing would be a larger `AnalyticalMemory`
> change.

### ASTRA-Sim granularity
ASTRA-Sim runs as **one persistent co-process**. The wrapper drives it **per scheduled
batch**: write the `.et` graph -> signal over stdin -> ASTRA-Sim cycle-level-simulates that
step -> prints `sys[i] iteration N finished, C cycles` on stdout -> `controller.parse_output`
advances the clock. So: **one ASTRA-Sim graph per batch, per instance, per iteration**.

### Files changed for the ASTRA-Sim-timed tiers
- C++ (needs `scripts/compile.sh`): `AstraMemoryAPI.hh` (enum +3), `Sys.hh`/`Sys.cc`
  (members + mapping), `Workload.cc` (issue_mem cases), `AnalyticalMemory.hh`/`.cc`
  (link term), and both analytical `main.cc` frontends (parse `flash/icms/coldstore_mem`).
  (ns3 frontend not updated — analytical backend only.)
- Chakra converter: `llm_converter.py` (`MemoryType` +3, `get_mem_type` strings).
- Python: `config_builder.py` (emit tier blocks + `memory_type`/`num_devices` knob),
  `scheduler.py` (`tier_loads = (tier, bytes, device_id)`), `trace_generator.py` (MEM_LOAD row).

---

## 4. CMX vs no-CMX — why the metrics improve

Same workload, two configs differing only by the `icms_mem` block. Causal chain:
1. `prefix_match` finds the reused context in the shallowest present pooled tier: **ICMS if
   configured, else COLDSTORE**.
2. The reload becomes a `MEM_LOAD` to that tier; ASTRA-Sim times it at
   `mem_latency + bytes/mem_bw + link` **plus any queueing** if the shared pool is busy.
   Per-reload: **ICMS ≈ 0.11–0.2 ms**, **COLDSTORE ≈ 2.35–3.3 ms** (~12–21×).
3. That reload sits **on the request's critical path** (data-dep parent of the compute).
4. Downstream:
   - **Reload-latency metric** ↓ (the per-reload analytical cost; contention-free reference).
   - **TTFT** ↓ (the reload precedes the first token; at scale, queueing adds to it).
   - **Throughput / total-clocks** improve when reloads are a large fraction of service
     time — and, crucially, **when many NPUs contend on the shared pool** (below).

### Refreshed results (contention-aware model, 2026-07-05)

| Experiment | CMX (ICMS) | noCMX (COLDSTORE) | CMX benefit | old (contention-free) |
|---|---|---|---|---|
| **Exp B** — 1 GPU, saturated burst (`cmx_burst_trace`) | 602.5 req/s, TTFT 26.1 ms | 505.9 req/s, TTFT 57.5 ms | **+19.1% tput, −54.6% TTFT** | +19%, −55% |
| **Exp C** — 72-GPU NVL72 rack, saturated (`cmx_sat72_trace`) | 19,790 req/s, 0.364 s | 2,698 req/s, 2.669 s | **~7.3× (+634%) tput** | +17.8% |

- **Exp B (1 GPU) is unchanged** — a single NPU has no pool to contend on, so it is pure
  formula parity with the old model.
- **Exp C (72 GPU) is dramatically larger than the old +17.8%.** The old Python-COMP model
  gave every GPU an *independent* reload (perfect parallelism), so falling through to
  COLDSTORE cost little. The contention-aware model serializes the 72 concurrent reloads on
  the **one shared** COLDSTORE pool, exposing it as the real bottleneck (2.5 s of serialized
  reloads). CMX's fast ICMS pool sidesteps it. This is the fidelity gain: at rack scale the
  benefit of a fast pooled tier is an order of magnitude, not ~18%.

**Where the true answer lies.** 7.3× is the *maximal-contention* extreme (`num_devices: 1`,
one shared FIFO channel). The no-contention extreme (`memory_type: PER_NPU_MEMORY_EXPANSION`,
or infinitely many channels) reproduces the old +17.8%. A real pool with *K* channels lands
in between — set it with each tier's `num_devices`.

**Contention proof (isolated A/B).** Same 72-GPU CMX workload, only the ICMS `memory_type`
differs: shared pool (`MEMORY_POOL`, `num_devices=1`) vs per-NPU
(`PER_NPU_MEMORY_EXPANSION`). Shared → **363.8M** total clocks / 19,790 req/s / TTFT fanning
out to 224 ms; per-NPU → **179.8M** / 40,051 req/s / flat ~35 ms. ~2× purely from
contention — a delta the old Python-COMP model structurally could not produce (its per-tier
reload metric is identical in both runs; contention shows up **only** in the sim clock).

---

## 5. Where the bottleneck is when you scale (the simulator, not the hardware)

**Model:** wall-clock ≈ Σ_iterations Σ_active-instances [ trace-gen + Chakra `.et` convert + ASTRA-Sim step ].
The Python main loop is **serial** over instances, so cost grows as **instances × iterations**.
(The deep tiers now going through ASTRA-Sim's memory model add negligible per-step cost — a
few extra `MEM_LOAD` nodes — and do not change this picture.)

**Measured instance scaling** (100-req burst, ~linear, ≈0.39 s/GPU):

| GPUs (racks) | 72 (1) | 144 (2) | 288 (4) | 576 (8) | 1152 (16) |
|---|---|---|---|---|---|
| wall time | 46 s | 82 s | 126 s | 225 s | 447 s |

**Profile (N=16 burst, cProfile cumulative)** — where the per-iteration time goes:
- **Chakra `.et` conversion** — `graph_generator.generate_graph` spawns the `chakra_converter`
  **subprocess per batch**; `posix.waitpid` ≈ 3.9 s. Process-spawn-per-batch is a top cost.
- **ASTRA-Sim IPC** — `controller.read_wait` reads ASTRA-Sim's stdout **line by line**
  (~358k `readline` calls, ≈ 3.4 s). Chatty stdio per iteration.
- **Trace generation** — cheap in steady state; a **one-time ~5.3 s** perf-DB / attention-table
  build (`_build_attention_table`) is a fixed startup cost, amortized at scale.

**So the bottleneck is the serial front-end orchestration** (trace file -> spawn Chakra
converter -> line-by-line ASTRA-Sim round-trip), **once per active instance per scheduling
step** — not the analytical network/memory math itself.

### How each scaling axis behaves
- **Instances / nodes ↑** → wall-clock grows ~linearly (serial per-instance loop; ASTRA-Sim
  topology also grows with total NPUs). This is the dominant axis.
- **Workload requests ↑** →
  - *Batched* (burst): little extra cost — bigger batches, ~same iteration count
    (288-req and 7,200-req NVL72 both ~48–50 s).
  - *Staggered / long decode / long prefills*: iteration count (and per-batch trace/graph
    size) grow → ~linear in iterations.
- **Fixed overhead** (perf-DB/attention-table build) is paid once, independent of scale.

### Practical guidance for the pod
- Don't run all 1,152 GPUs for iterative studies (linear cost + hours to saturate every GPU).
  **Simulate one saturated NVL72 rack and ×16** — inference racks are independent DP replicas
  (CMX pooled within a rack), so the pod aggregate is linear.
- If you need deeper scaling, the code-level levers are: **batch the ASTRA-Sim stdout reads**
  and **avoid spawning `chakra_converter` per batch** (persistent/in-process conversion).

---

## 6. Evolution: Python-injected → ASTRA-Sim-timed (why the numbers moved)

**v1 (original).** The deep tiers were modeled entirely in Python: `tier_load_latency_ns`
computed a scalar reload cost, injected as an **all-`LOCAL` COMP node** (`comp_time` = the
latency). This deliberately kept `memory_expansion.json`, the Chakra converter, and the
ASTRA-Sim binary **byte-identical** — zero C++/converter changes, easy to land. Its
limitation: ASTRA-Sim never saw the tiers as *memory*, so **no contention** — each GPU's
reload was independent, and a shared pool hit by 72 GPUs looked as cheap as one GPU.

**v2 (current).** The deep tiers are real ASTRA-Sim memory locations
(FLASH/ICMS/COLDSTORE), emitted into `memory_expansion.json` and reloaded via real
`MEM_LOAD` nodes (§3). ASTRA-Sim times them **and serializes concurrent reloads on the
shared pool**, so pooled-tier bandwidth contention is modeled. Cost: a small additive
C++/converter change and a recompile (`scripts/compile.sh`).

**Impact.** Single-GPU / uncontended results are unchanged (formula parity). Rack-scale
saturated results change substantially — the CMX-vs-noCMX benefit at 72 GPUs went from a
reported +17.8% to ~7.3× — because the shared COLDSTORE pool is now correctly a bottleneck.
The degree is tunable per tier via `num_devices` / `memory_type`.
