# #!/bin/bash

# --- [2026-07-08] TIER-POLICY: exclusive (cascading / write-back-on-eviction) ---------
#   NEW --tier-policy {inclusive,exclusive}. inclusive (default, unchanged): write-through
#   -- a cached prefix is copied into CPU AND every deeper tier at once. exclusive: a
#   prefix is written only to CPU and pushed ONE tier deeper ONLY when a tier evicts it,
#   so the deeper tiers stay EMPTY until the ones above fill (closer to real tiered-KV
#   systems; MEM_STORE write cost not modeled yet). Verified on example_trace (2n1g,
#   per_instance, COLDSTORE):
#     inclusive          : CPU 20 + JBOF 19 (32.5%); JBOF & COLD each 81.88 MB (eager copies)
#     exclusive (big CPU): CPU 20 only (16.7%); JBOF & COLD EMPTY (0 MB -- nothing spills)
#     exclusive (tiny caps -> vera_rubin_tray_2n1g_test.json): CASCADE CPU->JBOF->COLDSTORE,
#                          hits 13/17/9 (32.5%); COLDSTORE reload 4.41 ms.
#   The test config drastically shrinks CPU/JBOF (0.005/0.01 GiB) so the tiny example
#   overflows and the cascade is observable.
python -m serving \
  --cluster-config 'configs/cluster/vera_rubin_tray_2n1g_test.json' \
  --dtype float16 --block-size 16 \
  --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
  --cpu-scope per_instance --tier-policy exclusive \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/vera_rubin_tray_2n1g_test_exclusive.csv'

# --- CONFIG REFERENCE: VERA RUBIN TRAY (2 nodes x 2 trays x 2 superchips) --------------
#   superchip = 1 Vera CPU (1.5 TB LPDDR5X) + 2 Rubin GPUs (288 GB HBM4) = ONE TP=2
#   instance; tray = 2 superchips; node = 2 trays. vera_rubin_tray_{small,actual}.json are
#   2 nodes x 4 superchips = 16 GPUs. TP=2 (profiler has tp=1/tp=2 only; TP=2 == superchip).

# --- [2026-07-08] Fidelity check on vera_rubin_tray_2n1g.json (previous active) --------
#   Add "--tier-policy exclusive" to either run to switch from write-through to cascading.
# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_tray_2n1g.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#   --cpu-scope per_instance \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/vera_rubin_tray_2n1g_CPU.csv'
# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_tray_2n1g.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --cpu-scope per_instance \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/vera_rubin_tray_2n1g_COLDSTORE.csv'

# --- [2026-07-08] CPU-SCOPE demo: per-GPU CPU pools (--cpu-scope per_instance) (prev) --
#   per_node (default): one shared CPU pool per rack. per_instance: each GPU its own pool;
#   cross-GPU reuse routes to JBOF/COLDSTORE. On example_trace (2 GPUs): per_node -> CPU 39
#   (TTFT 14.40); per_instance -> CPU 20 + JBOF 19 (14.48). Omitting --cpu-scope == per_node.
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_multi_instance_tier.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --cpu-scope per_instance \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/prefix_cpu_pool_run_tier.csv'

# --- [2026-07-08] TIER-FIDELITY CHECK: NPU->CPU baseline vs full tier chain (prev) ----
#   Same small dataset (example_trace.jsonl), only --prefix-storage differs (CPU vs
#   COLDSTORE). Fixed the shared CPU prefix pool page_size (256 -> 1); afterwards both
#   runs report CPU 39 hits (32.5%), reload 0.02 ms, TTFT 14.40 ms (JBOF/COLDSTORE inert,
#   served from the shallowest tier = CPU). Fidelity restored.
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_multi_instance_tier.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/prefix_cpu_pool_run_small.csv'
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_multi_instance_tier.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/prefix_cpu_pool_run_tier.csv'


# ============================================================================
#  CLAUDE INVESTIGATION RUNS  (most recent first; only the TOP run is active,
#  older investigation runs are commented out per request)
# ============================================================================
#
# --- [2026-07-07c] PROGRESSION for the team doc: 8 -> 72(rack) -> 144(pod) GPU -------
#     Mini model + faithful Vera Rubin RATIO (bw/latency REAL, capacities scaled;
#     per-GPU NPU 3.9 / CPU 0.85 / JBOF 18.8 / COLD 188 GiB). SAME flags for WITH vs
#     WITHOUT JBOF -- the only difference is whether the config has a jbof_mem tier;
#     reuse is served from the shallowest tier that holds it. New cluster-wide TTFT/TBT
#     latency summary in serving/__main__.py. The gap GROWS once a rack shares the pool:
#       8  GPU (450 ctx) : JBOF 264.7 vs 114.7 req/s (2.3x); TTFT mean 341 vs 1471 ms; reload  392 vs  4836 ms
#       72 GPU (3200 ctx): JBOF 1104  vs 171   req/s (6.5x); TTFT p99 2.7s vs 33.8s (12.3x); reload 2.8s vs 34.4s
#      144 GPU (6000 ctx): JBOF 2142  vs 342   req/s (6.3x); TTFT p99 2.5s vs 31.2s;        reload 5.2s vs 63.8s
#     TBT identical WITH/WITHOUT at every scale -> tier changes PREFILL/TTFT, not decode.
#     TOKENS/s (JBOF vs no-JBOF): 1G 151k/111k; 8G 544k/236k; 72G 2.27M/0.35M; 144G 4.40M/0.70M.
#     NVIDIA claims "up to 5x higher tokens/s" for this tier -> faithful 72-GPU (1 rack) sim = 6.5x (brackets it).
#     Config bw/latency are REAL Vera Rubin (NPU 22000GB/s@10ns; CPU 1000@120ns; JBOF 64@80us + link 100@2us;
#     COLDSTORE 4@2ms + link 16@20us); only mem_size (capacity) is scaled down.
#     gen: make_pod_prop_mini_config.py <racks> <jbof|nojbof> <out> <gpr> 3.9 0.85 18.8 188
#          make_rack_fill.py <out.jsonl> <n_ctx> 2048 8 4 3000
#     Active = the simple 8-GPU JBOF first-run (the doc's opening example).


# python -m serving \
#   --cluster-config 'configs/cluster/pod_prop_mini_8gpu_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 \
#   --dataset 'workloads/pod_prop_8gpu.jsonl' \
#   --output 'outputs/pod_prop_8gpu_jbof.csv'




# WITHOUT JBOF: --cluster-config 'configs/cluster/pod_prop_mini_8gpu_nojbof.json' --output 'outputs/pod_prop_8gpu_nojbof.csv'
# 72-GPU rack: pod_prop_mini_72gpu_{jbof,nojbof}.json + workloads/pod_prop_72gpu.jsonl (3200 ctx)
#
# --- [2026-07-07b] 2-RACK (144-GPU: 2 nodes x 72) proportional JBOF vs non-JBOF ----
#     Same mini model + faithful Vera Rubin RATIO (NPU_KV:CPU:JBOF ~ 1:2.7:58), but
#     absolute caps scaled DOWN (per-GPU NPU 3.9 / CPU 0.85 / JBOF 18.8 / COLD 188 GiB)
#     so 144 GPUs reach JBOF with a tractable ~12k-request workload. JBOF+COLDSTORE are
#     ONE pod-wide shared pool across BOTH racks; CPU per-rack; NPU per-GPU. Needed the
#     kv_load/kv_evict trace rows to emit INTEGER byte counts (trace_generator.py) --
#     the mini model's fractional KV size overflowed the fixed-width column and merged
#     with the next field, crashing the Chakra converter.
#     gen: make_pod_prop_mini_config.py 2 <jbof|nojbof> <out> 72 3.9 0.85 18.8 188
#          make_rack_fill.py workloads/pod_prop_2rack.jsonl 6000 2048 8 4 3000
#     RESULT (144 GPU, 6000x2048-tok ctx): NPU 98.3% + CPU 99.9% saturate on BOTH racks; reuse
#     spills to the POD-WIDE shared deep tier (49.9% hit):
#       WITH JBOF : served from JBOF,      reload  5,164 ms, 2,142 req/s,  5.60 s
#       W/O  JBOF : falls to COLDSTORE,    reload 63,821 ms,   342 req/s, 35.07 s
#       => 12.4x faster reload, 6.3x throughput. Shared-pool contention GROWS the gap vs 1-GPU (1.73x).
# python -m serving \
#   --cluster-config 'configs/cluster/pod_prop_mini_2rack_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 \
#   --dataset 'workloads/pod_prop_2rack.jsonl' \
#   --output 'outputs/pod_prop_2rack_jbof.csv'
# WITHOUT JBOF: --cluster-config 'configs/cluster/pod_prop_mini_2rack_nojbof.json' --output 'outputs/pod_prop_2rack_nojbof.csv'
#
# --- [2026-07-07] PROPORTIONAL natural-spillover JBOF vs non-JBOF (mini model) ----
#     Chain NPU->CPU->JBOF->COLDSTORE (FLASH skipped -- JBOF/JBOF IS the flash tier).
#     Uses meta-llama/Llama-3.1-8B-mini4L: a COPY of Llama-3.1-8B with 4 layers
#     (~3.58 GiB weight) so an 8 GiB NPU keeps a healthy ~4.4 GiB KV headroom.
#     Tier caps preserve the real Vera Rubin ratio NPU_KV:CPU:JBOF ~= 1:2.75:58
#     (per-GPU NPU 8 / CPU 12 / JBOF 250 / COLD 2500 GiB); only capacities scaled,
#     bw/latency are real. Required a sim fix (memory_model.apply_kv_cache_events
#     now evicts unlocked LRU when NPU would overflow, instead of crashing --
#     snapshot of pre-fix state at ~/llmss_snapshot_pre_npu_fix; 72-GPU regression
#     re-run is bit-identical, so no impact on the earlier studies).
#     gen: make_pod_prop_mini_config.py <racks> <jbof|nojbof> <out> [gpr]; make_rack_fill.py
#     RESULT (1 GPU, 400x2048-tok ctx): working set NATURALLY fills NPU 99.7% +
#     CPU 98.2%, then spills the reused prefixes into the deep tier:
#       WITH JBOF : served from JBOF,      reload  360 ms, 137.4 req/s, 5.82 s
#       W/O  JBOF : falls to COLDSTORE,    reload 4600 ms,  79.5 req/s, 10.06 s
#       => 12.8x faster reload, 1.73x throughput (identical fill + hit tokens).
# python -m serving \
#   --cluster-config 'configs/cluster/pod_prop_mini_1gpu_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 \
#   --dataset 'workloads/mini_400.jsonl' \
#   --output 'outputs/mini_jbof_400.csv'
# WITHOUT JBOF (reuse falls to COLDSTORE): swap to the nojbof config:
# python -m serving --cluster-config 'configs/cluster/pod_prop_mini_1gpu_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 --dataset 'workloads/mini_400.jsonl' --output 'outputs/mini_nojbof_400.csv'
# Natural-spillover onset sweep (1 GPU): 100 ctx -> NPU only (JBOF 0); 400/700 -> spills to JBOF.
#
# --- [2026-07-06] TRUE 16-NODE SUPERPOD (pod-wide shared JBOF) — corrected model -
#     FIX: JBOF + COLDSTORE are now ONE POD-WIDE shared pool common to all racks
#     (a prefix cached by any rack is reusable by every rack), while NPU (per-GPU)
#     and CPU/FLASH (per-rack) stay local. The shared pool is reached via one
#     BlueField-4 channel PER RACK (num-devices=num_nodes) -> 16 parallel channels.
#     Code: serving/__main__.py (shared pool), scheduler.py + config_builder.py
#     (per-rack channels). Config gen: configs/cluster/make_pod_fillorder_config.py
#     <num_racks> <jbof|nojbof> <out> [gpus_per_rack]; workload make_rack_fill.py.
#     VERIFIED: (a) 2-node x1-GPU RR test -> cross-node reuse HITS shared JBOF
#     (512 tok) while that node's NPU/CPU/FLASH miss -> sharing proven.
#     (b) Linear scaling on REAL multi-node runs (both jbof & nojbof):
#         72 GPU : JBOF 632.9  / noJBOF 133.3 req/s  (latency 4.55s / 21.60s)
#        144 GPU : JBOF 1265.8 / noJBOF 266.6 req/s  = EXACTLY 2x; latency unchanged
#     => SuperPod 1,152 GPU (16 racks): CONFIRMED by the literal 16-node run --
#        JBOF 10,126.29 / noJBOF 2,133.08 req/s (4.75x); latency 4.55s vs 21.60s;
#        deep-tier reload JBOF 24.1s vs COLDSTORE 296.9s (12x); identical fill
#        order + 7,077,888 deep-tier hit tokens in both. Wall: JBOF 2h32m, noJBOF 1h40m.
#     Fill order (every rack): NPU 96% / CPU 84% / FLASH 95% -> shared JBOF 77%.
#     Active line = the true 16-node x 72 = 1,152-GPU JBOF run (~1hr; needs
#     --max-num-seqs 8). noJBOF + the 2-rack verification are commented below.
# python -m serving \
#   --cluster-config 'configs/cluster/pod_fillorder_16rack_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 \
#   --dataset 'workloads/pod_fill_16rack.jsonl' \
#   --output 'outputs/pod_fill_16rack_jbof.csv'
# 1,152-GPU noJBOF (reuse falls to shared pod-wide COLDSTORE):
# python -m serving --cluster-config 'configs/cluster/pod_fillorder_16rack_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 --dataset 'workloads/pod_fill_16rack.jsonl' --output 'outputs/pod_fill_16rack_nojbof.csv'
# 2-rack (144 GPU) verification (linear-scaling proof): make_pod_fillorder_config.py 2 jbof|nojbof <out> 72
# python -m serving --cluster-config 'configs/cluster/pod_fillorder_2rack_jbof.json' ... --dataset 'workloads/pod_fill_2rack.jsonl' ...
#
# --- [2026-07-05] SINGLE-RACK FILL-ORDER STUDY (superseded by the 16-node model) -
#     Goal: exercise the FULL hierarchy in order, then compare WITH-JBOF vs WITHOUT.
#     Config = one NVL72 rack with an ASCENDING, saturating capacity ladder (GiB):
#       NPU 16 (per-GPU) < CPU 2 < FLASH 6 < JBOF 120 < COLDSTORE 2000  (shared pools).
#       Latencies kept Vera-Rubin-faithful (HBM 10ns, LPDDR 120ns, local-SSD 10us,
#       JBOF 80us + 3us RDMA / 100 GB/s, storage 2ms). Caps scaled DOWN so a feasible
#       workload fills them (real 288GB HBM / 16TB JBOF can't be filled by any sim run).
#       gen: make_rack_fillorder_config.py 72 <jbof|nojbof>; workload make_rack_fill.py.
#       --max-num-seqs 8 keeps in-flight KV within the tiny HBM headroom (realistic).
#     Workload = 1,440 distinct 512-tok contexts (fill) + 1,440 reuses oldest-first (reload).
#     FILL ORDER (both): NPU 96% / CPU 84% / FLASH 95% filled; overflow -> JBOF 77% (jbof)
#       or COLDSTORE (nojbof).  Reuse served 20% from NPU (hot) + 30% from the deep tier.
#     RESULT (per 72-GPU rack):
#       reload cost : JBOF 1,504 ms   vs  COLDSTORE 18,556 ms   (12x)
#       throughput  : 632.9 req/s     vs  133.3 req/s           (4.75x)
#       mean TTFT   : 517 ms          vs  4,232 ms  (p99 1,468 vs 17,967 ms; ~8-12x)
#     SUPERPOD (16 racks = 1,152 GPUs): racks are independent DP replicas (JBOF pooled
#       per-rack), so throughput x16 -> JBOF ~10,126 vs noJBOF ~2,133 req/s; TTFT unchanged.
#     (single-rack; the 16-node pod model above supersedes this.)
# python -m serving \
#   --cluster-config 'configs/cluster/rack_fillorder_72gpu_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 \
#   --dataset 'workloads/rack_fill_trace.jsonl' \
#   --output 'outputs/rack_fill_72g_jbof.csv'
# 72 GPU noJBOF (reuse falls to shared COLDSTORE):
# python -m serving --cluster-config 'configs/cluster/rack_fillorder_72gpu_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --max-num-seqs 8 --dataset 'workloads/rack_fill_trace.jsonl' --output 'outputs/rack_fill_72g_nojbof.csv'
# 8 GPU sanity (fast; same fill-order signature): rack_fillorder_8gpu_jbof.json + rack_fill_small.jsonl
#
# --- [2026-07-05] VERA RUBIN JBOF STUDY (full-capacity, saturating burst) -------
#     Realistic caps (NPU 288GB/22TBps, CPU 768GB, JBOF 16TB, COLDSTORE 128TB); tiers
#     do NOT fill (too big) -- uses page-granularity reuse instead. RESULTS:
#       1 GPU : JBOF 602.5 vs noJBOF 505.9 req/s (+19%); TTFT 26.1 vs 57.5 ms (-55%)
#       72 GPU: JBOF 19,790 vs noJBOF 2,698 req/s (7.3x); TTFT 126 vs 1294 ms (~10x)
# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_72gpu_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' \
#   --output 'outputs/study_72g_jbof.csv'
# 72 GPU noJBOF:
# python -m serving --cluster-config 'configs/cluster/vera_rubin_72gpu_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' --output 'outputs/study_72g_nojbof.csv'
# 1 GPU JBOF / noJBOF (isolates per-reload latency; jbof_burst_trace):
# python -m serving --cluster-config 'configs/cluster/vera_rubin_jbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/study_1g_jbof.csv'
# python -m serving --cluster-config 'configs/cluster/vera_rubin_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/study_1g_nojbof.csv'
#
# --- [2026-07-05] Execution-FLOW TRACE demo (understand the program flow) -----
#     --trace-flow writes a plain-English step-by-step log of every wrapper function
#     (does/input/output) to outputs/flow_trace.log. --trace-flow-full = exhaustive.
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_single_instance.json' \
#   --dtype float16 --block-size 16 \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/trace_demo.csv' \
#   --num-reqs 2 --trace-flow
#
# --- [2026-07-05] Headline JBOF experiments REFRESHED with the contention-aware
#     model (deep tiers now timed by ASTRA-Sim; JBOF/COLDSTORE = shared pools).
#     Exp B (1 GPU, jbof_burst) -> ~UNCHANGED (single NPU = no contention):
#        JBOF 602.5 req/s, TTFT 26.1ms   vs   noJBOF 505.9 req/s, TTFT 57.5ms
#        => +19.1% throughput, -54.6% TTFT   (matches old contention-free model)
#     Exp C (72 GPU rack, jbof_sat72) -> JBOF benefit is MUCH larger than the old
#        +17.8%: 72 NPUs now contend on ONE shared pool, and COLDSTORE
#        serialization is punishing:
#        JBOF 19,790 req/s (0.364s)   vs   noJBOF 2,698 req/s (2.669s)
#        => ~7.3x throughput (+634%).   [old contention-free model reported +17.8%]
#     Contention degree is tunable per tier via num_devices / memory_type.
# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_72gpu_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' \
#   --output 'outputs/vr72sat_jbof.csv'
# Exp C noJBOF (72 GPU): reuse falls to the shared COLDSTORE pool -> heavy contention.
# python -m serving --cluster-config 'configs/cluster/vera_rubin_72gpu_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' --output 'outputs/vr72sat_nojbof.csv'
# Exp B JBOF / noJBOF (1 GPU burst):
# python -m serving --cluster-config 'configs/cluster/vera_rubin_jbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/vr_jbof_burst.csv'
# python -m serving --cluster-config 'configs/cluster/vera_rubin_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/vr_nojbof_burst.csv'
#
# --- [2026-07-05] Deep-tier CONTENTION proof: pooled vs per-NPU JBOF (ran) ----
#     Same 72-GPU jbof workload; only JBOF memory-type differs. Pooled (shared,
#     MEMORY_POOL num-devices=1) serializes all 72 NPUs; per-NPU
#     (PER_NPU_MEMORY_EXPANSION) runs them in parallel. RESULT: pooled 363.8M
#     clocks / 19,790 req/s / TTFT ladder to 224ms  vs  per-NPU 179.8M / 40,051
#     req/s / flat ~35ms  => contention ~2x. (old Python COMP model: no delta.)
# python -m serving --cluster-config 'configs/cluster/vera_rubin_72gpu_jbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' --output 'outputs/tier_contention_pooled.csv'
# python -m serving --cluster-config 'configs/cluster/vera_rubin_72gpu_jbof_pernpu.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' --output 'outputs/tier_contention_pernpu.csv'
#
# --- [2026-07-05] Deep-tier fidelity: routing + regression (WITH JBOF) -------
#     WHY: verified deep tiers route through ASTRA-Sim (real MEM_LOAD nodes);
#     memory_expansion.json gained flash/jbof/coldstore blocks; trace showed an
#     `jbof_load` row with weight_loc=JBOF:0. Kept for reference.
# python -m serving \
#   --cluster-config 'configs/cluster/cmp_with_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/tier_demo_trace.jsonl' \
#   --output 'outputs/tier_fidelity_with_jbof.csv'
#
# --- [2026-07-05] CXL fidelity check (superseded) ---------------------------
#     WHY: showed CXL is an ASTRA-Sim-timed tier (cxl_mem in memory_expansion.json)
#     vs the (then Python-only) deep tiers. Kept for reference.
# python -m serving --cluster-config 'configs/cluster/single_node_cxl_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/cxl_fidelity_check.csv' \
#     --num-req 10
# ============================================================================

# Single instance example (prefix caching in xPU memory is default now)
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_single_run.csv' \
#     --num-req 10

# Prefix cache with CPU Prefix Cache Pool example (multi-instance, single node)
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/prefix_cpu_pool_run.csv'

# Multi-tier prefix-cache spillover example (NPU -> CPU -> FLASH -> JBOF -> COLDSTORE)
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_tiered_instance.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/example_trace.jsonl' \
#   --output 'outputs/tiered_spillover_run.csv'

# Multi-tier spillover with a SHARED-PREFIX workload (exercises hits + write-through)
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_tiered_instance.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/shared_prefix_trace.jsonl' \
#   --output 'outputs/tiered_shared_prefix_run.csv'

# Multi-tier spillover DEMO: small caps + phased access evict prefix "A" down the
# chain, then reuse it -> served from COLDSTORE with Python-injected reload latency
# (a `coldstore_load` COMP node ~142 ms). Takes ~2-3 min (small NPU, long prefills).
# python -m serving \
#   --cluster-config 'configs/cluster/single_node_tiered_small.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/tier_demo_trace.jsonl' \
#   --output 'outputs/tiered_demo_run.csv'

# Tier-comparison: tiers are config-driven (a tier is used iff its block exists).
# WITH JBOF -> reused prefix served from the fast JBOF tier.
# python -m serving \
#   --cluster-config 'configs/cluster/cmp_with_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/tier_demo_trace.jsonl' \
#   --output 'outputs/cmp_with_jbof.csv'

# =====================================================================================
#  VERA RUBIN NVL72 EXPERIMENTS
#  Hierarchy: NPU=Rubin HBM4 -> CPU=Vera LPDDR5X -> JBOF=JBOF/JBOFP (Ethernet NVMe flash)
#             -> COLDSTORE=network storage.  The *_nojbof configs simply omit jbof_mem,
#             so the same reuse falls through to COLDSTORE (NVIDIA's "JBOF vs storage").
#  Config generator : python configs/cluster/make_vera_rubin_config.py <num_gpus> <jbof|nojbof> <out.json>
#  Workload generator: python workloads/generators/make_jbof_demo.py <out> <n_reuse> <a_len> <tail> <out_len> <ms> <stagger|burst>
#  All runs also write the CLI results to outputs/<name>.txt.
# -------------------------------------------------------------------------------------

# --- Exp 0: SMALL example that SATURATES the tiers (used to EXPLAIN the code). 1 GPU,
#     Vera Rubin latencies, small caps -> NPU ~96% / CPU ~75% full; reused context A served
#     from JBOF (~0.11 ms) with JBOF vs COLDSTORE (~2.35 ms) without.
#     [commented out 2026-07-05 — superseded by the CXL fidelity check at top]
# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_small_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/vr_sat_trace.jsonl' \
#   --output 'outputs/vr_small_jbof.csv'

# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_small_nojbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/vr_sat_trace.jsonl' \
#   --output 'outputs/vr_small_nojbof.csv'

# --- Exp A: 1 GPU, per-access demo (staggered) -> shows reload latency (~us vs ~ms) & TTFT
# python -m serving --cluster-config 'configs/cluster/vera_rubin_jbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_demo_trace.jsonl' --output 'outputs/vr_jbof.csv'
# python -m serving --cluster-config 'configs/cluster/vera_rubin_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_demo_trace.jsonl' --output 'outputs/vr_nojbof.csv'

# --- Exp B: 1 GPU, SATURATED burst (100 reuses) -> shows throughput & TTFT.
#     [contention-aware model, 2026-07-05: +19.1% throughput, -54.6% TTFT --
#      unchanged from the old model since a single NPU has no pool contention]
# python -m serving --cluster-config 'configs/cluster/vera_rubin_jbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/vr_jbof_burst.csv'
# python -m serving --cluster-config 'configs/cluster/vera_rubin_nojbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/vr_nojbof_burst.csv'

# --- Exp C: FULL NVL72 rack (72 GPUs), SATURATED ~100 reuses/GPU -> rack-level JBOF benefit.
#     [contention-aware model, 2026-07-05: JBOF 19,790 vs noJBOF 2,698 req/s => ~7.3x
#      (+634%) throughput; total latency 0.364s vs 2.669s. The OLD contention-free
#      Python-COMP model reported +17.8% -- it gave every GPU an independent reload;
#      the new model serializes the 72 concurrent reloads on the shared COLDSTORE
#      pool, exposing it as the real bottleneck. Tune with tier num_devices.]
#      x16 for the 16-rack superpod.
# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_72gpu_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' \
#   --output 'outputs/vr72sat_jbof.csv'

# python -m serving \
#   --cluster-config 'configs/cluster/vera_rubin_72gpu_nojbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_sat72_trace.jsonl' \
#   --output 'outputs/vr72sat_nojbof.csv'

# --- Scaling probe: 1..16 NVL72 racks (JBOF). Configs: vera_rubin_{72,144,288,576,1152}gpu_jbof.json
#     (generate a matching _nojbof with make_vera_rubin_config.py for a comparison at any scale).
# python -m serving --cluster-config 'configs/cluster/vera_rubin_1152gpu_jbof.json' \
#   --dtype float16 --block-size 16 --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/jbof_burst_trace.jsonl' --output 'outputs/vr_1152gpu.csv'

# WITHOUT JBOF (jbof_mem absent from config) -> same reuse falls to slow COLDSTORE.
# python -m serving \
#   --cluster-config 'configs/cluster/cmp_no_jbof.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/tier_demo_trace.jsonl' \
#   --output 'outputs/cmp_no_jbof.csv'

# Multi instance example
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_multi_run.csv' \
#     --num-req 10

# # PD example
# python -m serving --cluster-config 'configs/cluster/single_node_pd_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pd_run.csv' \
#     --num-req 10

# # CXL example
# python -m serving --cluster-config 'configs/cluster/single_node_cxl_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_cxl_run.csv' \
#     --num-req 10

# # Prefix cache with CPU Prefix Cache Pool example (Single Node)
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#      --dtype float16 --block-size 16 \
#     --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_prefix_cpu_mem_pool_run.csv' \
#     --num-req 10

# # Prefix cache with CPU Prefix Cache Pool example (Dual Node)
# python -m serving --cluster-config 'configs/cluster/dual_node_multi_instance.json' \
#     --dtype float16 --block-size 16 \
#     --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_dual_prefix_cpu_mem_pool_run.csv' \
#     --num-req 10

# # Power model example
# python -m serving --cluster-config 'configs/cluster/single_node_power_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_power_run.csv' \
#     --num-req 10 --log-interval 0.1

# # PIM example
# python -m serving --cluster-config 'configs/cluster/single_node_pim_instance.json' \
#     --dtype float16 --block-size 16 --enable-attn-offloading \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pim_run.csv' \
#     --num-req 10 --log-level WARNING

# # Sub-batch interleaving example
# python -m serving --cluster-config 'configs/cluster/single_node_pim_instance.json' \
#     --dtype float16 --block-size 16 --enable-attn-offloading --enable-sub-batch-interleaving \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pim_sub_batch_run.csv' \
#     --num-req 10 --log-level WARNING



# MoE example
# python -m serving --cluster-config 'configs/cluster/single_node_moe_single_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_run.csv' \
#     --num-req 10

# MoE DP+EP with agentic session example (SWE-bench)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
#     --dtype float16 --block-size 16 \
#     --dataset 'workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl' --output 'outputs/example_moe_dp_ep_run.csv' \
#     --num-req 1 # session count in agentic workload


# -----------------------------------------------------------------------------------------------
#    Deprecated examples (may not be up to date with the latest codebase, kept for reference)
# -----------------------------------------------------------------------------------------------


# Prefix caching example (disabled: prefix caching in xPU memory is default now)
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dtype float16 --block-size 16 --enable-prefix-caching \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_prefix_run.csv' \
#     --num-req 10

# NS-3 example
# Note: NS-3 integration is currently a work in progress. The following command is a placeholder and may not work until the NS-3 integration is complete.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dtype float16 --block-size 16 --network-backend 'ns3' \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_ns3_run.csv' \
#     --num-req 10 