"""Simulation entry point: ``python -m serving --cluster-config <...> [...]``.

Parses CLI args, generates ASTRA-Sim input files via ``serving.core.config_builder``,
spawns the ASTRA-Sim subprocess, and runs the iteration loop:
``router.route -> scheduler.schedule -> trace_generator -> graph -> ASTRA-Sim
-> scheduler.add_done`` until every request completes.
"""

import os
import subprocess
import argparse
import json
import numpy as np
from time import time
from collections import defaultdict

from serving.core.scheduler import *
from serving.core.request import *
from serving.core.utils import *
from serving.core.controller import *
from serving.core.memory_model import *
from serving.core.graph_generator import *
from serving.core.trace_generator import *
from serving.core.pim_model import *
from serving.core.config_builder import *
from serving.core.router import *
from serving.core.power_model import *
from serving.core.logger import *
from serving.core import dashboard
from serving.core.run_paths import build_run_paths, resolve_run_id
import sys as flush

from pyinstrument import Profiler


def _pad_batch_to_max(batch, max_len):
    """Pad a batch up to ``max_len`` for DP-sync.

    Mirrors vLLM's CUDA-graph DP padding: every DP rank's forward runs at
    ``max(num_tokens_across_dp)``. We bump the high-level counters so
    dense layers, lm_head, and the MoE compute path all reflect the
    padded shape — but we deliberately leave ``decode_k_list`` /
    prefill lists untouched so attention continues to see only the real
    decodes. FlashAttention's varlen kernel gives padded ``seq_len=0``
    entries zero compute in real vLLM, and extending ``decode_k_list``
    with ``kv=1`` dummies would instead collapse ``kv_decode_mean``
    toward 1 and push the attention lookup far outside the profiled
    sweep.

    MoE AG/RS comm size is anchored separately to ``max_total_len`` (no
    ``× group_size``) in the iteration loop — that calibrates the
    bandwidth model against the same ``link_bw`` AllReduce already uses.

    Request-completion accounting (`scheduler.add_done`) reads
    ``batch.requests`` and ``batch.end``, not these mutated token-list
    fields, so it is unaffected.
    """
    pad = max_len - batch.total_len
    if pad <= 0:
        return
    batch.total_len = max_len
    batch.kv_len += pad                  # each dummy contributes kv=1
    batch.num_decode += pad              # counted for lm_head / dense shape


def _runtime_limit(value):
    return float('inf') if value == 0 else value


def _cluster_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join("..", path)


def _load_cluster_config_for_overrides(path):
    with open(_cluster_config_path(path), "r") as f:
        return json.load(f)


def _resolve_output_file(path, run_id):
    if path is None:
        return None
    return path.replace("{run_id}", run_id)


def _prepare_ns3_config(astra_sim, run_paths):
    template = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
    output_dir = os.path.join(run_paths.inputs_root, "ns3", "output")
    config_path = os.path.join(run_paths.inputs_root, "ns3", "config.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    replacements = {
        "FLOW_FILE": os.path.join(output_dir, "flow.txt"),
        "TRACE_FILE": os.path.join(output_dir, "trace.txt"),
        "TRACE_OUTPUT_FILE": os.path.join(output_dir, "mix.tr"),
        "FCT_OUTPUT_FILE": os.path.join(output_dir, "fct.txt"),
        "PFC_OUTPUT_FILE": os.path.join(output_dir, "pfc.txt"),
        "QLEN_MON_FILE": os.path.join(output_dir, "qlen.txt"),
    }

    for path in (replacements["FLOW_FILE"], replacements["TRACE_FILE"]):
        open(path, "w").close()

    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in replacements:
                f.write(f"{parts[0]} {replacements[parts[0]]}\n")
            else:
                f.write(line)
    return config_path


def _iter_raw_instances(cluster_config):
    for node in cluster_config.get("nodes", []):
        for instance in node.get("instances", []):
            yield instance


def _resolve_instance_dtype(instance, cli_dtype, dtype_to_bits):
    dtype = instance.get("dtype", cli_dtype)
    if dtype is None:
        config = get_config(instance["model_name"])
        torch_dtype = config.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype in dtype_to_bits:
            dtype = torch_dtype
        else:
            dtype = "bfloat16"
    if dtype not in dtype_to_bits:
        raise ValueError(f"Unsupported dtype '{dtype}' for instance {instance.get('instance_id')}")
    return dtype


def _build_instance_runtime_configs(instances, args, dtype_to_bits):
    runtime_configs = []
    for instance_id, instance in enumerate(instances):
        dtype = _resolve_instance_dtype(instance, args.dtype, dtype_to_bits)
        kv_cache_dtype = instance.get("kv_cache_dtype", args.kv_cache_dtype)
        if kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(f"Unsupported kv_cache_dtype '{kv_cache_dtype}' for instance {instance_id}")

        enable_attn_offloading = instance.get("enable_attn_offloading", args.enable_attn_offloading)
        enable_sub_batch_interleaving = instance.get(
            "enable_sub_batch_interleaving", args.enable_sub_batch_interleaving)
        if enable_sub_batch_interleaving and not enable_attn_offloading:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving without attention offloading")

        runtime_configs.append({
            "max_num_seqs": _runtime_limit(instance.get("max_num_seqs", args.max_num_seqs)),
            "max_num_batched_tokens": _runtime_limit(
                instance.get("max_num_batched_tokens", args.max_num_batched_tokens)),
            "long_prefill_token_threshold": instance.get(
                "long_prefill_token_threshold", args.long_prefill_token_threshold),
            "block_size": instance.get("block_size", args.block_size),
            "dtype": dtype,
            "fp": dtype_to_bits[dtype],
            "kv_cache_dtype": kv_cache_dtype,
            "enable_chunked_prefill": instance.get(
                "enable_chunked_prefill", args.enable_chunked_prefill),
            "enable_prefix_caching": instance.get(
                "enable_prefix_caching", args.enable_prefix_caching),
            "prioritize_prefill": instance.get("prioritize_prefill", args.prioritize_prefill),
            "enable_local_offloading": instance.get(
                "enable_local_offloading", args.enable_local_offloading),
            "enable_attn_offloading": enable_attn_offloading,
            "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
            "enable_block_copy": instance.get("enable_block_copy", args.enable_block_copy),
        })
    return runtime_configs


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(prog='python -m serving',
                                     description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, default='configs/cluster/single_node_single_instance.json',
                        help='path to cluster config JSON defining node topology, instance layout, hardware, and memory hierarchy')
    parser.add_argument('--max-num-seqs', type=int, default=128,
                        help='maximum number of sequences in a batch (0 = unlimited)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                        help='maximum number of tokens processed per iteration across all requests (the total token budget). '
                        'With chunked prefill, long inputs are split across iterations; '
                        'without chunked prefill, this effectively caps max input length')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0,
                        help='per-request token cap per step for chunked prefill (0 = disabled). '
                        'Limits how many tokens a single prefill request consumes per iteration, '
                        'preventing long prompts from monopolizing the token budget. '
                        'When 0, a single prefill can consume the entire budget')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32', 'fp8', 'int8'], default=None,
                        help='model weight data type (vLLM-style). When omitted, defaults to the model config\'s '
                        '``torch_dtype`` (falling back to bfloat16). Overrides only take effect if the profiler '
                        'produced matching data under perf/<hw>/<model>/<variant>/tp<N>/')
    parser.add_argument('--request-routing-policy', type=str, choices=['LOAD', 'RR', 'RAND', 'CUSTOM'], default='LOAD',
                        help='request routing policy across instances: LOAD (vLLM-style weighted least-loaded, default), '
                        'RR (round-robin), RAND (random), CUSTOM (user-defined)')
    parser.add_argument('--expert-routing-policy', type=str,
                        choices=['BALANCED', 'RR', 'RAND', 'CUSTOM'],
                        default='BALANCED',
                        help='expert token routing policy for MoE models: '
                        'BALANCED (default; analytical pigeonhole approximation of '
                        'a trained load-balanced learned gate), '
                        'RR (round-robin), RAND (uniform random per token), '
                        'CUSTOM (user-defined)')
    parser.add_argument('--enable-block-copy', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Replay one transformer block\'s trace across every '
                        'layer instead of re-computing the routing per layer — '
                        'cuts trace-generation time roughly num_hidden_layers× '
                        'on MoE models. Safe with BALANCED (deterministic); '
                        'RR/RAND get a small per-layer variance averaged out. '
                        'Disable only for CUSTOM policies that need faithful '
                        'per-layer variance.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True,
                        help='enable prefix caching via RadixAttention to reuse KV cache across requests '
                        'with shared prefixes (default: enabled). Use --no-enable-prefix-caching to disable')
    parser.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True,
                        help='enable chunked prefill to split long prefill requests across multiple iterations, '
                        'matching vLLM v1 behavior (default: enabled). Use --no-enable-chunked-prefill to disable')
    parser.add_argument('--enable-prefix-sharing', action='store_true', default=False,
                        help='enable second-tier prefix cache pooling across instances within a node')
    parser.add_argument('--prefix-storage', type=str,
                        choices=['None', 'CPU', 'CXL', 'FLASH', 'JBOF', 'COLDSTORE'], default='None',
                        help='deepest prefix-cache spill tier. None = NPU only; CXL = legacy standalone '
                        'second tier. CPU/FLASH/JBOF/COLDSTORE select the depth of the '
                        'NPU->CPU->FLASH->JBOF->COLDSTORE chain: the named tier and every shallower tier '
                        'are used, deeper tiers are unused (spill past the limit is dropped). '
                        'Requires --enable-prefix-sharing for FLASH/JBOF/COLDSTORE.')
    parser.add_argument('--cpu-scope', type=str, choices=['per_node', 'per_instance'], default='per_node',
                        help="scope of the CPU (host-DRAM) prefix-cache tier when --enable-prefix-sharing "
                        "is on. 'per_node' (default, unchanged): ONE CPU pool per node, shared by all its "
                        "GPUs -- a request on any GPU can reuse another GPU's offloaded KV directly from "
                        "CPU; the pool holds the node total (sum of the GPUs' cpu_mem). 'per_instance': each "
                        "GPU gets its OWN private CPU pool, sized at the GPU's own cpu_mem if the instance "
                        "block declares one, else the node's cpu_mem split evenly among its GPUs (node total "
                        "unchanged). Models real per-GPU host memory (e.g. Vera Rubin's 1 CPU : 2 GPU "
                        "NVLink-C2C LPDDR); cross-GPU reuse can then NOT be served from CPU and instead "
                        "routes to the pod-wide JBOF/COLDSTORE tiers. Declare per-instance `cpu_mem` in an "
                        "instance block (like `npu_mem`) to give GPUs different host-memory sizes.")
    parser.add_argument('--tier-policy', type=str, choices=['inclusive', 'exclusive'], default='inclusive',
                        help="how a cached prefix occupies the below-NPU storage tiers (CPU/FLASH/JBOF/"
                        "COLDSTORE). 'inclusive' (default, unchanged): write-through -- a copy is written "
                        "into CPU AND every deeper tier at once. 'exclusive': cascading / write-back-on-"
                        "eviction -- a prefix is written only to CPU (the shallowest storage tier) and pushed "
                        "one level deeper ONLY when a tier evicts it, so it lives in exactly one storage tier "
                        "and deeper tiers stay empty until the ones above fill. Closer to real tiered-KV "
                        "systems; MEM_STORE write cost is not modeled either way (yet).")
    parser.add_argument('--dashboard', action='store_true', default=False,
                        help='emit a live-metrics JSON snapshot every log interval (and at the end) for '
                        'the Streamlit dashboard. Off by default (zero overhead). See --dashboard-file.')
    parser.add_argument('--dashboard-file', type=str, default='outputs/dashboard/live.json',
                        help='where the live-metrics JSON is written when --dashboard is set '
                        '(relative paths resolve from the repo root). Default outputs/dashboard/live.json.')
    parser.add_argument('--session-metrics', action='store_true', default=False,
                        help='for multi-turn / agentic-session workloads (records with "sub_requests"): '
                        'split the TTFT summary into first-turn (cold) vs resumed-turn (warm, context '
                        'reloaded from a lower tier). Exposes the pause/resume "context parking" benefit — '
                        'e.g. JBOF vs COLDSTORE reload on turn resume. Off by default; no effect otherwise.')
    parser.add_argument('--enable-local-offloading', action='store_true', default=False,
                        help='enable weight offloading to local (NPU) memory. '
                        'Recommended to disable unless weight memory access is not counted in profiling')
    parser.add_argument('--enable-attn-offloading', action='store_true', default=False,
                        help='enable attention computation offloading to PIM (Processing-In-Memory) devices')
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', default=False,
                        help='enable sub-batch interleaving to overlap XPU and PIM computation. '
                        'Requires --enable-attn-offloading')
    parser.add_argument('--prioritize-prefill', action='store_true', default=False,
                        help='prioritize prefill requests over decode requests in scheduling')
    parser.add_argument('--block-size', type=int, default=16,
                        help='KV cache block size in tokens (number of tokens per block)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to .jsonl dataset file with request traces. '
                        'If None, requests must be added manually in serving/__main__.py')
    parser.add_argument('--output', type=str, default=None,
                        help='path for per-request CSV output with latency metrics (TTFT, TPOT, ITL). '
                        'If None, results are printed to stdout only. Supports {run_id} placeholder')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique id for this simulation run. Intermediate ASTRA-Sim inputs are written under '
                        'astra-sim/inputs/runs/<run-id>. If omitted, a process-unique id is generated')
    parser.add_argument('--inputs-root', type=str, default=None,
                        help='override the root directory for generated ASTRA-Sim inputs. Defaults to '
                        'astra-sim/inputs/runs/<run-id>')
    parser.add_argument('--skip-prefill', action='store_true', default=False,
                        help='skip the prefill phase, running decode only')
    parser.add_argument('--num-reqs', type=int, default=0,
                        help='number of entries (requests or sessions) to load from the dataset. '
                        'For agentic datasets, each entry is a session with multiple sub-requests. '
                        '0 = load all entries')
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='interval in seconds between throughput/memory usage log messages')
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], default='WARNING',
                        help='logging verbosity: WARNING (minimal), INFO (per-iteration details), DEBUG (per-layer memory)')
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8'], default='auto',
                        help='KV cache data type: auto (use default profile.csv) or fp8 (use profile_fp8.csv, halves KV cache memory)')
    parser.add_argument('--network-backend', type=str, choices=['analytical', 'ns3'], default='analytical',
                        help='network simulation backend: analytical (fast, default) or ns3 (detailed, WIP)')

    parser.add_argument('--trace-flow', nargs='?', const='outputs/flow_trace.log', default=None,
                        help='Write a human-readable execution-flow log (every wrapper function call, in '
                             'order, with its layer/file, one-line description, inputs and output) to this '
                             'file, overwritten each run. If the flag is given with no value, defaults to '
                             'outputs/flow_trace.log. For readability, use a tiny workload (e.g. --num-reqs 2).')
    parser.add_argument('--trace-flow-full', action='store_true', default=False,
                        help='With --trace-flow: log EVERY call/return (exhaustive, large file) instead of the '
                             'default compact view that shows each function once + a call-count inventory.')
    args = parser.parse_args()

    # Optional execution-flow trace (purely for understanding the program flow).
    # Installed as early as possible so the whole pipeline below is captured.
    # NOTE: relative paths resolve against the repo root (the wrapper chdirs into
    # astra-sim/ during a run), so outputs/flow_trace.log lands in the repo outputs/.
    if getattr(args, 'trace_flow', None):
        from serving.core.flow_tracer import install as _install_flow_trace
        _install_flow_trace(args.trace_flow, full=args.trace_flow_full)

    args.run_id = resolve_run_id(args.run_id)
    run_paths = build_run_paths(astra_sim, args.run_id, args.inputs_root)
    args.inputs_root = run_paths.inputs_root
    args.output = _resolve_output_file(args.output, args.run_id)

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    print_banner()
    print_input_config(args=args)
    print_markup("[sim.heading]▶ Starting simulation...[/]\n")
    flush.stdout.flush()
    
    _dtype_to_bits = {'float16': 16, 'bfloat16': 16, 'float32': 32, 'fp8': 8, 'int8': 8}
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    cpu_scope=args.cpu_scope
    tier_policy=args.tier_policy
    dataset=args.dataset
    output_file=args.output
    is_init = not args.skip_prefill
    num_req=args.num_reqs
    log_interval=args.log_interval
    network_backend = args.network_backend
    raw_cluster_config = _load_cluster_config_for_overrides(args.cluster_config)
    raw_instances = list(_iter_raw_instances(raw_cluster_config))
    build_enable_local_offloading = args.enable_local_offloading or any(
        inst.get("enable_local_offloading", False) for inst in raw_instances)
    build_enable_attn_offloading = args.enable_attn_offloading or any(
        inst.get("enable_attn_offloading", False) for inst in raw_instances)
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(
        astra_sim, args.cluster_config, build_enable_local_offloading, build_enable_attn_offloading,
        inputs_root=run_paths.inputs_root)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    cpu_mem_bw = cluster["cpu_mem_bw"]
    cpu_mem_latency = cluster["cpu_mem_latency"]
    inst_cpu_mem_size = cluster["inst_cpu_mem_size"]          # per-instance override or None
    inst_cpu_mem_bw = cluster["inst_cpu_mem_bw"]
    inst_cpu_mem_latency = cluster["inst_cpu_mem_latency"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    instance_runtime_configs = _build_instance_runtime_configs(instances, args, _dtype_to_bits)
    any_prefix_caching = any(cfg["enable_prefix_caching"] for cfg in instance_runtime_configs)
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=run_paths.network_config
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'ns3':
        network=_prepare_ns3_config(astra_sim, run_paths)
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    memory=run_paths.memory_config
    system=run_paths.system_config
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    # ---- Per-instance CPU (host-DRAM) resolution for the prefix-cache tier ----
    # `--cpu-scope` + optional per-instance `cpu_mem` decide CPU-pool sizing:
    #   sched_cpu_*   -> value handed to each scheduler/MemoryModel (drives the
    #                    NON-shared offload path & the reload metric): the declared
    #                    per-instance value, else the node's cpu_mem (UNCHANGED default).
    #   pool_cpu_size -> the prefix-cache POOL capacity per instance: declared, else
    #                    the node's cpu_mem SPLIT evenly among its GPUs (node total
    #                    unchanged). per_node sums these into the one shared pool.
    def _node_gpus(nid):
        return len(node2inst_mapping[nid])
    def _node_cpu_declared(nid):
        return any(inst_cpu_mem_size[j] is not None for j in node2inst_mapping[nid])
    sched_cpu_size = [inst_cpu_mem_size[i] if inst_cpu_mem_size[i] is not None
                      else cpu_mem_size[inst2node_mapping[i]] for i in range(num_instances)]
    sched_cpu_bw = [inst_cpu_mem_bw[i] if inst_cpu_mem_bw[i] is not None
                    else cpu_mem_bw[inst2node_mapping[i]] for i in range(num_instances)]
    sched_cpu_latency = [inst_cpu_mem_latency[i] if inst_cpu_mem_latency[i] is not None
                         else cpu_mem_latency[inst2node_mapping[i]] for i in range(num_instances)]
    pool_cpu_size = [inst_cpu_mem_size[i] if inst_cpu_mem_size[i] is not None
                     else cpu_mem_size[inst2node_mapping[i]] / _node_gpus(inst2node_mapping[i])
                     for i in range(num_instances)]

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    # Tier membership is CONFIG-DRIVEN: a deep tier is used iff its block is
    # present in the cluster config (any subset works — e.g. drop `jbof_mem` to
    # skip JBOF). --prefix-storage still caps the depth (and enables the chain);
    # the active deep tiers are those PRESENT in the config, in canonical order,
    # up to and including the named cap. CXL stays a standalone legacy tier.
    _TIER_CHAIN = ['CPU', 'FLASH', 'JBOF', 'COLDSTORE']
    _NAME_TO_DEVICE = {'FLASH': Device.FLASH, 'JBOF': Device.JBOF, 'COLDSTORE': Device.COLDSTORE}
    _tier_cfgs = cluster.get("tier_configs", {})

    def _tier_in_config(name):
        c = _tier_cfgs.get(name)
        return bool(c) and len(c["size"]) == num_nodes and all(s and s > 0 for s in c["size"])

    if prefix_storage in ('CPU', 'FLASH', 'JBOF', 'COLDSTORE'):
        storage_medium = 'CPU'
        cap_idx = _TIER_CHAIN.index(prefix_storage)
        deep_tier_names = [t for t in _TIER_CHAIN[1:cap_idx + 1] if _tier_in_config(t)]
        skipped = [t for t in _TIER_CHAIN[1:cap_idx + 1] if not _tier_in_config(t)]
        active_chain = "NPU -> CPU" + "".join(f" -> {t}" for t in deep_tier_names)
        print_markup(f"Active prefix-cache tiers: {active_chain}"
                     + (f"   (skipped, absent from config: {', '.join(skipped)})" if skipped else ""))
    elif prefix_storage == 'CXL':
        storage_medium = 'CXL'
        deep_tier_names = []
    else:
        storage_medium = 'None'
        deep_tier_names = []

    if deep_tier_names and not (any_prefix_caching and enable_prefix_sharing):
        raise RuntimeError(
            f"--prefix-storage {prefix_storage} (multi-tier spillover) requires "
            f"--enable-prefix-caching and --enable-prefix-sharing")

    pool_device = None
    if storage_medium == "CPU":
        pool_device = Device.CPU
    elif storage_medium == "CXL":
        pool_device = Device.CXL

    # Deep-tier shared pools (per node) + per-instance specs handed to schedulers.
    deep_tier_pools = {}                                   # tier_name -> [pool per node]
    deep_tier_specs_by_inst = {i: [] for i in range(num_instances)}

    if any_prefix_caching and enable_prefix_sharing and storage_medium != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []

        def _pool_kv_bytes_per_token(inst_ids):
            """KV bytes per token for a shared pool."""
            kv_shapes = {
                (
                    instances[i]["model_name"],
                    instance_runtime_configs[i]["fp"],
                    instance_runtime_configs[i]["kv_cache_dtype"],
                )
                for i in inst_ids
            }
            if len(kv_shapes) > 1:
                raise RuntimeError(
                    "Shared prefix pool requires instances to share model, "
                    f"dtype, and kv_cache_dtype; got {kv_shapes}"
                )
            model = instances[inst_ids[0]]['model_name']
            cfg = instance_runtime_configs[inst_ids[0]]
            return full_cluster_kv_bytes_per_token(model, cfg["fp"], cfg["kv_cache_dtype"])

        if storage_medium == 'CPU':
            # page_size=1 (token granularity) for the CPU pool in BOTH scopes, like
            # the CXL pool, the deep tiers, and the non-shared CPU second-tier
            # (memory_model.py). A larger page floors every insert AND match to a
            # multiple of it, so any prefix shorter than one page is silently
            # dropped -> the NPU->CPU baseline would retain/serve nothing.
            if cpu_scope == 'per_instance':
                # PER-INSTANCE (per-GPU) CPU: each GPU gets its OWN private CPU pool,
                # modeling real per-GPU host memory (e.g. Vera Rubin's 1 CPU : 2 GPU
                # NVLink-C2C LPDDR). Capacity = the instance's own `cpu_mem` if it
                # declares one, else the node's cpu_mem SPLIT evenly among its GPUs
                # (so the node total is unchanged; see pool_cpu_size). Because the
                # pools are private, cross-GPU prefix reuse CANNOT be served from CPU
                # and falls through to the pod-wide JBOF/COLDSTORE tiers -- faithful.
                for inst_id in range(num_instances):
                    if pool_cpu_size[inst_id] <= 0:
                        raise RuntimeError(f"Memory size for prefix storage type {storage_medium} is invalid")
                    prefix_pools.append(RadixCache(
                                                node_id=0,
                                                device=storage_medium,
                                                page_size=1,
                                                capacity=pool_cpu_size[inst_id] * GB_TO_BYTE,
                                                kv_size=_pool_kv_bytes_per_token([inst_id]),
                                                enable_kv_cache_events=True))
                # Each instance maps to its own private pool.
                prefix_pool_inst_mapping = {i: i for i in range(num_instances)}
            else:
                # PER-NODE (default): ONE CPU pool per node, shared by all its GPUs.
                # Capacity = the sum of the node's per-instance cpu_mem when any GPU
                # declares its own (else the node-level cpu_mem, byte-exact default).
                for i in range(num_prefix_pool):
                    _node_cap = (sum(pool_cpu_size[j] for j in node2inst_mapping[i])
                                 if _node_cpu_declared(i) else cpu_mem_size[i])
                    if _node_cap > 0:
                        new_prefix_pool = RadixCache(
                                                    node_id=0,
                                                    device=storage_medium,
                                                    page_size=1,
                                                    capacity = _node_cap * GB_TO_BYTE,
                                                    kv_size=_pool_kv_bytes_per_token(node2inst_mapping[i]),
                                                    enable_kv_cache_events=True)
                        prefix_pools.append(new_prefix_pool)
                    else:
                        raise RuntimeError(f"Memory size for prefix storage type {storage_medium} is invalid")
                # This means one node shares one prefix pool
                prefix_pool_inst_mapping = inst2node_mapping

            # Build the deeper spill-tier pools. FLASH (local SSD) is PER-NODE
            # (per-rack). JBOF (JBOF) and COLDSTORE (network storage) are ONE
            # POD-WIDE shared pool common to ALL racks: a prefix cached by any
            # rack is reusable by every rack, because the physical JBOF/storage
            # tier is pooled behind BlueField-4 and reachable pod-wide over
            # Spectrum-X. Pod-wide capacity = sum of the per-node contributions
            # declared in the config (e.g. 16 racks x 120 GiB = 1920 GiB JBOF).
            _POD_WIDE_DEEP_TIERS = {"JBOF", "COLDSTORE"}
            tier_cfgs = cluster.get("tier_configs", {})
            for tname in deep_tier_names:
                tcfg = tier_cfgs.get(tname)
                if tcfg is None or len(tcfg["size"]) < num_prefix_pool:
                    raise RuntimeError(
                        f"--prefix-storage {prefix_storage} requires '{tname.lower()}_mem' "
                        f"to be configured in the cluster config")
                for i in range(num_prefix_pool):
                    if tcfg["size"][i] <= 0:
                        raise RuntimeError(
                            f"--prefix-storage {prefix_storage} requires '{tname.lower()}_mem' "
                            f"(mem_size > 0) on node {i}")
                if tname in _POD_WIDE_DEEP_TIERS:
                    # ONE shared pool for the whole pod; every node maps to it.
                    total_cap = int(sum(tcfg["size"][i] for i in range(num_prefix_pool)) * GB_TO_BYTE)
                    shared = RadixCache(
                        node_id=0,
                        device=tname,
                        page_size=1,   # token granularity, like the CXL pool
                        capacity=total_cap,
                        kv_size=_pool_kv_bytes_per_token(list(range(num_instances))),
                        enable_kv_cache_events=False)
                    deep_tier_pools[tname] = [shared] * num_prefix_pool
                else:
                    pools = []
                    for i in range(num_prefix_pool):
                        pools.append(RadixCache(
                            node_id=i,
                            device=tname,
                            page_size=1,   # token granularity, like the CXL pool
                            capacity=int(tcfg["size"][i] * GB_TO_BYTE),
                            kv_size=_pool_kv_bytes_per_token(node2inst_mapping[i]),
                            enable_kv_cache_events=False))
                    deep_tier_pools[tname] = pools

            # Reporting capacity: pod-wide total for shared tiers, per-node for FLASH.
            _deep_tier_total_gb = {
                t: (sum(tier_cfgs[t]["size"][i] for i in range(num_prefix_pool))
                    if t in _POD_WIDE_DEEP_TIERS else None)
                for t in deep_tier_names}

            for inst_id in range(num_instances):
                node = inst2node_mapping[inst_id]
                specs = []
                for tname in deep_tier_names:
                    tcfg = tier_cfgs[tname]
                    _cap_gb = (_deep_tier_total_gb[tname]
                               if tname in _POD_WIDE_DEEP_TIERS else tcfg["size"][node])
                    specs.append({
                        'device': _NAME_TO_DEVICE[tname], 'name': tname,
                        'cache': deep_tier_pools[tname][node],
                        'capacity_gb': _cap_gb,
                        'mem_bw': tcfg["mem_bw"][node], 'mem_latency': tcfg["mem_latency"][node],
                        'link_bw': tcfg["link_bw"][node], 'link_latency': tcfg["link_latency"][node],
                    })
                deep_tier_specs_by_inst[inst_id] = specs

        elif storage_medium == 'CXL':
            if cluster["cxl_mem_size"] > 0:
                new_prefix_pool = RadixCache(
                                            node_id=None,
                                            device=storage_medium,
                                            page_size=1,
                                            capacity = cluster["cxl_mem_size"] * GB_TO_BYTE,
                                            kv_size=_pool_kv_bytes_per_token(list(range(num_instances))),
                                            enable_kv_cache_events=True)
                prefix_pools.append(new_prefix_pool)
                # This means every instance shares the same universal prefix pool (maybe fixed later)
                prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
            else:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance

        inst_cfg = instance_runtime_configs[instance_id]

        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id,
            inst_cfg["max_num_seqs"], inst_cfg["max_num_batched_tokens"],
            instance["num_npus"], instance["tp_size"], instance["pp_size"],
            instance["npu_mem"]["mem_size"], sched_cpu_size[instance_id],
            inst2npu_mapping[instance_id], instance["pd_type"],
            inst_cfg["fp"], inst_cfg["block_size"], num_req,
            inst_cfg["prioritize_prefill"], inst_cfg["enable_prefix_caching"],
            enable_prefix_sharing, prefix_pool, pool_device, inst_cfg["enable_chunked_prefill"],
            inst_cfg["long_prefill_token_threshold"],
            cxl_mem,
            ep_size=instance.get("ep_total", 1),
            kv_cache_dtype=inst_cfg["kv_cache_dtype"],
            deep_tiers=deep_tier_specs_by_inst.get(instance_id, []),
            cpu_mem_bw=sched_cpu_bw[instance_id],
            cpu_mem_latency=sched_cpu_latency[instance_id],
            tier_policy=tier_policy,
        ))

    # Controller for astra-sim process communication
    controller = Controller(total_npu)
    # Global Request Router
    router = Router(num_instances, schedulers, num_req, request_routing_policy)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None
    # Load requests into router (routed in real-time during simulation)
    if dataset != None:
        router.load_requests(dataset, enable_prefix_caching=any_prefix_caching, is_init=is_init)
    else:
        # Manually adding request (legacy: route all upfront)
        for i in range(16):
            for sched in schedulers:
                sched.add_request([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    # intervals-per-second = 1/log_interval; scales per-interval token counts to
    # tokens/sec. MUST be float division: integer // gave 0 for any log_interval > 1
    # (e.g. --log-interval 2), zeroing all throughput and crashing on 1/RATIO below.
    RATIO = FREQ/INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # `args` is reused as a list inside the loop (subprocess cmd), so capture the
    # session-metrics flag up front (same reason as dash_enabled below).
    session_metrics_enabled = bool(args.session_metrics)

    # -------------------- Live dashboard (--dashboard) --------------------
    dash_enabled = bool(args.dashboard)          # NOTE: `args` is reused as a list
    dash_file = args.dashboard_file              # inside the loop (subprocess cmd),
    dash_history = []                            # so capture the flags up front.
    dash_write_path = None
    dash_config = None
    if dash_enabled:
        # The sim chdirs into astra-sim/ during a run, so a relative output path
        # needs the same "../" hop the CSV/txt outputs use to land at repo root.
        _dp = args.dashboard_file
        dash_write_path = _dp if os.path.isabs(_dp) else f'../{_dp}'
        # Per-tier bw/latency spec (for the dashboard config panel). NPU from the
        # first instance, CPU from node 0, deep tiers from the first configured node.
        _tier_specs = []
        try:
            _npu0 = instances[0].get("npu_mem", {})
            _tier_specs.append({"name": "NPU", "scope": "per-GPU",
                                "mem_bw": _npu0.get("mem_bw"), "mem_latency": _npu0.get("mem_latency")})
            _tier_specs.append({"name": "CPU", "scope": cpu_scope.replace("_", "-"),
                                "mem_bw": cpu_mem_bw[0] if cpu_mem_bw else None,
                                "mem_latency": cpu_mem_latency[0] if cpu_mem_latency else None})
            for _tn in deep_tier_names:
                _tc = _tier_cfgs.get(_tn, {})
                _n = next((i for i, s in enumerate(_tc.get("size", [])) if s and s > 0), 0)
                _tier_specs.append({
                    "name": _tn,
                    "scope": "per-node" if _tn == "FLASH" else "pooled",
                    "mem_bw": _tc.get("mem_bw", [None])[_n], "mem_latency": _tc.get("mem_latency", [None])[_n],
                    "link_bw": _tc.get("link_bw", [None])[_n], "link_latency": _tc.get("link_latency", [None])[_n],
                })
        except Exception:
            _tier_specs = []
        dash_config = {
            "models": sorted({inst["model_name"] for inst in instances}),
            "num_nodes": num_nodes, "num_instances": num_instances, "total_npu": total_npu,
            "tp_sizes": sorted({inst["tp_size"] for inst in instances}),
            "cpu_scope": cpu_scope, "tier_policy": tier_policy,
            "prefix_storage": str(prefix_storage),
            "block_size": args.block_size, "dtype": args.dtype,
            "cluster_config": os.path.basename(args.cluster_config),
            "dataset": os.path.basename(dataset) if dataset else None,
            "tier_specs": _tier_specs,
        }
        # Initial snapshot so the dashboard shows the config before the first interval.
        try:
            dashboard.write_snapshot(dash_write_path, dashboard.build_snapshot(
                "starting", clock_ns=0, freq=FREQ, wall_seconds=0.0, config=dash_config,
                cpu_scope=cpu_scope, req_cnt=0, total_prompt=0, total_gen=0,
                requests_total=len(getattr(router, "_pending_requests", [])),
                schedulers=schedulers, num_nodes=num_nodes, num_instances=num_instances))
        except Exception:
            pass

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = router.get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time), inputs_root=run_paths.inputs_root)
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True, inputs_root=run_paths.inputs_root)
    # set first workload file
    workload = get_workload(None, None, event=True, inputs_root=run_paths.inputs_root)
    # run subprocess
    args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # DP group synchronization: defer trace generation until all members have scheduled
    # dp_groups maps dp_group_name -> list of instance_ids
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst["instance_id"])
    # Reverse lookup: instance_id -> dp_group_name
    inst_dp_group = {}
    for dg, members in dp_groups.items():
        for inst_id in members:
            inst_dp_group[inst_id] = dg
    # Pending batches per DP group (waiting for all members to schedule)
    dp_pending = {dg: {} for dg in dp_groups}  # dp_group -> {instance_id: (new_req, sys)}
    # Pre-generated workloads ready to submit on next "Waiting"
    dp_ready_workloads = {}  # instance_id -> workload_path

    # ----------------------------------- Start simulation loop ------------------------------------
    # Starting simulation, one while loop processes one iteration
    while True:
        
        out = controller.read_wait(p)
        out_dict = controller.parse_output(out[-2])
        
        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']

        # Route newly arrived requests to instances based on current load
        if dataset is not None:
            router.route_arrived_requests(current)

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current,
                        last_end_time[instance_id], last_calc_time[instance_id], num_npus=instances[instance_id]["num_npus"])
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        # An instance can span multiple NPUs. Only update end-time when sys is the first NPU of the instance.
        # waiting_request[instance_id] = True means the instance has no batch to run (idle).
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        prompt_t, gen_t, finished_reqs = schedulers[instance_id].add_done(id, sys, current)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(finished_reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Notify router of completed requests for dependency chain release
        if instances[instance_id]["pd_type"] != "prefill":
            for req in finished_reqs:
                router.notify_request_completed(req.id, current)

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(finished_reqs) > 0:
            router.transfer_prefill_request(finished_reqs)

        # schedule requests
        new_req = schedulers[instance_id].schedule(current, sys, id)
        responded = False  # track whether we already sent a response to ASTRA-Sim

        # Check if a pre-generated workload is ready for this instance (from DP sync)
        if new_req is None and instance_id in dp_ready_workloads:
            controller.write_flush(p, dp_ready_workloads.pop(instance_id))
            responded = True
        # DP group: truly idle instance (no inflight batch) — create dummy batch so ALLTOALL syncs
        elif new_req is None and instance_id in inst_dp_group and sys == inst2npu_mapping[instance_id] and len(schedulers[instance_id].inflight) == 0:
            dg = inst_dp_group[instance_id]
            if dp_pending[dg]:
                # Emit a 1-token dummy; the uniform pad-to-max pass below
                # brings it (and any undersized real peers) up to the
                # group's max_total_len, matching vLLM's CUDA-graph DP padding.
                logger.debug(f"Instance {instance_id} is idle but DP group {dg} has pending batches. Creating dummy batch for synchronization.")
                dummy = Batch(schedulers[instance_id].get_batch_id(), instances[instance_id]["model_name"],
                              1, 1, [1], [], 0, 1, [], [], [1], current, 0)
                dummy.fired.append(sys)
                dp_pending[dg][instance_id] = (dummy, inst2node_mapping[instance_id])

                if len(dp_pending[dg]) == len(dp_groups[dg]):
                    # All DP members accounted for — pad every batch to the
                    # group's max (vLLM CUDA-graph DP padding) and generate.
                    config = get_config(instances[instance_id]["model_name"])
                    max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                    for b, _ in dp_pending[dg].values():
                        _pad_batch_to_max(b, max_total_len)
                    # MoE AG/RS comm size is anchored to ``max_total_len``
                    # (not ``max × group_size``). The trace generator divides
                    # this by ep_total internally for the per-rank AG chunk
                    # and uses the same value for the RS pre-scatter buffer.
                    # Empirically this matches real NCCL AG/RS bandwidth on
                    # PCIe 5.0 at the same ``link_bw`` that already calibrates
                    # AllReduce — i.e. ASTRA-Sim's Ring half-duplex model
                    # ends up correct for AR but 2× over real AG/RS, and the
                    # "× group_size" we used previously stacked the two errors.
                    sum_total_len = max_total_len

                    # Shared workload folder for all DP members
                    first_inst_id = dp_groups[dg][0]
                    first_batch = dp_pending[dg][first_inst_id][0]
                    dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                    for inst_id in dp_groups[dg]:
                        batch, nid = dp_pending[dg][inst_id]
                        inst = instances[inst_id]
                        inst_cfg = instance_runtime_configs[inst_id]
                        generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                       inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                       nid, inst_id,
                                       inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                       placement[inst_id], block_mode_on[inst_id],
                                       expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                       inst_cfg["enable_attn_offloading"],
                                       power_model, pim_models[nid],
                                       inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                       dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                       tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                       dp_sum_total_len=sum_total_len,
                                       enable_block_copy=inst_cfg["enable_block_copy"],
                                       inputs_root=run_paths.inputs_root)
                        generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                       inst_id, inst2npu_mapping[inst_id],
                                       inst_cfg["enable_local_offloading"],
                                       workload_name=dp_workload_name,
                                       inputs_root=run_paths.inputs_root)
                        if inst_id != instance_id:
                            dp_ready_workloads[inst_id] = get_workload(batch, inst["hardware"], inst_id,
                                                                    workload_name=dp_workload_name,
                                                                    inputs_root=run_paths.inputs_root)

                    dp_pending[dg].clear()
                    workload = get_workload(dummy, instances[instance_id]["hardware"], instance_id,
                                            workload_name=dp_workload_name,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
                    responded = True
                else:
                    controller.write_flush(p, "pass")
                    responded = True
        # runnable batch exists
        elif new_req is not None:
            if sys == inst2npu_mapping[instance_id]:  # first NPU of the instance
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                dg = inst_dp_group.get(instance_id)

                if dg is not None:
                    # DP group: defer trace generation until all members scheduled
                    dp_pending[dg][instance_id] = (new_req, node_id)

                    if len(dp_pending[dg]) == len(dp_groups[dg]):
                        # All DP members have scheduled — pad every batch to
                        # the group's max (vLLM CUDA-graph DP padding) so
                        # smaller batches gain dummy decodes that all layers
                        # still compute over.
                        config = get_config(instance["model_name"])
                        max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                        for b, _ in dp_pending[dg].values():
                            _pad_batch_to_max(b, max_total_len)
                        # See twin block above: anchor MoE comm to max_total_len
                        # (no group-size multiplier).
                        sum_total_len = max_total_len

                        # Shared workload folder for all DP members
                        first_inst_id = dp_groups[dg][0]
                        first_batch = dp_pending[dg][first_inst_id][0]
                        dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                        for inst_id in dp_groups[dg]:
                            batch, nid = dp_pending[dg][inst_id]
                            inst = instances[inst_id]
                            inst_cfg = instance_runtime_configs[inst_id]
                            generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                           inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                           nid, inst_id,
                                           inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                           placement[inst_id], block_mode_on[inst_id],
                                           expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                           inst_cfg["enable_attn_offloading"],
                                           power_model, pim_models[nid],
                                           inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                           dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                           tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                           dp_sum_total_len=sum_total_len,
                                           enable_block_copy=inst_cfg["enable_block_copy"],
                                           inputs_root=run_paths.inputs_root)
                            generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                           inst_id, inst2npu_mapping[inst_id],
                                           inst_cfg["enable_local_offloading"],
                                           workload_name=dp_workload_name,
                                           inputs_root=run_paths.inputs_root)
                            if inst_id != instance_id:
                                dp_ready_workloads[inst_id] = get_workload(batch, inst["hardware"], inst_id,
                                                                        workload_name=dp_workload_name,
                                                                        inputs_root=run_paths.inputs_root)

                        dp_pending[dg].clear()
                        workload = get_workload(new_req, instance["hardware"], instance_id,
                                                workload_name=dp_workload_name,
                                                inputs_root=run_paths.inputs_root)
                        controller.write_flush(p, workload)
                    else:
                        # Waiting for other DP members — send pass
                        controller.write_flush(p, "pass")
                        responded = True
                else:
                    # Independent instance: generate trace immediately
                    inst_cfg = instance_runtime_configs[instance_id]
                    generate_trace(new_req, instance["hardware"], instance["tp_size"], instance["pp_size"],
                                   instance["local_ep"], instance["ep_total"],
                                   instance["pd_type"],
                                   node_id, instance_id,
                                   inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                   placement[instance_id], block_mode_on[instance_id],
                                   expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                   inst_cfg["enable_attn_offloading"], power_model, pim_models[node_id],
                                   inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                   dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                   tp_dim=instance["tp_dim"], ep_dim=instance["ep_dim"],
                                   enable_block_copy=inst_cfg["enable_block_copy"],
                                   inputs_root=run_paths.inputs_root)
                    generate_graph(new_req, instance["hardware"], instance["num_npus"], node_id,
                                   instance_id, inst2npu_mapping[instance_id],
                                   inst_cfg["enable_local_offloading"],
                                   inputs_root=run_paths.inputs_root)
                    workload = get_workload(new_req, instance["hardware"], instance_id,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
            elif new_req is not None:
                # Non-first NPU: pick up existing batch workload
                workload = get_workload(new_req, instances[instance_id]["hardware"], instance_id,
                                        inputs_root=run_paths.inputs_root)
                controller.write_flush(p, workload)

        # check time to store throughput (only print on start NPU to avoid transient states)
        if current > last_log + INTERVAL and sys == inst2npu_mapping[instance_id]:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            if dash_enabled:
                try:
                    # build_snapshot appends this interval's timeline point
                    # (throughput + per-tier memory + TTFT/TBT) to dash_history.
                    dashboard.write_snapshot(dash_write_path, dashboard.build_snapshot(
                        "running", clock_ns=current, freq=FREQ, wall_seconds=time() - start_time,
                        config=dash_config, cpu_scope=cpu_scope, req_cnt=req_cnt,
                        total_prompt=total_prompt, total_gen=total_gen,
                        requests_total=len(getattr(router, "_pending_requests", [])),
                        schedulers=schedulers, num_nodes=num_nodes, num_instances=num_instances,
                        history=dash_history, t_s=last_log / FREQ, record_point=True,
                        live_prompt_tps=prompt_th * RATIO, live_gen_tps=gen_th * RATIO))
                except Exception:
                    pass
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            # Heartbeat timestamp stays in the terminal's default
            # colour — bright enough to scan, not so dim that it
            # disappears. (The per-log-record [HH:MM:SS.mmm] stays
            # dim via sim.time because it appears every other line.)
            print_markup(
                f"{log_time_str} "
                f"[blue]Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,[/] "
                f"[blue]Avg generation throughput: {gen_th * RATIO:.1f} tokens/s[/]"
            )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                running_reqs = sum(len(batch.requests) for batch in schedulers[inst_id].inflight)
                waiting_reqs = len([req for req in schedulers[inst_id].request if req.arrival <= current])

                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_util = (mem.npu_used / mem.npu_mem * 100.0) if mem.npu_mem else 0.0

                line = (
                    f"{log_indent+tree_indent}Running Instance\\[{inst_id}]: "
                    f"{running_reqs} reqs, Waiting: {waiting_reqs} reqs, "
                    f"Total # {schedulers[inst_id].num_npus} NPUs, "
                    f"Each NPU Memory Usage {npu_used_mb:.2f} MB "
                    f"({npu_util:.3f} % Used)"
                )
                if schedulers[inst_id].enable_prefix_caching:
                    line += schedulers[inst_id].memory.npu_prefix_cache.format_prefix_info()
                print_markup(line)

            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    inst_usage = []
                    if any_prefix_caching and enable_prefix_sharing and storage_medium == "CPU":
                        if cpu_scope == 'per_instance':
                            # prefix_pools is indexed by GLOBAL instance_id in this
                            # scope -> sum the node's own instances' private pools.
                            node_cpu_usage = sum(prefix_pools[j].total_size() * prefix_pools[j].kv_size
                                                 for j in inst_ids)
                        else:
                            node_cpu_usage = prefix_pools[node_id].total_size() * prefix_pools[node_id].kv_size
                    else:
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (node_cpu_usage / (cpu_mem_size[node_id]*GB_TO_BYTE)) * 100
                    has_deep = bool(deep_tier_names)
                    if storage_medium != "CXL" and not has_deep and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    line = (
                        f"{log_indent+tree_indent}Node\\[{node_id}]: "
                        f"Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, "
                        f"{cpu_util:.3f} % Used "
                    )
                    if any_prefix_caching and enable_prefix_sharing and storage_medium == "CPU":
                        if cpu_scope == 'per_instance':
                            line += "".join(prefix_pools[j].format_prefix_info() for j in inst_ids)
                        else:
                            line += prefix_pools[node_id].format_prefix_info()

                    if (any_prefix_caching and enable_prefix_sharing and storage_medium == "CPU") or (len(inst_ids) == 1):
                        print_markup(line)
                    else:
                        parts = []
                        for j, inst_cpu_usage in enumerate(inst_usage):
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            parts.append(f"Instance\\[{inst_ids[j]}]: {inst_cpu_util:.2f} %")
                        print_markup(line + "(" + ", ".join(parts) + ")")

                    # Deep spill tiers (FLASH/JBOF/COLDSTORE) usage for this node.
                    for ti, tname in enumerate(deep_tier_names):
                        pools = deep_tier_pools.get(tname, [])
                        if node_id >= len(pools):
                            continue
                        dp = pools[node_id]
                        dused = dp.total_size() * dp.kv_size
                        dutil = (dused / dp.capacity * 100) if dp.capacity else 0
                        if (not power_modeling and i == num_nodes - 1
                                and ti == len(deep_tier_names) - 1):
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}Node\\[{node_id}] {tname}: "
                            f"Usage {dused/MB_TO_BYTE:.2f} MB, {dutil:.3f} % Used"
                        )

            ######### Per CXL Metrics #########
            if any_prefix_caching and storage_medium == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = cxl_pool.total_size() * cxl_pool.kv_size
                        cxl_util = cxl_usage / cxl_pool.capacity
                        if not power_modeling and cxl_id == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[{cxl_id}]: "
                            f"Total CXL Device Memory Usage "
                            f"{cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used"
                        )
                else:
                    enabled_inst_ids = [
                        inst_id for inst_id, sched in enumerate(schedulers)
                        if sched.enable_prefix_caching
                    ]
                    for pos, inst_id in enumerate(enabled_inst_ids):
                        second_tier = getattr(
                            schedulers[inst_id].memory, "second_tier_prefix_cache", None)
                        if second_tier is None:
                            continue
                        cxl_usage = second_tier.total_size() * second_tier.kv_size
                        cxl_util = cxl_usage / second_tier.capacity
                        if not power_modeling and pos == len(enabled_inst_ids) - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[0]/Instance\\[{inst_id}]: "
                            f"Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, "
                            f"{cxl_util:.3f} % Used"
                        )

            ######### Power Modeling #########
            if power_modeling:
                tree_indent = '└─'
                print_markup(
                    f"{log_indent+tree_indent}"
                    f"Avg power consumption: {power_model.get_current_power(current)} W"
                )
        # check if all requests are done for current instance#
        # NOTE: 'instance_id' could occur in duplicate, because 'npu2inst_mapping[sys]' is not one-to-one mapping
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty() and not router.has_pending_requests() and not router.has_deferred_sessions():
            # For DP groups: only mark done when ALL members of the group are empty
            dg = inst_dp_group.get(instance_id)
            if dg is not None:
                all_dp_empty = all(
                    schedulers[inst_id].is_request_empty() and len(schedulers[inst_id].inflight) == 0
                    for inst_id in dp_groups[dg]
                )
                if not all_dp_empty:
                    # Other DP members still have work — keep this instance alive for dummy waves
                    if not responded:
                        controller.write_flush(p, "pass")
                    flush.stdout.flush()
                    continue

            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)
            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["num_npus"] == 1 else 2):
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()
                
                # check memory leak before exit
                schedulers[inst_idx].memory.is_free()

                print_rule()
                print_markup("[sim.heading]▶ Exiting simulation...[/]\n")
                controller.write_flush(p, "exit")
                break
            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None and not responded:
            # If all instances are idle but deferred sessions have pending
            # requests with future arrival times (tool calls still running),
            # advance current time so the next iteration can pick them up.
            if router.has_deferred_sessions() or router.has_pending_requests():
                next_arrival = router.get_next_pending_arrival()
                if next_arrival is not None and next_arrival > current:
                    current = next_arrival
            controller.write_flush(p, "pass")
        
        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    # calculate prefix caching metrics (per tier: NPU -> CPU/CXL -> deep tiers)
    tier_requested_tokens = 0
    tier_hit_tokens = {}          # Device -> hit tokens (summed across instances)
    tier_reload_ns = {}           # Device -> analytical reload latency (ns), summed
    tier_reload_bytes = {}        # Device -> reloaded bytes, summed
    if any_prefix_caching:
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            req_i, hits_i = schedulers[i].memory.tier_hit_report()
            tier_requested_tokens += req_i
            for dev, h in hits_i.items():
                tier_hit_tokens[dev] = tier_hit_tokens.get(dev, 0) + h
            rl_i, rb_i = schedulers[i].memory.reload_report()
            for dev, v in rl_i.items():
                tier_reload_ns[dev] = tier_reload_ns.get(dev, 0) + v
            for dev, v in rb_i.items():
                tier_reload_bytes[dev] = tier_reload_bytes.get(dev, 0) + v

    def _tier_usage_bytes():
        """Per-Device (used_bytes, capacity_bytes), pool-aware (dedup shared pools)."""
        usage = {}
        # NPU: per-instance, summed.
        nu = nc = 0
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            npc = schedulers[i].memory.npu_prefix_cache
            nu += npc.total_memory_usage(); nc += npc.capacity
        usage[Device.NPU] = (nu, nc)
        # CPU / CXL second tier.
        if pool_device is not None:
            if enable_prefix_sharing:
                u = sum(p.total_memory_usage() for p in prefix_pools)
                c = sum(p.capacity for p in prefix_pools)
            else:
                u = c = 0
                for i in range(num_instances):
                    st = getattr(schedulers[i].memory, "second_tier_prefix_cache", None)
                    if st is not None:
                        u += st.total_memory_usage(); c += st.capacity
            usage[pool_device] = (u, c)
        # Deep tiers. FLASH is per-node (distinct pools -> sum all); JBOF/COLDSTORE
        # are ONE pod-wide pool referenced once per node -> dedup by object id so
        # the shared pool is counted exactly once.
        for tname in deep_tier_names:
            pools = deep_tier_pools.get(tname, [])
            uniq = []                       # dedup by identity (builtin id() is
            for p in pools:                 # shadowed by a local 'id' in main())
                if all(p is not q for q in uniq):
                    uniq.append(p)
            usage[_NAME_TO_DEVICE[tname]] = (
                sum(p.total_memory_usage() for p in uniq),
                sum(p.capacity for p in uniq))
        return usage

    # This is total system's throughput
    total_latency = current/FREQ
    print_rule()
    print_markup("[sim.heading]▶ Simulation results...[/]\n")
    print_markup(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print_rule("[sim.tagline]Throughput Results[/]")
    print_markup(f"Total requests:                                                     {req_cnt}")
    print_markup(f"Total clocks (ns):                                                  {current}")
    print_markup(f"Total latency (s):                                                  {total_latency:.3f}")
    print_markup(f"Total input tokens:                                                 {total_prompt}")
    print_markup(f"Total generated tokens:                                             {total_gen}")
    print_markup(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print_markup(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print_markup(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print_markup(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print_markup(f"Throughput per {1/RATIO} sec (\\[prompt_throughput], \\[gen_throughput]): {throughput}")
    print_rule()
    if any_prefix_caching:
        print_rule("[sim.tagline]Prefix Caching Results (per tier)[/]")
        print_markup(f"Total requested prompt tokens:                                      {tier_requested_tokens}")
        # Ordered tier chain: NPU -> CPU/CXL -> deep tiers (depth-limited).
        ordered = [Device.NPU]
        if pool_device is not None:
            ordered.append(pool_device)
        for tname in deep_tier_names:
            ordered.append(_NAME_TO_DEVICE[tname])
        usage = _tier_usage_bytes()
        if tier_requested_tokens > 0:
            total_hit = 0
            print_markup(f"{'Tier':<10}{'Hit tokens':>11}{'Hit %':>9}{'Reload(ms)':>12}{'Usage(MB)':>14}{'Cap(GB)':>14}{'% used':>9}")
            for dev in ordered:
                h = tier_hit_tokens.get(dev, 0)
                total_hit += h
                ratio = (h / tier_requested_tokens) * 100
                used, cap = usage.get(dev, (0, 0))
                pct = (used / cap * 100) if cap else 0.0
                rl_ms = tier_reload_ns.get(dev, 0) / 1e6
                print_markup(
                    f"{dev.name:<10}{h:>11}{ratio:>8.2f}%{rl_ms:>12.2f}"
                    f"{used/MB_TO_BYTE:>14.2f}{cap/GB_TO_BYTE:>14.2f}{pct:>8.3f}%")
            print_markup(f"Total prefix hit ratio (%):                                         {(total_hit/tier_requested_tokens)*100:.2f}")
            # Reload-latency metric: cost of serving hits from below NPU. NPU hits
            # are free; deeper tiers cost more, so a config that keeps hits in a
            # faster tier (e.g. JBOF vs COLDSTORE) shows a lower total here.
            total_reload_ns = sum(tier_reload_ns.values())
            reloaded_tokens = sum(tier_hit_tokens.get(d, 0) for d in ordered if d != Device.NPU)
            print_markup(f"Total prefix-reload latency (ms):                                   {total_reload_ns/1e6:.3f}")
            if current:
                print_markup(f"Reload latency as % of total sim time:                              {total_reload_ns/current*100:.3f}")
            if reloaded_tokens > 0:
                print_markup(f"Mean reload latency per 1K reloaded tokens (ms):                    {(total_reload_ns/1e6)/(reloaded_tokens/1000):.3f}")
        else:
            print_markup("Prefix hit ratio (%):                                               N/A (no requests tracked)")
        print_rule()
    if power_modeling:
        print_rule("[sim.tagline]Power Modeling Results[/]")
        total_energy = power_model.get_final_energy(current)
        print_markup(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print_markup(f"Power per {1/RATIO} sec (W): {power_model.power_time_series}")
        print_rule()
    # ---- Cluster-wide latency summary (pooled across ALL instances) ----
    # TTFT / TBT are the metrics that expose the tiered-cache benefit: a reused
    # prefix served from a fast tier keeps TTFT low, while a slow-tier reload
    # inflates the TTFT tail. These used to be printed per-instance (unreadable at
    # pod scale), so pool every finished request and report the distribution once.
    all_ttft = [r.ttft    for i in range(num_instances) for r in schedulers[i].done if r.ttft    is not None and r.ttft    >= 0]
    all_tpot = [r.tpot    for i in range(num_instances) for r in schedulers[i].done if r.tpot    is not None and r.tpot    >= 0]
    all_e2e  = [r.latency for i in range(num_instances) for r in schedulers[i].done if r.latency is not None and r.latency >= 0]
    n_done   = sum(len(schedulers[i].done) for i in range(num_instances))
    print_rule("[sim.tagline]Latency Summary (cluster-wide)[/]")
    print_markup(f"Finished requests: {n_done}    Instances: {num_instances}")
    print_markup(f"{'Metric':<18}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>10}    (ms)")
    def _lat_row(name, values):
        if not values:
            print_markup(f"{name:<18}{'--':>10}{'--':>10}{'--':>10}{'--':>10}{'--':>10}")
            return
        a = np.array(values, dtype=float) / 1_000_000.0   # ns -> ms
        print_markup(f"{name:<18}{np.mean(a):>10.2f}{np.percentile(a,50):>10.2f}"
                     f"{np.percentile(a,90):>10.2f}{np.percentile(a,99):>10.2f}{np.max(a):>10.2f}")
    _lat_row("TTFT", all_ttft)
    _lat_row("TBT (per-token)", all_tpot)
    _lat_row("E2E latency", all_e2e)
    print_rule()

    # ---- Session / resume latency (multi-turn "context parking", --session-metrics) ----
    # A conversation turn N>=1 must reload its accumulated context (parked in a lower
    # tier while the session was idle) before it can answer -> that reload lands in the
    # turn's TTFT. Splitting TTFT by first-turn (cold, no reload) vs resumed-turn (warm,
    # reloaded) isolates the tier-reload cost, so a fast tier (JBOF) vs a slow one
    # (COLDSTORE) shows up directly in resumed-turn TTFT. No effect unless the workload
    # carries sessions (records with "sub_requests").
    if session_metrics_enabled:
        done_reqs = [r for i in range(num_instances) for r in schedulers[i].done]
        first_ttft   = [r.ttft for r in done_reqs if r.sub_request_index == 0 and r.ttft is not None and r.ttft >= 0]
        resumed_ttft = [r.ttft for r in done_reqs if (r.sub_request_index or 0) >= 1 and r.ttft is not None and r.ttft >= 0]
        n_sessions = len({r.session_id for r in done_reqs if r.session_id is not None})
        print_rule("[sim.tagline]Session / resume latency (multi-turn)[/]")
        if n_sessions == 0:
            print_markup("No multi-turn sessions in this run (workload has no 'sub_requests').")
        else:
            print_markup(f"Sessions: {n_sessions}    first-turn (cold): {len(first_ttft)}    "
                         f"resumed-turn (warm / context reloaded): {len(resumed_ttft)}")
            print_markup(f"{'Metric':<20}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>10}    (ms)")
            _lat_row("TTFT first-turn", first_ttft)
            _lat_row("TTFT resumed-turn", resumed_ttft)
            if first_ttft and resumed_ttft:
                _delta = (np.mean(resumed_ttft) - np.mean(first_ttft)) / 1e6
                print_markup(f"Resume overhead (resumed - first, mean ms):                         {_delta:.2f}")
        print_rule()

    # Final dashboard snapshot (status="done") so the live view reflects the end state.
    if dash_enabled:
        try:
            dashboard.write_snapshot(dash_write_path, dashboard.build_snapshot(
                "done", clock_ns=current, freq=FREQ, wall_seconds=time() - start_time,
                config=dash_config, cpu_scope=cpu_scope, req_cnt=req_cnt,
                total_prompt=total_prompt, total_gen=total_gen,
                requests_total=len(getattr(router, "_pending_requests", [])),
                schedulers=schedulers, num_nodes=num_nodes, num_instances=num_instances,
                history=dash_history, live_prompt_tps=0.0, live_gen_tps=0.0))
            print_markup(f"Dashboard live metrics: {dash_file}")
        except Exception:
            pass

    # Per-instance detail only for small runs (avoids hundreds of blocks at pod scale).
    if num_instances <= 4:
        for i in range(num_instances):
            print_rule(f"[sim.tagline]Instance \\[{i}][/]")
            schedulers[i].print_result()
            print_rule()
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)

        # Persist the full CLI output alongside the CSV, same base name (.txt),
        # so the end-of-simulation results can be read later.
        txt_file = os.path.splitext(output_file)[0] + '.txt'
        txt_path = txt_file if os.path.isabs(txt_file) else f'../{txt_file}'
        txt_dir = os.path.dirname(txt_path)
        if txt_dir:
            os.makedirs(txt_dir, exist_ok=True)
        print(f"Saving console output to: {txt_file}")
        save_output_text(txt_path)


if __name__ == "__main__":
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
