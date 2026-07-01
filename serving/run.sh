# #!/bin/bash

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

# Multi-tier prefix-cache spillover example (NPU -> CPU -> FLASH -> ICMS -> COLDSTORE)
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
# WITH ICMS -> reused prefix served from the fast ICMS tier.
python -m serving \
  --cluster-config 'configs/cluster/cmp_with_icms.json' \
  --dtype float16 --block-size 16 \
  --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
  --dataset 'workloads/tier_demo_trace.jsonl' \
  --output 'outputs/cmp_with_icms.csv'

# WITHOUT ICMS (icms_mem absent from config) -> same reuse falls to slow COLDSTORE.
# python -m serving \
#   --cluster-config 'configs/cluster/cmp_no_icms.json' \
#   --dtype float16 --block-size 16 \
#   --enable-prefix-caching --enable-prefix-sharing --prefix-storage COLDSTORE \
#   --dataset 'workloads/tier_demo_trace.jsonl' \
#   --output 'outputs/cmp_no_icms.csv'

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