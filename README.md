<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/img/llmservingsim_full_primary_dark_transparent.png">
    <img alt="LLMServingSim" src="docs/static/img/llmservingsim_full_primary_transparent.png" width="70%">
  </picture>
</p>

<h3 align="center">
A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure
</h3>

<p align="center">
| <a href="https://llmservingsim.ai"><b>Website</b></a> | <a href="https://llmservingsim.ai/docs/getting-started/overview"><b>Documentation</b></a> | <a href="https://llmservingsim.ai/docs/contributor/welcome"><b>Contribute</b></a> | <a href="https://llmservingsim.ai/contact"><b>Contact</b></a> | <a href="https://llmservingsim.ai/changelog"><b>Changelog</b></a> |
</p>

We have built an LLMServingSim website to help you get started with the simulator. Please visit [llmservingsim.ai](https://llmservingsim.ai) for documentation, contribution guides, and team contact info.

## About

LLMServingSim is a cycle-level simulator for LLM serving infrastructure. It pairs a Python frontend that mirrors vLLM's continuous-batching scheduler with the ASTRA-Sim C++ analytical network backend, and drives both from per-hardware latency data captured by a vLLM-based layerwise profiler. The result is a unified environment for studying heterogeneous accelerators, disaggregated memory tiers (CPU / CXL / PIM), MoE routing, and multi-instance parallelism (TP / PP / EP / DP) end-to-end.

## Getting Started

> **This fork (DevHSA)** adds a vendor-neutral **JBOF** pooled-flash KV-cache tier to the memory
> hierarchy (**NPU → CPU → FLASH → JBOF → COLDSTORE**) and a cluster-wide **TTFT / TBT** latency
> summary. The ASTRA-Sim backend carrying those changes lives in the
> [`astra-sim-jbof`](https://github.com/DevHSA/astra-sim-jbof) submodule — so clone **recursively**.

```bash
git clone --recursive https://github.com/DevHSA/Storage-for-AI.git
cd Storage-for-AI
# forgot --recursive?  ->  git submodule update --init --recursive

./scripts/docker-sim.sh           # launch the simulator container
./scripts/compile.sh              # build ASTRA-Sim + install the Chakra converter
                                  #   protobuf gencode/runtime mismatch? -> pip3 install --upgrade protobuf

# workload traces are GENERATED (not stored in git); build one, then run the
# active example in run.sh -- a simple 8-GPU JBOF run:
python workloads/generators/make_rack_fill.py workloads/pod_prop_8gpu.jsonl 450 2048 8 4 3000
./serving/run.sh
```

**New to LLM serving concepts** (tokens, KV cache, prefill/decode, TP, batching)? Start with the
**[`docs/llmservingsim_primer.html`](docs/llmservingsim_primer.html)** primer — it builds every concept
from the ground up (with worked Llama-3.1-8B numbers) and ties it to this code.

**Ready to run it?** Open the guided walkthrough in
**[`docs/llmservingsim_intro.html`](docs/llmservingsim_intro.html)** (view it in a browser) — it
starts from a 1-GPU run, explains the config file, the workload format, and the output metrics
(TTFT / TBT), then scales up to a faithful Vera Rubin rack and pod.

For the upstream simulator's installation details, container choices, CLI flags, and the full
example set, see the [LLMServingSim docs](https://llmservingsim.ai/docs/getting-started/overview).

## Publications

**ISPASS 2026**  
*LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure*  
Jaehong Cho<sup>\*</sup>, Hyunmin Choi<sup>\*</sup>, Guseul Heo, Jongse Park (KAIST) [[Paper]](https://doi.org/10.1109/ISPASS69572.2026.00012)  
<sup>\*</sup>Equal contribution  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18879965.svg)](https://doi.org/10.5281/zenodo.18879965)

**CAL 2025**  
*LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure*  
Jaehong Cho, Hyunmin Choi, Jongse Park (KAIST)  [[Paper]](https://doi.org/10.1109/LCA.2025.3628325)

**IISWC 2024**  
*LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale*  
Jaehong Cho, Minsu Kim, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)  [[Paper]](https://doi.org/10.1109/IISWC63097.2024.00012)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12803583.svg)](https://doi.org/10.5281/zenodo.12803583)

## Citation

If you use LLMServingSim in your research, please cite:

```bibtex
@INPROCEEDINGS{11527300,
    author={Cho, Jaehong and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
    booktitle={2026 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)}, 
    title={{LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure}}, 
    year={2026},
    pages={1-14},
    doi={10.1109/ISPASS69572.2026.00012}
}

@ARTICLE{11224567,
    author={Cho, Jaehong and Choi, Hyunmin and Park, Jongse},
    journal={IEEE Computer Architecture Letters},
    title={{LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving
            Techniques in LLM Infrastructure}},
    year={2025},
    volume={24},
    number={02},
    pages={361-364},
    doi={10.1109/LCA.2025.3628325},
    ISSN={1556-6064},
    publisher={IEEE Computer Society},
    address={Los Alamitos, CA, USA},
    month=jul
}

@INPROCEEDINGS{10763697,
    author={Cho, Jaehong and Kim, Minsu and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
    booktitle={2024 IEEE International Symposium on Workload Characterization (IISWC)},
    title={{LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving
            at Scale}},
    year={2024},
    pages={15-29},
    doi={10.1109/IISWC63097.2024.00012}
}
```
