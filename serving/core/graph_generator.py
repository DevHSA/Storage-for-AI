import os
import subprocess
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path

logger = get_logger("GraphGenerator")

# In-process Chakra LLM converter. The CLI path spawned a fresh Python interpreter
# per graph (`python -m chakra.src.converter.converter LLM ...`); its ~37 ms
# startup+import dominated many-instance runs (profiling: ~43% of frontend time).
# LLMConverter is exactly what that CLI subcommand ('LLM' -> convert_llm) runs, so
# importing it and calling it directly produces byte-identical .et output with no
# subprocess. Falls back to the subprocess CLI if the import is ever unavailable.
try:
    from chakra.src.converter.llm_converter import LLMConverter as _LLMConverter
except Exception:
    _LLMConverter = None

def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    if _LLMConverter is not None:
        # In-process: identical to the CLI 'LLM' subcommand (convert_llm) — same
        # class, same positional args -> byte-identical .et, no interpreter spawn.
        logger.debug("Generating graph in-process (LLMConverter): %s -> %s",
                     trace_path, output_path, extra={"node_id": node_id, "instance_id": instance_id})
        _LLMConverter(trace_path, output_path, num_npus, npu_offset, enable_local_offloading).convert()
    else:
        cmd = [
            'python', '-m', 'chakra.src.converter.converter', 'LLM',
            '--input', trace_path,
            '--output', output_path,
            '--num-npus', str(num_npus),
            '--npu-offset', str(npu_offset),
        ]
        if enable_local_offloading:
            cmd.append('--local-offloading')
        logger.debug("Generating graph via subprocess: %s", " ".join(cmd), extra={"node_id": node_id, "instance_id": instance_id})
        subprocess.run(cmd, cwd=chakra, text=True)
    return
