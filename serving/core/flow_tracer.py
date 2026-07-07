"""Execution-flow tracer for the LLMServingSim Python wrapper.

Writes a plain-English, step-by-step log of the functions that run during a
simulation, in call order. Each step reads like a sentence:

    Step 12  ·  MemoryModel.prefix_match   [memory-model]   serving/core/memory_model.py:623   (called by Scheduler.schedule_with_prefix)
        This function: probes NPU + every pooled tier for the request's cached prefix ...
        The input to this function is: req=<Request id=0>
        The function output: 2

Reading top-to-bottom tells you what ran, in what order, what each function is
for, what it received, and what it produced. A per-layer call-count INVENTORY at
the bottom lists every function that ran (the complete "what was called" list).

Default (compact) view: each distinct function is shown ONCE, at its first call,
in call order — a real run calls the same helpers thousands of times (once per
model layer), so showing every call is unreadable. Use --trace-flow-full for the
exhaustive every-call version.

Scope: only functions defined under serving/ are traced (not stdlib / 3rd-party).
"What it does" comes from each function's docstring first line, with curated
overrides (`_DESC`) for the key pipeline steps — notably the ASTRA-Sim co-process
boundary in controller.py. This traces only the Python wrapper; the ASTRA-Sim C++
binary runs as a separate co-process (write_flush hands off; read_wait/parse_output
read its reply).

Enable:  python -m serving ... --trace-flow [path]        (default outputs/flow_trace.log)
         python -m serving ... --trace-flow [path] --trace-flow-full
The file is overwritten at the start of every run.
"""

import os
import sys
import threading
import atexit

# --- paths / scope -----------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../serving/core
_SERVING_DIR = os.path.dirname(_THIS_DIR)                       # .../serving
_REPO_ROOT = os.path.dirname(_SERVING_DIR)                      # repo root
_SELF_FILE = os.path.abspath(__file__)

_roots = [_SERVING_DIR]

# --- layer labels (by module basename) --------------------------------------
_LAYERS = {
    "__main__": "orchestration",
    "scheduler": "scheduler",
    "memory_model": "memory-model",
    "trace_generator": "trace-gen",
    "graph_generator": "graph-gen(Chakra)",
    "config_builder": "config-builder",
    "controller": "ASTRA-Sim-IPC",
    "request": "request",
    "logger": "logging",
    "power_model": "power-model",
    "radix_cache": "prefix-cache",
    "radix_tree": "prefix-cache",
    "router": "router",
    "pim_model": "pim-model",
    "run_paths": "run-paths",
    "utils": "utils",
    "flow_tracer": "tracer",
}

# --- curated descriptions for key steps (override docstring) ------------------
_DESC = {
    "main": "Top-level entrypoint: parse CLI, build the cluster, run the scheduling loop, and print results.",
    "build_cluster_config": "Parses the cluster JSON and WRITES the ASTRA-Sim input files (network.yml / system.json / memory_expansion.json, including the deep tiers).",
    "generate_trace": "Turns one scheduled Batch into a per-layer text trace (including kv_evict and deep-tier reload MEM rows).",
    "generate_graph": "Converts the text trace into a Chakra .et graph by spawning the chakra_converter subprocess.",
    "prefix_match": "Probes the NPU cache and every pooled tier for the request's cached prefix, assigning each segment to the shallowest tier that still holds it.",
    "tier_load_latency_ns": "Computes a deep-tier reload latency analytically = mem_latency + bytes/mem_bw + link term (used for the reload metric).",
    "write_flush": "HANDS OFF TO the ASTRA-Sim co-process: writes the step signal to its stdin so it simulates this batch.",
    "read_wait": "BLOCKS on the ASTRA-Sim co-process's stdout until it finishes simulating this step.",
    "parse_output": "Parses ASTRA-Sim's stdout line into the step's cycle count and advances the simulated clock.",
    "check_end": "Checks ASTRA-Sim's output for the end-of-simulation marker.",
    "schedule": "Runs one scheduling step for this instance (picks the next batch of requests to simulate).",
    "schedule_with_prefix": "Scheduling step WITH prefix caching: match cached prefixes, apply the token budget, evict from NPU, and build the Batch (+ tier reloads).",
    "add_done": "Post-step bookkeeping: mark finished requests and write computed prefixes through into every tier.",
}

# --- state -------------------------------------------------------------------
_local = threading.local()
_lock = threading.Lock()
_fp = None
_installed = False
_full = False
_seq = 0
_records = []                 # compact mode: prose blocks, in call order
_seen = {}                    # qualname -> {count, layer, rel, desc}
_order = []                   # qualnames in first-seen order


def _stack():
    st = getattr(_local, "stack", None)
    if st is None:
        st = []
        _local.stack = st
    return st


def _next_seq():
    global _seq
    _seq += 1
    return _seq


def _in_scope(filename):
    if not filename:
        return False
    af = os.path.abspath(filename)
    if af == _SELF_FILE:
        return False
    return any(af.startswith(r + os.sep) for r in _roots)


def _layer_for(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return _LAYERS.get(base) or os.path.basename(os.path.dirname(filename)) or base


def _describe(code, qualname):
    if qualname in _DESC:
        return _DESC[qualname]
    short = qualname.split(".")[-1]
    if short in _DESC:
        return _DESC[short]
    consts = code.co_consts
    if consts and isinstance(consts[0], str) and consts[0].strip():
        return consts[0].strip().splitlines()[0].strip()
    return "(no docstring — inspect the source for details)"


def _summarize(v, depth=0, maxlen=140):
    """Compact one-line summary of a value (shape/len/id, not a giant repr)."""
    try:
        if v is None:
            return "None"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return repr(v if len(v) <= maxlen else v[:maxlen] + f"...(+{len(v)-maxlen}c)")
        if isinstance(v, (bytes, bytearray)):
            return f"<{type(v).__name__} of {len(v)} bytes>"
        tn = type(v).__name__
        if tn == "ndarray":
            return f"<ndarray shape={getattr(v, 'shape', '?')} dtype={getattr(v, 'dtype', '?')}>"
        if tn in ("DataFrame", "Series"):
            return f"<{tn} shape={getattr(v, 'shape', '?')}>"
        if isinstance(v, dict):
            ks = [str(k) for k in list(v.keys())[:6]]
            return f"<dict of {len(v)} items, keys={ks}>"
        if isinstance(v, (list, tuple, set, frozenset)):
            n = len(v)
            extra = (", e.g. " + _summarize(next(iter(v)), depth + 1, 50)) if (n and depth == 0) else ""
            return f"<{tn} of {n} items{extra}>"
        for a in ("id", "batch_id", "request_id", "instance_id", "node_id", "name", "device"):
            if hasattr(v, a):
                try:
                    return f"<{tn} {a}={getattr(v, a)!r}>"
                except Exception:
                    break
        if type(v).__module__ == "builtins":
            r = repr(v)
            return r if len(r) <= maxlen else r[:maxlen] + "..."
        return f"<{tn} object>"
    except Exception:
        return "<value could not be summarized>"


def _fmt_inputs(frame, code):
    n = code.co_argcount + code.co_kwonlyargcount
    names = code.co_varnames[:n]
    cls = ""
    parts = []
    loc = frame.f_locals
    for a in names:
        if a in ("self", "cls") and a in loc:
            try:
                cls = type(loc[a]).__name__ + "."
            except Exception:
                cls = ""
            continue
        parts.append(f"{a}={_summarize(loc.get(a))}")
    return cls, (", ".join(parts) if parts else "(no arguments)")


def _write(line=""):
    if _fp is not None:
        _fp.write(line + "\n")


def _emit_block(rec):
    ind = "    " * rec["depth"]
    _write()
    _write(f"{ind}Step {rec['seq']}  ·  {rec['qn']}   [{rec['layer']}]   {rec['rel']}:{rec['line']}   (called by {rec['caller']})")
    _write(f"{ind}    This function: {rec['desc']}")
    _write(f"{ind}    The input to this function is: {rec['input']}")
    _write(f"{ind}    The function output: {rec['output']}")


def _profiler(frame, event, arg):
    if event != "call" and event != "return":
        return
    code = frame.f_code
    fn = code.co_filename
    if not _in_scope(fn):
        return
    name = code.co_name
    if name.startswith("<"):
        return
    st = _stack()
    try:
        if event == "call":
            cls, inp = _fmt_inputs(frame, code)
            qn = cls + name
            rel = os.path.relpath(fn, _REPO_ROOT)
            layer = _layer_for(fn)
            parent = st[-1]["qn"] if st else "(top level)"
            with _lock:
                r = _seen.get(qn)
                if r is None:
                    r = {"count": 0, "layer": layer, "rel": rel, "desc": _describe(code, qn)}
                    _seen[qn] = r
                    _order.append(qn)
                r["count"] += 1
                first = r["count"] == 1
            logged = _full or first
            depth = sum(1 for e in st if e["logged"])
            entry = {"qn": qn, "logged": logged, "rec": None, "sdepth": depth}
            if logged:
                rec = {"seq": _next_seq(), "depth": depth, "qn": qn, "layer": layer,
                       "rel": rel, "line": frame.f_lineno, "caller": parent,
                       "desc": r["desc"], "input": inp, "output": "(did not return — errored or still running)"}
                if _full:
                    # stream immediately (output line comes at return)
                    with _lock:
                        ind = "    " * depth
                        _write()
                        _write(f"{ind}Step {rec['seq']}  ·  {qn}   [{layer}]   {rel}:{frame.f_lineno}   (called by {parent})")
                        _write(f"{ind}    This function: {rec['desc']}")
                        _write(f"{ind}    The input to this function is: {inp}")
                    entry["rec"] = rec
                else:
                    with _lock:
                        _records.append(rec)
                    entry["rec"] = rec
            st.append(entry)
        else:  # return
            if not st:
                return
            e = st.pop()
            if e["logged"] and e["rec"] is not None:
                out = _summarize(arg)
                if _full:
                    with _lock:
                        _write(f"{'    ' * e['sdepth']}    The function output: {out}")
                else:
                    e["rec"]["output"] = out
    except Exception:
        pass


def install(path=None, full=False, roots=None):
    """Start flow tracing.
    path : log file (overwritten). Relative paths resolve against the repo root
           (NOT cwd — the wrapper chdirs into astra-sim/ during a run).
    full : log every call (exhaustive) instead of first-occurrence-per-function.
    roots: extra directory prefixes to include beyond serving/.
    """
    global _fp, _installed, _roots, _full
    if _installed:
        return
    _full = bool(full)
    if roots:
        _roots = _roots + [os.path.abspath(r) for r in roots]
    if not path:
        path = "outputs/flow_trace.log"
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _fp = open(path, "w", buffering=1, encoding="utf-8")
    _fp.write("=" * 100 + "\n")
    _fp.write("LLMServingSim — execution flow, in plain English (Python wrapper only)\n")
    _fp.write("=" * 100 + "\n")
    _fp.write("command : " + " ".join(sys.argv) + "\n")
    _fp.write("log file: " + path + "\n")
    _fp.write("view    : " + ("FULL — every call, in order" if _full
                              else "COMPACT — each function shown once (first call), in call order") + "\n")
    _fp.write("\nHow to read this file:\n")
    _fp.write("  * Steps are listed in the order functions were called (Step 1, 2, 3, ...).\n")
    _fp.write("  * Indentation shows nesting: an indented step was called by the step above it at one-less indent.\n")
    _fp.write("  * Each step says what the function does, the input it received, and the value it returned.\n")
    if not _full:
        _fp.write("  * Compact view: each function appears once. The INVENTORY at the very bottom lists\n")
        _fp.write("    every function with its TOTAL call count (e.g. the scheduling loop repeats many times).\n")
    _fp.write("  * ASTRA-Sim's C++ simulator runs as a separate process: 'write_flush' hands a step to it,\n")
    _fp.write("    and 'read_wait'/'parse_output' read back the cycle count. Its C++ internals are not traced.\n")
    _fp.write("=" * 100 + "\n")
    if _full:
        _fp.write("\n")   # full mode streams below
    threading.setprofile(_profiler)
    sys.setprofile(_profiler)
    _installed = True
    atexit.register(_shutdown)


def _shutdown():
    global _fp, _installed
    try:
        sys.setprofile(None)
        threading.setprofile(None)
    except Exception:
        pass
    with _lock:
        if _fp is not None:
            if not _full:
                for rec in _records:
                    _emit_block(rec)
            # Inventory: every function that ran, grouped by layer, with counts.
            _write("\n\n" + "=" * 100)
            _write("INVENTORY — every wrapper function that ran (grouped by layer, with total call count)")
            _write("=" * 100)
            by_layer = {}
            for qn in _order:
                r = _seen[qn]
                by_layer.setdefault(r["layer"], []).append((qn, r))
            for layer in sorted(by_layer):
                _write(f"\n[{layer}]")
                for qn, r in sorted(by_layer[layer], key=lambda x: -x[1]["count"]):
                    _write(f"  called {r['count']:>6} time(s)   {qn}")
                    _write(f"                          {r['desc']}")
            total_funcs = len(_order)
            total_calls = sum(r["count"] for r in _seen.values())
            _write("\n" + "-" * 100)
            _write(f"{total_funcs} distinct wrapper functions ran, {total_calls} total calls.")
            _write("[end of flow trace]")
            _fp.close()
            _fp = None
    _installed = False
