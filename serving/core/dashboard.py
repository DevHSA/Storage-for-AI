"""Live-metrics emitter for the LLMServingSim dashboard.

The simulator calls :func:`write_snapshot` every log interval (and once at the
end) with a snapshot built by :func:`build_snapshot`. The snapshot is a single
JSON object containing every metric shown on the console output PLUS live
per-tier memory occupancy (with pooled-vs-per-device breakdown) and running
counters. A separate Streamlit app polls the JSON file and renders it.

Design notes:
- Robust: every section is wrapped so a dashboard hiccup never crashes the sim.
- Tier SCOPE is inferred from cache-object identity across instances:
  one shared object  -> "pooled"; one per node -> "per_node"; one per instance
  -> "per_instance". Per-device tiers carry a per-device split AND a total.
- Latency percentiles mirror the console "Latency Summary" exactly (ns -> ms).
"""

import json
import os
import tempfile

try:
    import numpy as np
except Exception:                       # numpy always present in the sim env
    np = None


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def _safe_call(fn, default=0):
    try:
        return fn()
    except Exception:
        return default


def _pct(used, cap):
    return (used / cap * 100.0) if cap else 0.0


def _percentiles(values_ns):
    """mean/p50/p90/p99/max in ms (ns input); None-safe, empty -> zeros+n=0."""
    vals = [v for v in values_ns if v is not None and v >= 0]
    if not vals or np is None:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "n": len(vals)}
    a = np.array(vals, dtype=float) / 1_000_000.0        # ns -> ms
    return {
        "mean": float(np.mean(a)), "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)), "n": len(vals),
    }


def _device_label(scope, insts, nodes):
    if scope == "pooled":
        return "pod-wide"
    if scope == "per_node":
        return f"node {nodes[0]}" if len(nodes) == 1 else f"nodes {nodes}"
    # per_instance / per_device
    return f"inst {insts[0]}" if len(insts) == 1 else f"insts {insts}"


def _gather_tier(name, cache_getter, schedulers, num_nodes, num_instances):
    """Build one tier's usage record from every scheduler's cache object.

    Dedups by object identity so a pooled (shared) cache is counted once; a
    per-node / per-instance tier yields one entry per distinct cache."""
    seen, order = {}, []
    for s in schedulers:
        c = _safe_call(lambda: cache_getter(s), None)
        if c is None:
            continue
        k = id(c)
        if k not in seen:
            seen[k] = {"cache": c, "insts": [], "nodes": set()}
            order.append(k)
        seen[k]["insts"].append(getattr(s, "instance_id", -1))
        seen[k]["nodes"].add(getattr(s, "node_id", -1))
    if not order:
        return None

    n_unique = len(order)
    if n_unique == 1:
        scope = "pooled"
    elif num_nodes > 1 and n_unique == num_nodes:
        scope = "per_node"
    elif n_unique == num_instances:
        scope = "per_instance"
    else:
        scope = "per_device"

    devices, total_used, total_cap = [], 0, 0
    for k in order:
        rec = seen[k]
        c = rec["cache"]
        used = _safe_call(c.total_memory_usage, 0)
        cap = getattr(c, "capacity", 0) or 0
        devices.append({
            "label": _device_label(scope, sorted(rec["insts"]), sorted(rec["nodes"])),
            "used_bytes": used, "cap_bytes": cap, "pct": _pct(used, cap),
        })
        total_used += used
        total_cap += cap
    return {
        "name": name, "scope": scope, "devices": devices,
        "total_used_bytes": total_used, "total_cap_bytes": total_cap,
        "total_pct": _pct(total_used, total_cap),
    }


# --------------------------------------------------------------------------- #
#  Snapshot builder
# --------------------------------------------------------------------------- #
def build_snapshot(status, *, clock_ns, freq, wall_seconds, config,
                   req_cnt, total_prompt, total_gen, requests_total,
                   schedulers, num_nodes, num_instances,
                   throughput_history=None, live_prompt_tps=0.0, live_gen_tps=0.0):
    """Assemble the full live-metrics dict. Never raises (best-effort)."""
    sim_seconds = (clock_ns / freq) if freq else 0.0

    # ---- counters -------------------------------------------------------- #
    running = _safe_call(lambda: sum(
        len(b.requests) for s in schedulers for b in s.inflight), 0)
    waiting = _safe_call(lambda: sum(
        len([r for r in s.request if r.arrival <= clock_ns]) for s in schedulers), 0)
    batches = _safe_call(lambda: sum(max(0, s.batch_ids + 1) for s in schedulers), 0)
    counters = {
        "requests_total": requests_total,
        "requests_finished": req_cnt,
        "requests_running": running,
        "requests_waiting": waiting,
        "prompt_tokens": total_prompt,          # prefill tokens
        "decode_tokens": total_gen,             # generated (decode) tokens
        "total_tokens": total_prompt + total_gen,
        "batches": batches,
    }

    # ---- throughput ------------------------------------------------------ #
    thr = {
        "req_per_s": (req_cnt / sim_seconds) if sim_seconds else 0.0,
        "prompt_tok_per_s": (total_prompt / sim_seconds) if sim_seconds else 0.0,
        "decode_tok_per_s": (total_gen / sim_seconds) if sim_seconds else 0.0,
        "total_tok_per_s": ((total_prompt + total_gen) / sim_seconds) if sim_seconds else 0.0,
        "live_prompt_tok_per_s": live_prompt_tps,
        "live_decode_tok_per_s": live_gen_tps,
        "history": list(throughput_history or []),      # [{t_s, prompt_tps, decode_tps}]
    }

    # ---- tiers (memory occupancy) --------------------------------------- #
    tiers = []
    # NPU KV prefix cache (per instance/GPU)
    npu = _gather_tier("NPU", lambda s: getattr(s.memory, "npu_prefix_cache", None),
                       schedulers, num_nodes, num_instances)
    if npu:
        tiers.append(npu)
    # CPU second tier (scope depends on --cpu-scope)
    cpu = _gather_tier("CPU", lambda s: getattr(s.memory, "second_tier_prefix_cache", None),
                       schedulers, num_nodes, num_instances)
    if cpu:
        tiers.append(cpu)
    # Deep tiers (FLASH per-node; JBOF/COLDSTORE pooled) — names from first sched
    deep_names = _safe_call(lambda: [t["name"] for t in schedulers[0].memory.deep_tiers], [])
    for dn in deep_names:
        def getter(s, _dn=dn):
            for t in getattr(s.memory, "deep_tiers", []):
                if t["name"] == _dn:
                    return t["cache"]
            return None
        rec = _gather_tier(dn, getter, schedulers, num_nodes, num_instances)
        if rec:
            tiers.append(rec)

    # Extra NPU HBM view (weight + KV) per instance — context, not a KV tier
    npu_hbm = []
    for s in schedulers:
        m = s.memory
        # npu_used = weight + active KV + retained prefix KV; floor at the resident
        # weight so the HBM view never reads below it (npu_used can drop to 0 in the
        # end-of-run teardown state).
        used = max(getattr(m, "npu_used", 0), getattr(m, "weight", 0))
        cap = getattr(m, "npu_mem", 0)
        npu_hbm.append({
            "instance": getattr(s, "instance_id", -1),
            "node": getattr(s, "node_id", -1),
            "num_npus": getattr(s, "num_npus", 1),
            "used_bytes": used, "cap_bytes": cap, "pct": _pct(used, cap),
            "weight_bytes": getattr(m, "weight", 0),
        })

    # ---- prefix-cache hits per tier (aggregated across instances) ------- #
    total_requested, hit_tokens, reload_ns = 0, {}, {}
    for s in schedulers:
        req_i, hits_i = _safe_call(s.memory.tier_hit_report, (0, {}))
        total_requested += req_i or 0
        for dev, h in (hits_i or {}).items():
            hit_tokens[dev.name if hasattr(dev, "name") else str(dev)] = \
                hit_tokens.get(dev.name if hasattr(dev, "name") else str(dev), 0) + h
        rl, _rb = _safe_call(s.memory.reload_report, ({}, {}))
        for dev, ns in (rl or {}).items():
            key = dev.name if hasattr(dev, "name") else str(dev)
            reload_ns[key] = reload_ns.get(key, 0) + ns
    prefix_tiers = []
    for dev, h in hit_tokens.items():
        prefix_tiers.append({
            "name": dev, "hit_tokens": h,
            "hit_pct": (h / total_requested * 100.0) if total_requested else 0.0,
            "reload_ms": reload_ns.get(dev, 0) / 1_000_000.0,
        })
    prefix_cache = {
        "requested_tokens": total_requested,
        "total_hit_ratio": (sum(hit_tokens.values()) / total_requested * 100.0)
                           if total_requested else 0.0,
        "tiers": prefix_tiers,
    }

    # ---- latency (running, mirrors console Latency Summary) ------------- #
    all_ttft = [r.ttft for s in schedulers for r in s.done]
    all_tpot = [r.tpot for s in schedulers for r in s.done]
    all_e2e = [r.latency for s in schedulers for r in s.done]
    all_itl = [x for s in schedulers for r in s.done for x in (r.itl or [])]
    latency = {
        "finished": sum(len(s.done) for s in schedulers),
        "ttft": _percentiles(all_ttft),
        "tbt": _percentiles(all_tpot),
        "itl": _percentiles(all_itl),
        "e2e": _percentiles(all_e2e),
    }

    return {
        "schema": 1,
        "status": status,                       # "running" | "done"
        "wall_seconds": round(wall_seconds, 3),
        "sim": {"clock_ns": clock_ns, "sim_seconds": sim_seconds},
        "config": config,
        "counters": counters,
        "throughput": thr,
        "tiers": tiers,
        "npu_hbm": npu_hbm,
        "prefix_cache": prefix_cache,
        "latency": latency,
    }


def write_snapshot(path, snapshot):
    """Atomically write the snapshot JSON (tmp file + os.replace)."""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        # mkstemp is 0600; the sim may run as root (Docker) while the dashboard
        # reads as the host user, so make the snapshot world-readable.
        try:
            os.chmod(tmp, 0o644)
        except Exception:
            pass
        os.replace(tmp, path)
    except Exception:
        pass
