"""Live-metrics emitter for the LLMServingSim dashboard.

The simulator calls :func:`write_snapshot` every log interval (and once at the
end) with a snapshot built by :func:`build_snapshot`. The snapshot is a single
JSON object with every metric on the console output PLUS live per-tier memory
occupancy (explicit scope + per-device split), per-instance drill-down, tier
bw/latency specs, and a timeline history (throughput, tier memory, TTFT, TBT
per interval) so the dashboard can draw over-time charts.

Robust: every section is best-effort so a dashboard hiccup never crashes the sim.
"""

import json
import os
import tempfile

try:
    import numpy as np
except Exception:
    np = None


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _safe_call(fn, default=0):
    try:
        return fn()
    except Exception:
        return default


def _pct(used, cap):
    return (used / cap * 100.0) if cap else 0.0


def _percentiles(values_ns):
    vals = [v for v in values_ns if v is not None and v >= 0]
    if not vals or np is None:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "n": len(vals)}
    a = np.array(vals, dtype=float) / 1_000_000.0        # ns -> ms
    return {"mean": float(np.mean(a)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)), "n": len(vals)}


def _scope_for(name, cpu_scope):
    """Explicit, unambiguous scope label per tier (independent of topology)."""
    if name == "NPU":
        return "per-GPU"
    if name == "CPU":
        return (cpu_scope or "per_node").replace("_", "-")   # per-instance / per-node
    if name == "FLASH":
        return "per-node"
    return "pooled"                                          # JBOF / COLDSTORE


def _device_label(insts, nodes, scope):
    if scope == "pooled":
        return "pod-wide"
    if len(insts) == 1:
        return f"GPU {insts[0]}"
    if len(nodes) == 1:
        return f"node {nodes[0]}"
    return f"insts {insts}"


def _gather_tier(name, cache_getter, schedulers, scope):
    """One tier's usage from every scheduler's cache, deduped by object identity
    (pooled = counted once; per-node/per-instance = one entry per distinct cache).
    The tier's SCOPE label is set explicitly by the caller."""
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
    devices, total_used, total_cap = [], 0, 0
    for k in order:
        rec = seen[k]
        c = rec["cache"]
        used = _safe_call(c.total_memory_usage, 0)
        cap = getattr(c, "capacity", 0) or 0
        devices.append({"label": _device_label(sorted(rec["insts"]), sorted(rec["nodes"]), scope),
                        "used_bytes": used, "cap_bytes": cap, "pct": _pct(used, cap)})
        total_used += used
        total_cap += cap
    return {"name": name, "scope": scope, "devices": devices,
            "total_used_bytes": total_used, "total_cap_bytes": total_cap,
            "total_pct": _pct(total_used, total_cap)}


# --------------------------------------------------------------------------- #
#  Snapshot builder
# --------------------------------------------------------------------------- #
def build_snapshot(status, *, clock_ns, freq, wall_seconds, config, cpu_scope,
                   req_cnt, total_prompt, total_gen, requests_total,
                   schedulers, num_nodes, num_instances,
                   history=None, t_s=None, record_point=False,
                   live_prompt_tps=0.0, live_gen_tps=0.0):
    sim_seconds = (clock_ns / freq) if freq else 0.0

    # ---- counters -------------------------------------------------------- #
    running = _safe_call(lambda: sum(len(b.requests) for s in schedulers for b in s.inflight), 0)
    waiting = _safe_call(lambda: sum(
        len([r for r in s.request if r.arrival <= clock_ns]) for s in schedulers), 0)
    batches = _safe_call(lambda: sum(max(0, s.batch_ids + 1) for s in schedulers), 0)
    counters = {
        "requests_total": requests_total, "requests_finished": req_cnt,
        "requests_running": running, "requests_waiting": waiting,
        "prompt_tokens": total_prompt, "decode_tokens": total_gen,
        "total_tokens": total_prompt + total_gen, "batches": batches,
    }

    # ---- tiers (explicit scope) ----------------------------------------- #
    tiers = []
    npu = _gather_tier("NPU", lambda s: getattr(s.memory, "npu_prefix_cache", None),
                       schedulers, _scope_for("NPU", cpu_scope))
    if npu:
        tiers.append(npu)
    cpu = _gather_tier("CPU", lambda s: getattr(s.memory, "second_tier_prefix_cache", None),
                       schedulers, _scope_for("CPU", cpu_scope))
    if cpu:
        tiers.append(cpu)
    deep_names = _safe_call(lambda: [t["name"] for t in schedulers[0].memory.deep_tiers], [])
    for dn in deep_names:
        def getter(s, _dn=dn):
            for t in getattr(s.memory, "deep_tiers", []):
                if t["name"] == _dn:
                    return t["cache"]
            return None
        rec = _gather_tier(dn, getter, schedulers, _scope_for(dn, cpu_scope))
        if rec:
            tiers.append(rec)

    # ---- per-instance drill-down + NPU HBM ------------------------------ #
    instances_info, npu_hbm = [], []
    for s in schedulers:
        m = s.memory
        running_i = _safe_call(lambda: sum(len(b.requests) for b in s.inflight), 0)
        waiting_i = _safe_call(lambda: len([r for r in s.request if r.arrival <= clock_ns]), 0)
        req_i, hits_i = _safe_call(m.tier_hit_report, (0, {}))
        hit_sum = sum((hits_i or {}).values())
        used = max(getattr(m, "npu_used", 0), getattr(m, "weight", 0))
        cap = getattr(m, "npu_mem", 0)
        kv = _safe_call(lambda: m.npu_prefix_cache.total_memory_usage(), 0)
        instances_info.append({
            "instance": getattr(s, "instance_id", -1), "node": getattr(s, "node_id", -1),
            "num_npus": getattr(s, "num_npus", 1), "running": running_i, "waiting": waiting_i,
            "hbm_pct": _pct(used, cap), "npu_kv_bytes": kv,
            "hit_ratio": (hit_sum / req_i * 100.0) if req_i else 0.0,
        })
        npu_hbm.append({
            "instance": getattr(s, "instance_id", -1), "node": getattr(s, "node_id", -1),
            "num_npus": getattr(s, "num_npus", 1), "used_bytes": used, "cap_bytes": cap,
            "pct": _pct(used, cap), "weight_bytes": getattr(m, "weight", 0),
        })

    # ---- prefix-cache hits + reload per tier ---------------------------- #
    total_requested, hit_tokens, reload_ns = 0, {}, {}
    for s in schedulers:
        req_i, hits_i = _safe_call(s.memory.tier_hit_report, (0, {}))
        total_requested += req_i or 0
        for dev, h in (hits_i or {}).items():
            key = dev.name if hasattr(dev, "name") else str(dev)
            hit_tokens[key] = hit_tokens.get(key, 0) + h
        rl, _rb = _safe_call(s.memory.reload_report, ({}, {}))
        for dev, ns in (rl or {}).items():
            key = dev.name if hasattr(dev, "name") else str(dev)
            reload_ns[key] = reload_ns.get(key, 0) + ns
    prefix_tiers = [{
        "name": dev, "hit_tokens": h,
        "hit_pct": (h / total_requested * 100.0) if total_requested else 0.0,
        "reload_ms": reload_ns.get(dev, 0) / 1_000_000.0,
    } for dev, h in hit_tokens.items()]
    prefix_cache = {
        "requested_tokens": total_requested,
        "total_hit_ratio": (sum(hit_tokens.values()) / total_requested * 100.0) if total_requested else 0.0,
        "tiers": prefix_tiers,
    }

    # ---- latency (running) ---------------------------------------------- #
    latency = {
        "finished": sum(len(s.done) for s in schedulers),
        "ttft": _percentiles([r.ttft for s in schedulers for r in s.done]),
        "tbt": _percentiles([r.tpot for s in schedulers for r in s.done]),
        "itl": _percentiles([x for s in schedulers for r in s.done for x in (r.itl or [])]),
        "e2e": _percentiles([r.latency for s in schedulers for r in s.done]),
    }

    # ---- timeline history (append this interval's point) ---------------- #
    if record_point and history is not None:
        history.append({
            "t_s": t_s if t_s is not None else sim_seconds,
            "prompt_tps": live_prompt_tps, "decode_tps": live_gen_tps,
            "mem": {t["name"]: t["total_used_bytes"] for t in tiers},
            "ttft": latency["ttft"]["mean"], "tbt": latency["tbt"]["mean"],
        })
    thr = {
        "req_per_s": (req_cnt / sim_seconds) if sim_seconds else 0.0,
        "prompt_tok_per_s": (total_prompt / sim_seconds) if sim_seconds else 0.0,
        "decode_tok_per_s": (total_gen / sim_seconds) if sim_seconds else 0.0,
        "total_tok_per_s": ((total_prompt + total_gen) / sim_seconds) if sim_seconds else 0.0,
        "live_prompt_tok_per_s": live_prompt_tps, "live_decode_tok_per_s": live_gen_tps,
        "history": list(history or []),
    }

    return {
        "schema": 2, "status": status, "wall_seconds": round(wall_seconds, 3),
        "sim": {"clock_ns": clock_ns, "sim_seconds": sim_seconds},
        "config": config, "counters": counters, "throughput": thr,
        "tiers": tiers, "instances": instances_info, "npu_hbm": npu_hbm,
        "prefix_cache": prefix_cache, "latency": latency,
    }


def write_snapshot(path, snapshot):
    """Atomically write the snapshot JSON (world-readable: sim may be root)."""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        try:
            os.chmod(tmp, 0o644)
        except Exception:
            pass
        os.replace(tmp, path)
    except Exception:
        pass
