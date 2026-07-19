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


def _kv_unit(m):
    """KV-cache granularity for one MemoryModel.

    * ``bytes_per_token_per_gpu`` = ``get_kv(1)`` = ``2 * kv_dim * n_layer *
      kv_dtype_bytes / tp_size`` -- the KV bytes one token adds on a single GPU
      (K and V, all layers; sharded across the tp_size ranks).
    * A KV *block* (vLLM "page") is ``block_size`` tokens and is the SMALLEST
      unit the NPU allocates/evicts, so ``block_bytes_per_gpu`` is the smallest
      KV-cache unit on the NPU.
    * The deep tiers (CPU / JBOF / COLDSTORE) store the WHOLE context, not a
      single rank's shard, so their per-token / per-block size is the per-GPU
      figure times ``tp_size`` (``*_full``)."""
    per_tok_gpu = _safe_call(lambda: int(m.get_kv(1)), 0)
    tp = _safe_call(lambda: int(m.num_npus), 1) or 1
    bs = _safe_call(lambda: int(m.block_size), 0)
    return {
        "model": getattr(m, "model", ""),
        "tp_size": tp,
        "block_size": bs,                                # tokens per block (page)
        "kv_dtype_bytes": _safe_call(lambda: int(m.kv_fp), 0),
        "kv_heads": _safe_call(lambda: int(m.kv_head), 0),
        "head_dim": _safe_call(lambda: int(m.head_dim), 0),
        "n_layer": _safe_call(lambda: int(m.n_layer), 0),
        "bytes_per_token_per_gpu": per_tok_gpu,          # KV bytes / token / GPU
        "bytes_per_token_full": per_tok_gpu * tp,        # full-context / token (deep tiers)
        "block_bytes_per_gpu": per_tok_gpu * bs,         # smallest NPU KV unit / GPU
        "block_bytes_full": per_tok_gpu * bs * tp,       # one block, whole context
    }


def _percentiles(values_ns):
    vals = [v for v in values_ns if v is not None and v >= 0]
    if not vals or np is None:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "n": len(vals)}
    a = np.array(vals, dtype=float) / 1_000_000.0        # ns -> ms
    return {"mean": float(np.mean(a)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)), "n": len(vals)}


TRAY_GPUS = 4   # a Vera Rubin tray = 4 GPUs (2 TP=2 superchips); dashboard-only grouping


def _assign_trays(schedulers):
    """Group consecutive instances WITHIN a node into trays of up to TRAY_GPUS
    GPUs (a Vera Rubin tray = 4 GPUs = 2 superchips). Purely a reporting rollup —
    the simulator still runs the underlying TP=N instances unchanged."""
    by_node = {}
    for s in schedulers:
        by_node.setdefault(getattr(s, "node_id", 0), []).append(s)
    trays, tray_idx = [], 0
    for node in sorted(by_node):
        ss = sorted(by_node[node], key=lambda s: getattr(s, "instance_id", 0))
        cur, acc = None, 0
        for s in ss:
            g = getattr(s, "num_npus", 1)
            if cur is None or acc + g > TRAY_GPUS:
                cur = {"tray": tray_idx, "node": node, "scheds": []}
                trays.append(cur)
                tray_idx += 1
                acc = 0
            cur["scheds"].append(s)
            acc += g
    return trays


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
        return f"inst {insts[0]}"      # one cache per instance (a TP=N superchip)
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
            seen[k] = {"cache": c, "insts": [], "nodes": set(), "gpus": 0}
            order.append(k)
        seen[k]["insts"].append(getattr(s, "instance_id", -1))
        seen[k]["nodes"].add(getattr(s, "node_id", -1))
        seen[k]["gpus"] += getattr(s, "num_npus", 1)
    if not order:
        return None
    devices, total_used, total_cap = [], 0, 0
    for k in order:
        rec = seen[k]
        c = rec["cache"]
        used = _safe_call(c.total_memory_usage, 0)
        cap = getattr(c, "capacity", 0) or 0
        devices.append({"label": _device_label(sorted(rec["insts"]), sorted(rec["nodes"]), scope),
                        "gpus": rec["gpus"], "used_bytes": used, "cap_bytes": cap, "pct": _pct(used, cap)})
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
        _ku = _kv_unit(m)
        instances_info.append({
            "instance": getattr(s, "instance_id", -1), "node": getattr(s, "node_id", -1),
            "num_npus": getattr(s, "num_npus", 1), "running": running_i, "waiting": waiting_i,
            "hbm_pct": _pct(used, cap), "npu_kv_bytes": kv,
            "hit_ratio": (hit_sum / req_i * 100.0) if req_i else 0.0,
            "kv_per_token_bytes": _ku["bytes_per_token_per_gpu"],   # per GPU/rank
            "kv_block_bytes": _ku["block_bytes_per_gpu"],           # smallest NPU KV unit
            "kv_block_size": _ku["block_size"],
        })
        npu_hbm.append({
            "instance": getattr(s, "instance_id", -1), "node": getattr(s, "node_id", -1),
            "num_npus": getattr(s, "num_npus", 1), "used_bytes": used, "cap_bytes": cap,
            "pct": _pct(used, cap), "weight_bytes": getattr(m, "weight", 0),
            "kv_per_token_bytes": _ku["bytes_per_token_per_gpu"],
            "kv_block_bytes": _ku["block_bytes_per_gpu"],
        })

    # ---- tray rollup (dashboard-only grouping: <=4 GPUs / tray) --------- #
    trays_info = []
    for t in _assign_trays(schedulers):
        ss = t["scheds"]
        hbm_used = sum(max(getattr(s.memory, "npu_used", 0), getattr(s.memory, "weight", 0)) for s in ss)
        hbm_cap = sum(getattr(s.memory, "npu_mem", 0) for s in ss)
        cpu_seen = {}
        for s in ss:
            c = getattr(s.memory, "second_tier_prefix_cache", None)
            if c is not None:
                cpu_seen[id(c)] = c
        cpu_used = sum(_safe_call(c.total_memory_usage, 0) for c in cpu_seen.values())
        cpu_cap = sum((getattr(c, "capacity", 0) or 0) for c in cpu_seen.values())
        hrs = []
        for s in ss:
            req_i, hits_i = _safe_call(s.memory.tier_hit_report, (0, {}))
            hrs.append((sum((hits_i or {}).values()) / req_i * 100.0) if req_i else 0.0)
        trays_info.append({
            "tray": t["tray"], "node": t["node"],
            "num_gpus": sum(getattr(s, "num_npus", 1) for s in ss),
            "num_cpus": len(cpu_seen), "num_instances": len(ss),
            "instances": [getattr(s, "instance_id", -1) for s in ss],
            "hbm_used_bytes": hbm_used, "hbm_cap_bytes": hbm_cap, "hbm_pct": _pct(hbm_used, hbm_cap),
            "npu_kv_bytes": sum(_safe_call(lambda s=s: s.memory.npu_prefix_cache.total_memory_usage(), 0) for s in ss),
            "cpu_used_bytes": cpu_used, "cpu_cap_bytes": cpu_cap, "cpu_pct": _pct(cpu_used, cpu_cap),
            "running": sum(_safe_call(lambda s=s: sum(len(b.requests) for b in s.inflight), 0) for s in ss),
            "waiting": sum(_safe_call(lambda s=s: len([r for r in s.request if r.arrival <= clock_ns]), 0) for s in ss),
            "hit_ratio": (sum(hrs) / len(hrs)) if hrs else 0.0,
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

    # ---- session / resume latency (multi-turn "context parking") -------- #
    # Emitted only when the workload carries agentic sessions (records with
    # sub_requests, so requests are tagged with a sub_request_index). Splits
    # TTFT into first-turn (cold) vs resumed-turn (warm — the conversation's
    # parked context is reloaded from a lower tier before answering), so the
    # pause/resume benefit (JBOF-fast vs COLDSTORE-slow reload) is visible.
    sessions = None
    _sess_done = [r for s in schedulers for r in s.done if getattr(r, "session_id", None) is not None]
    if _sess_done:
        _first = [r.ttft for r in _sess_done
                  if getattr(r, "sub_request_index", None) == 0 and r.ttft is not None and r.ttft >= 0]
        _resumed = [r.ttft for r in _sess_done
                    if (getattr(r, "sub_request_index", None) or 0) >= 1 and r.ttft is not None and r.ttft >= 0]
        _fp, _rp = _percentiles(_first), _percentiles(_resumed)
        sessions = {
            "n_sessions": len({r.session_id for r in _sess_done}),
            "first_turn": _fp, "resumed_turn": _rp,
            "resume_overhead_ms": (_rp["mean"] - _fp["mean"]) if (_first and _resumed) else 0.0,
        }

    # ---- KV-cache unit sizing (per-token & per-block granularity) ------- #
    # Reported once for the run (all instances of the same model share it) with
    # a ``uniform`` flag; heterogeneous instances also carry their own numbers
    # in ``instances[].kv_*``.
    kv_unit = None
    _kus = [_kv_unit(s.memory) for s in schedulers]
    if _kus:
        base = dict(_kus[0])
        base["uniform"] = all(
            u["bytes_per_token_per_gpu"] == _kus[0]["bytes_per_token_per_gpu"]
            and u["block_size"] == _kus[0]["block_size"] for u in _kus)
        base["n_instances"] = len(_kus)
        kv_unit = base

    # ---- timeline history (append this interval's point) ---------------- #
    if record_point and history is not None:
        t_now = t_s if t_s is not None else sim_seconds
        t_prev = history[-1]["t_s"] if history else 0.0
        # TTFT binned by FIRST-TOKEN time (not by completion). r.ttft (ns) is set
        # once, at the first output token, to (first_token_clock - arrival); so the
        # first token happened at sim-time (arrival + r.ttft)/freq. Average the TTFT
        # of every request whose first token landed in THIS interval (t_prev, t_now].
        # A request that has produced its first token is in a running (inflight)
        # batch or in done, and ttft >= 0; still-waiting requests have ttft == -1.
        _ft = []
        for s in schedulers:
            _cands = list(getattr(s, "done", []))
            for b in getattr(s, "inflight", []):
                _cands.extend(getattr(b, "requests", []))
            for r in _cands:
                tt = getattr(r, "ttft", -1)
                if tt is None or tt < 0:
                    continue
                ft_s = ((getattr(r, "arrival", 0) + tt) / freq) if freq else 0.0
                if t_prev < ft_s <= t_now:
                    _ft.append(tt / 1e6)   # ns -> ms
        ttft_ft = (sum(_ft) / len(_ft)) if _ft else None
        history.append({
            "t_s": t_now,
            "prompt_tps": live_prompt_tps, "decode_tps": live_gen_tps,
            "mem": {t["name"]: t["total_used_bytes"] for t in tiers},
            "ttft": latency["ttft"]["mean"], "tbt": latency["tbt"]["mean"],
            # NEW: mean TTFT of requests whose first token was produced in this
            # interval -> the live TTFT curve (flat/low), vs "ttft" which is the
            # cumulative mean over only FINISHED requests (0 until the first
            # completion, then jumps).
            "ttft_ft": ttft_ft,
        })
    # peak + idle-robust "active" (busy-interval) throughput from the timeline.
    # total_*_tok_per_s divide by the FULL sim time (makespan) -> the standard
    # average, but diluted by idle gaps. active_* averages only the intervals
    # that actually produced tokens, so it reflects the true serving rate even
    # for bursty / gapped workloads. peak_* is the best single interval.
    _tot_series = [(float(p.get("prompt_tps", 0) or 0) + float(p.get("decode_tps", 0) or 0))
                   for p in (history or [])]
    _active = [x for x in _tot_series if x > 0]
    thr = {
        "req_per_s": (req_cnt / sim_seconds) if sim_seconds else 0.0,
        "prompt_tok_per_s": (total_prompt / sim_seconds) if sim_seconds else 0.0,
        "decode_tok_per_s": (total_gen / sim_seconds) if sim_seconds else 0.0,
        "total_tok_per_s": ((total_prompt + total_gen) / sim_seconds) if sim_seconds else 0.0,
        "peak_total_tok_per_s": (max(_tot_series) if _tot_series else 0.0),
        "active_total_tok_per_s": ((sum(_active) / len(_active)) if _active else 0.0),
        "live_prompt_tok_per_s": live_prompt_tps, "live_decode_tok_per_s": live_gen_tps,
        "history": list(history or []),
    }

    return {
        "schema": 2, "status": status, "wall_seconds": round(wall_seconds, 3),
        "sim": {"clock_ns": clock_ns, "sim_seconds": sim_seconds},
        "config": config, "counters": counters, "throughput": thr,
        "tiers": tiers, "trays": trays_info, "instances": instances_info, "npu_hbm": npu_hbm,
        "prefix_cache": prefix_cache, "latency": latency, "sessions": sessions,
        "kv_unit": kv_unit,
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
