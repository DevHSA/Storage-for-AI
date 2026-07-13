"""LLMServingSim — live metrics dashboard (Streamlit).

Reads the JSON snapshot the simulator writes when run with ``--dashboard``
(default ``outputs/dashboard/live.json``) and renders a live, auto-refreshing
view of throughput, request/token/batch counters, per-tier memory occupancy
(pooled vs per-device, with the per-device split), prefix-cache hits, and
latency percentiles.

Run it (on the host, from the repo root)::

    pip install streamlit
    streamlit run serving/dashboard/app.py

Then start a sim with ``--dashboard`` (in the container) and watch it update.
Point at a different snapshot via the sidebar or ``LLMSS_DASHBOARD_FILE``.
"""

import json
import os
import time

import streamlit as st

# --------------------------------------------------------------------------- #
#  Config / helpers
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_FILE = os.environ.get(
    "LLMSS_DASHBOARD_FILE", os.path.join(_REPO_ROOT, "outputs", "dashboard", "live.json"))

_STATUS_BADGE = {
    "starting": ("🟡", "starting"),
    "running": ("🟢", "running"),
    "done": ("🔵", "done"),
}


def _fmt_bytes(b):
    b = float(b or 0)
    for unit, div in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def _fmt_int(n):
    return f"{int(n):,}"


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "not found"
    except Exception as e:                       # partial write mid-poll, etc.
        return None, str(e)


# --------------------------------------------------------------------------- #
#  Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="LLMServingSim — live", page_icon="📊", layout="wide")

with st.sidebar:
    st.header("⚙️ Dashboard")
    metrics_file = st.text_input("Live-metrics JSON", value=_DEFAULT_FILE)
    refresh_sec = st.slider("Refresh every (s)", 0.5, 10.0, 2.0, 0.5)
    auto = st.toggle("Auto-refresh", value=True)
    st.caption("Run the sim with `--dashboard` to populate this file.")

data, err = _load(metrics_file)

st.title("📊 LLMServingSim — live metrics")

if data is None:
    st.warning(f"Waiting for metrics at `{metrics_file}` … ({err}). "
               f"Start a run with `--dashboard`.")
    if auto:
        time.sleep(refresh_sec)
        st.rerun()
    st.stop()

# ---- header: status + config banner + progress -------------------------- #
cfg = data.get("config") or {}
status = data.get("status", "?")
emoji, label = _STATUS_BADGE.get(status, ("⚪", status))
sim = data.get("sim", {})
counters = data.get("counters", {})

top = st.columns([1.2, 2.2, 1.3, 1.3])
top[0].markdown(f"### {emoji} {label}")
top[1].markdown(
    f"**Model(s):** {', '.join(cfg.get('models', []) or ['?'])}  \n"
    f"**Topology:** {cfg.get('num_nodes','?')} node(s) · "
    f"{cfg.get('num_instances','?')} instance(s) · {cfg.get('total_npu','?')} GPU(s) · "
    f"TP {cfg.get('tp_sizes','?')}")
top[2].markdown(
    f"**tier-policy:** `{cfg.get('tier_policy','?')}`  \n"
    f"**cpu-scope:** `{cfg.get('cpu_scope','?')}`  \n"
    f"**prefix-storage:** `{cfg.get('prefix_storage','?')}`")
top[3].markdown(
    f"**sim clock:** {sim.get('sim_seconds',0):.3f} s  \n"
    f"**wall:** {data.get('wall_seconds',0):.1f} s  \n"
    f"**dtype/block:** {cfg.get('dtype','?')} / {cfg.get('block_size','?')}")

done_r = counters.get("requests_finished", 0)
total_r = counters.get("requests_total", 0) or 0
st.progress(min(1.0, done_r / total_r) if total_r else (1.0 if status == "done" else 0.0),
            text=f"Requests finished {done_r} / {total_r}")

# ---- KPI tiles ----------------------------------------------------------- #
thr = data.get("throughput", {})
k = st.columns(6)
k[0].metric("Requests finished", _fmt_int(done_r), f"{counters.get('requests_running',0)} running")
k[1].metric("Waiting", _fmt_int(counters.get("requests_waiting", 0)))
k[2].metric("Prompt tokens", _fmt_int(counters.get("prompt_tokens", 0)))
k[3].metric("Decode tokens", _fmt_int(counters.get("decode_tokens", 0)))
k[4].metric("Batches", _fmt_int(counters.get("batches", 0)))
k[5].metric("Req/s", f"{thr.get('req_per_s',0):.2f}")

k2 = st.columns(6)
k2[0].metric("Prompt tok/s", f"{thr.get('prompt_tok_per_s',0):.1f}")
k2[1].metric("Decode tok/s", f"{thr.get('decode_tok_per_s',0):.1f}")
k2[2].metric("Total tok/s", f"{thr.get('total_tok_per_s',0):.1f}")
k2[3].metric("Live prompt tok/s", f"{thr.get('live_prompt_tok_per_s',0):.1f}")
k2[4].metric("Live decode tok/s", f"{thr.get('live_decode_tok_per_s',0):.1f}")
lat = data.get("latency", {})
k2[5].metric("TTFT mean (ms)", f"{lat.get('ttft',{}).get('mean',0):.2f}")

st.divider()

# ---- throughput chart + tier memory ------------------------------------- #
left, right = st.columns([1, 1])

with left:
    st.subheader("⚡ Throughput over sim time")
    hist = thr.get("history") or []
    if hist:
        import pandas as pd
        df = pd.DataFrame(hist).set_index("t_s")[["prompt_tps", "decode_tps"]]
        df.columns = ["prompt tok/s", "decode tok/s"]
        st.line_chart(df, height=260)
    else:
        st.caption("No throughput samples yet (increase run length or lower --log-interval).")

with right:
    st.subheader("🧠 Memory by tier")
    for t in data.get("tiers", []):
        scope = t.get("scope", "?")
        badge = "🌐 pooled" if scope == "pooled" else f"🧩 {scope}"
        used, cap = t.get("total_used_bytes", 0), t.get("total_cap_bytes", 0)
        pct = t.get("total_pct", 0)
        st.markdown(f"**{t['name']}** &nbsp; `{badge}` &nbsp; "
                    f"{_fmt_bytes(used)} / {_fmt_bytes(cap)} ({pct:.2f}%)")
        st.progress(min(1.0, pct / 100.0))
        devs = t.get("devices", [])
        if scope != "pooled" and len(devs) > 1:
            import pandas as pd
            ddf = pd.DataFrame([{
                "device": d["label"],
                "used": _fmt_bytes(d["used_bytes"]),
                "cap": _fmt_bytes(d["cap_bytes"]),
                "% used": round(d["pct"], 2),
            } for d in devs])
            st.dataframe(ddf, hide_index=True, use_container_width=True)

st.divider()

# ---- prefix cache + latency + NPU HBM ----------------------------------- #
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.subheader("🎯 Prefix cache")
    pc = data.get("prefix_cache", {})
    st.metric("Overall hit ratio", f"{pc.get('total_hit_ratio',0):.2f}%",
              f"{pc.get('requested_tokens',0):,} requested")
    rows = [{"tier": x["name"], "hit tokens": x["hit_tokens"],
             "hit %": round(x["hit_pct"], 2), "reload ms": round(x["reload_ms"], 2)}
            for x in pc.get("tiers", [])]
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with c2:
    st.subheader("⏱️ Latency (ms)")
    order = [("TTFT", "ttft"), ("TBT/token", "tbt"), ("ITL", "itl"), ("E2E", "e2e")]
    rows = []
    for name, key in order:
        m = lat.get(key, {})
        rows.append({"metric": name, "mean": round(m.get("mean", 0), 2),
                     "p50": round(m.get("p50", 0), 2), "p90": round(m.get("p90", 0), 2),
                     "p99": round(m.get("p99", 0), 2), "max": round(m.get("max", 0), 2)})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"over {lat.get('finished',0)} finished requests")

with c3:
    st.subheader("🖥️ NPU HBM (weight + KV)")
    hbm = data.get("npu_hbm", [])
    if hbm:
        import pandas as pd
        hdf = pd.DataFrame([{
            "instance": h["instance"], "GPUs": h["num_npus"],
            "used": _fmt_bytes(h["used_bytes"]), "cap": _fmt_bytes(h["cap_bytes"]),
            "% used": round(h["pct"], 2), "weight": _fmt_bytes(h["weight_bytes"]),
        } for h in hbm])
        st.dataframe(hdf, hide_index=True, use_container_width=True)

st.caption(f"source: `{metrics_file}` · status **{status}** · "
           f"updated {data.get('wall_seconds',0):.1f}s into the run")

# ---- auto refresh -------------------------------------------------------- #
if auto and status != "done":
    time.sleep(refresh_sec)
    st.rerun()
