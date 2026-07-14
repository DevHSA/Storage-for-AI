"""Generate a MULTI-TURN (session pause/resume, "context parking") workload.

This exercises the decode-serving use-case where a conversation's KV state is
PARKED to a lower tier while the user is idle between turns, then RELOADED to
answer the next turn. With JBOF the reload is fast (~80us); without it the state
falls to COLDSTORE (~2ms) — so the tier shows up in resumed-turn TTFT.

HOW THIS DIFFERS FROM THE OTHER WORKLOADS
-----------------------------------------
- Flat traces (example_trace / sharegpt / make_rack_fill / make_tier_demo):
  every JSONL line is ONE INDEPENDENT request; all arrival times are fixed up
  front; there is no dependency between requests. Reuse (if any) is incidental
  prefix overlap.
- This generator emits AGENTIC SESSIONS: each JSONL line is a whole SESSION with
  a `sub_requests` list (the turns). Only turn 0 arrives up front; each later
  turn is released DYNAMICALLY by the router `tool_duration_ns` after the prior
  turn finishes (the inter-turn idle gap — the "user thinking" / parking window).
  Crucially, turn t's `input_tok_ids` = turn (t-1)'s (input + output) ids + a few
  NEW user tokens, so the conversation is a GROWING SHARED PREFIX that must be
  reloaded (not recomputed) each turn. During the idle gap, other active sessions
  fill the NPU and evict this session's context down to CPU->JBOF->COLDSTORE;
  the next turn reloads it. That reload lands in the turn's TTFT (it is a prefill
  over the accumulated context), which is exactly what --session-metrics splits
  out as "resumed-turn TTFT".

Schema (matches serving/core/router.py load_requests / _load_agentic_session):
  {"session_id","arrival_time_ns","sub_requests":[
     {"input_toks","output_toks","input_tok_ids","output_tok_ids","tool_duration_ns"}, ...]}

Usage:
  python make_multiturn.py <out.jsonl> [n_sessions] [n_turns] [sys_len]
                           [user_len] [out_len] [gap_ms] [stagger_ms]
Defaults: 32 sessions x 4 turns, 512-tok system/context, 16 new user tok/turn,
          16 decode tok/turn, 500 ms idle gap, 50 ms session stagger.
"""
import json
import sys

out         = sys.argv[1]  if len(sys.argv) > 1 else "workloads/multiturn_trace.jsonl"
n_sessions  = int(sys.argv[2])   if len(sys.argv) > 2 else 32
n_turns     = int(sys.argv[3])   if len(sys.argv) > 3 else 4
sys_len     = int(sys.argv[4])   if len(sys.argv) > 4 else 512   # shared system/context prefix
user_len    = int(sys.argv[5])   if len(sys.argv) > 5 else 16    # NEW user tokens per turn
out_len     = int(sys.argv[6])   if len(sys.argv) > 6 else 16    # decoded tokens per turn
gap_ms      = float(sys.argv[7]) if len(sys.argv) > 7 else 500.0 # idle gap between turns
stagger_ms  = float(sys.argv[8]) if len(sys.argv) > 8 else 50.0  # spacing between session starts

gap_ns = int(gap_ms * 1_000_000)
# Disjoint token-id block per session so each session's KV is unique (and so a
# session's growing context is a contiguous, reusable prefix). Sized for the
# whole conversation: system prompt + every turn's (user + output) tokens.
stride = sys_len + n_turns * (user_len + out_len) + 64

with open(out, "w") as f:
    for s in range(n_sessions):
        base = 1 + s * stride
        cursor = base
        ctx = list(range(cursor, cursor + sys_len))    # turn-0 shared context
        cursor += sys_len
        subs = []
        for t in range(n_turns):
            user = list(range(cursor, cursor + user_len)); cursor += user_len
            input_ids = ctx + user                      # accumulated context + new user msg
            out_ids = list(range(cursor, cursor + out_len)); cursor += out_len
            subs.append({
                "input_toks": len(input_ids),
                "output_toks": out_len,                 # generated tokens this turn
                "input_tok_ids": input_ids,
                "output_tok_ids": out_ids,
                "tool_duration_ns": gap_ns,             # idle gap before the NEXT turn
            })
            ctx = input_ids + out_ids                   # next turn reuses this full sequence
        f.write(json.dumps({
            "session_id": f"s{s}",
            "arrival_time_ns": int(s * stagger_ms * 1_000_000),
            "sub_requests": subs,
        }) + "\n")

total_turns = n_sessions * n_turns
print(f"wrote {n_sessions} sessions x {n_turns} turns = {total_turns} requests to {out}\n"
      f"  system/context prefix {sys_len} tok; +{user_len} user & {out_len} decode per turn;\n"
      f"  idle gap {gap_ms} ms between turns; session stagger {stagger_ms} ms.\n"
      f"  turn t input = turn (t-1) (input+output) + {user_len} new tokens  (growing reusable prefix).")
