"""
Print a readable summary of a run log (works on incomplete runs too).

    python -m evaluation.summarize_run runs\\run_<id>.jsonl

For each vehicle, every proposal is traced through the chain
    observation (seq, age) -> proposal -> gate decision -> command result -> feedback
using request_id and observation_id, so you can see exactly what the agent saw
when it asked for each command, what it was told afterwards (feedback code), and
where the drone really ended up (ground truth, evaluation side only).
"""
import sys

from evaluation.recorder import read_run


def summarize(path) -> str:
    log = read_run(path)
    ev = log.events
    out = []
    start = ev[0] if ev and ev[0].get("kind") == "run_start" else {}
    git = start.get("git") or {}
    out.append(f"run {start.get('run_id', '?')}  commit {str(git.get('commit'))[:10]}"
               f"{' (dirty)' if git.get('dirty') else ''}")
    status = (f"complete, outcome {log.outcome}" if log.complete
              else "INCOMPLETE (no run_end: crashed or interrupted)")
    if log.truncated_tail:
        status += ", last line cut off"
    out.append(f"{len(ev)} events, {status}")
    pipe = (start.get("config") or {}).get("pipeline")
    if pipe:
        out.append(f"pipeline: assessor={pipe.get('trust_assessor')} gate={pipe.get('action_gate')} "
                   f"hooks={pipe.get('attack_hooks')}")

    obs = {e["observation"]["observation_id"]: e for e in ev if e.get("kind") == "observation"}
    steps = {e["request_id"]: e["step"] for e in ev if e.get("kind") == "agent_step"}
    # Attach each event to the proposal it belongs to, in log order: a replayed
    # request_id is a NEW proposal (rejected), not the original one again.
    chains, open_ = [], {}
    for e in ev:
        k = e.get("kind")
        if k == "proposal":
            open_[(e.get("vehicle_id"), e.get("request_id"))] = c = {"proposal": e}
            chains.append(c)
            continue
        rid = (e.get("decision") or e.get("result") or e.get("feedback") or e).get("request_id")
        c = open_.get((e.get("vehicle_id"), rid))
        if c is not None and k in ("port_rejected", "gate_decision", "gate_error", "command_result",
                                   "command_cancelled", "command_error", "feedback"):
            c.setdefault(k, e)

    vehicles = sorted({e.get("vehicle_id") for e in ev
                       if e.get("kind") in ("proposal", "llm_session", "agent_end")
                       and e.get("vehicle_id")})
    for vid in vehicles:
        out.append(f"\n{vid}")
        mine = [e for e in ev if e.get("vehicle_id") == vid]
        n_obs = sum(1 for e in mine if e.get("kind") == "observation")
        n_tamp = sum(1 for e in mine if e.get("kind") == "observation" and e.get("tampered_by"))
        n_drop = sum(1 for e in mine if e.get("kind") == "observation_dropped")
        n_err = sum(1 for e in mine if e.get("kind") == "observation_error")
        rate = [e.get("rate_hz") for e in mine if e.get("kind") == "sampling"]
        out.append(f"  observations: {n_obs} delivered ({n_tamp} tampered), {n_drop} dropped, "
                   f"{n_err} errors" + (f"; sampled at {rate[0]:g} Hz" if rate else "; on demand"))
        for c in chains:
            e = c["proposal"]
            if e.get("vehicle_id") != vid:
                continue
            rid = e.get("request_id")
            o = obs.get(e.get("observation_id"))
            if o and o.get("vehicle_id") != vid:
                seq, o = f"?(a {o.get('vehicle_id')} observation)", None
            else:
                seq = o["observation"]["sequence"] if o else "?"
            tamper = f" TAMPERED by {o['tampered_by']}" if o and o.get("tampered_by") else ""
            age = e.get("observation_age_s")
            age_txt = f" ({age:.2f}s old)" if isinstance(age, (int, float)) else ""
            a = e.get("action")
            label = steps.get(rid) or (a.get("action_type", "?") if isinstance(a, dict) else "?")
            line = f"  {label:<9} obs#{seq}{age_txt}{tamper} -> "
            fb = (c.get("feedback") or {}).get("feedback")
            if "port_rejected" in c:
                rj = c["port_rejected"]
                line += f"PORT REJECTED [{rj.get('code', '?')}]: {rj['reason']}"
            else:
                d = (c.get("gate_decision") or {}).get("decision")
                line += (f"gate {d['outcome']}{'' if d['assessed'] else ' (unassessed)'} -> " if d
                         else "gate ERROR -> " if "gate_error" in c else "no gate decision -> ")
                r = (c.get("command_result") or {}).get("result")
                if r:
                    line += f"{r['status']} {r['elapsed_s']:.1f}s"
                    if r.get("corrections"):
                        line += f" [{r['corrections']} corr]"
                    ft = r.get("final_truth")
                    if ft:
                        line += f"  truth N={ft['north_m']:.1f} E={ft['east_m']:.1f} alt={ft['altitude_m']:.1f}"
                elif "command_cancelled" in c:
                    line += "CANCELLED"
                elif "command_error" in c:
                    line += f"ERROR {c['command_error'].get('error')}"
                elif fb:
                    line += f"not executed [{fb['code']}]"
                else:
                    line += "no result (run ended?)"
                if r and fb:
                    line += f"  -> told: {fb['code']}"
            out.append(line)
        sess = [x for x in mine if x.get("kind") == "llm_session"]
        if sess:
            turns = [x for x in mine if x.get("kind") == "llm_turn"]
            lat = [t["latency_s"] for t in turns if isinstance(t.get("latency_s"), (int, float))]
            tok_in = sum((t.get("usage") or {}).get("prompt_token_count") or 0 for t in turns)
            tok_out = sum((t.get("usage") or {}).get("candidates_token_count") or 0 for t in turns)
            errs = sum(1 for x in mine if x.get("kind") == "llm_error")
            tools = [x for x in mine if x.get("kind") == "tool_call"]
            out.append(f"  llm: {sess[0].get('provider')}:{sess[0].get('model')}, {len(turns)} turns"
                       + (f", latency mean {sum(lat) / len(lat):.1f}s max {max(lat):.1f}s" if lat else "")
                       + f", tokens in {tok_in} out {tok_out}, {len(tools)} tool calls, "
                       f"{errs} model errors")
            seen_err = {}
            for x in mine:
                if x.get("kind") == "llm_error":
                    key = f"{x.get('error')}: {str(x.get('detail', ''))[:120]}"
                    seen_err[key] = seen_err.get(key, 0) + 1
            for k, n in seen_err.items():
                out.append(f"  ! model error x{n}: {k}")
            for x in mine:
                if x.get("kind") == "agent_comment":
                    out.append(f"  comment: {x.get('text')}")
        end = [x for x in ev if x.get("kind") == "agent_end" and x.get("vehicle_id") == vid]
        if end:
            out.append(f"  agent: {end[-1]['ended']}")
        sd = [x for x in ev if x.get("kind") == "shutdown" and x.get("vehicle_id") == vid]
        if sd:
            rep = sd[-1]["report"]
            out.append(f"  shutdown: {'landed+disarmed' if isinstance(rep, dict) and rep.get('disarmed') else rep}")

    sep = [x for x in ev if x.get("kind") == "separation"]
    if sep:
        s = sep[-1]
        out.append(f"\nseparation: {s['status'].upper()}")
        for p in s.get("pairs", []):
            m = p.get("min_3d_m")
            out.append(f"  {'-'.join(p['pair'])}: min {m:.2f} m during '{p.get('phase_at_min')}'"
                       if m is not None else f"  {'-'.join(p['pair'])}: no samples")
        for x in s.get("violations", []) + s.get("coverage_problems", []):
            out.append(f"  ! {x}")
    out.append(f"collisions recorded: {sum(1 for x in ev if x.get('kind') == 'collision')}")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    print(summarize(sys.argv[1]))
