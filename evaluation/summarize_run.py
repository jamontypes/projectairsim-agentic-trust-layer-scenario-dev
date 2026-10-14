"""
Print a readable summary of a run log (works on incomplete runs too).

    python -m evaluation.summarize_run runs\\run_<id>.jsonl

For each vehicle, every proposal is traced through the chain
    observation (seq) -> proposal -> gate decision -> command result
using request_id and observation_id, so you can see exactly what the agent saw
when it asked for each command, and where the drone really ended up.
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
    decisions = {e["decision"]["request_id"]: e["decision"] for e in ev if e.get("kind") == "gate_decision"}
    results = {e["result"]["request_id"]: e["result"] for e in ev if e.get("kind") == "command_result"}
    rejected = {e["request_id"]: e for e in ev if e.get("kind") == "port_rejected"}
    cancelled = {e["request_id"]: e for e in ev if e.get("kind") == "command_cancelled"}
    corrections = {}
    for e in ev:
        if e.get("kind") == "correction":
            corrections.setdefault(e["request_id"], []).append(e)
    steps = {e["request_id"]: e["step"] for e in ev if e.get("kind") == "agent_step"}

    vehicles = sorted({e.get("vehicle_id") for e in ev if e.get("kind") == "proposal"})
    for vid in vehicles:
        out.append(f"\n{vid}")
        for e in ev:
            if e.get("kind") != "proposal" or e.get("vehicle_id") != vid:
                continue
            rid = e["request_id"]
            o = obs.get(e.get("observation_id"))
            seq = o["observation"]["sequence"] if o else "?"
            tamper = f" TAMPERED by {o['tampered_by']}" if o and o.get("tampered_by") else ""
            a = e["action"]
            line = f"  {steps.get(rid, a['action_type']):<9} obs#{seq}{tamper} -> "
            if rid in rejected:
                line += f"PORT REJECTED: {rejected[rid]['reason']}"
            else:
                d = decisions.get(rid)
                line += f"gate {d['outcome']}{'' if d and d['assessed'] else ' (unassessed)'} -> " if d else "no gate decision -> "
                r = results.get(rid)
                if r:
                    line += f"{r['status']} {r['elapsed_s']:.1f}s"
                    if r.get("corrections"):
                        line += f" [{r['corrections']} corr]"
                    ft = r.get("final_truth")
                    if ft:
                        line += f"  truth N={ft['north_m']:.1f} E={ft['east_m']:.1f} alt={ft['altitude_m']:.1f}"
                elif rid in cancelled:
                    line += "CANCELLED"
                else:
                    line += "no result (run ended?)"
            out.append(line)
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
