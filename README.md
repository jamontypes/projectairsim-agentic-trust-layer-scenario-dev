# Project AirSim Agentic Trust Layer

Port of [uas-agentic-trust-layer](https://github.com/krolva/uas-agentic-trust-layer)
from BlueSky to [Project AirSim](https://github.com/iamaisim/ProjectAirSim).
An agent proposes drone actions; deterministic layers validate telemetry and
gate actions before anything reaches the simulator.

```
                 Simulator  <-- executor (flight control, ground truth) <--+
                     |                                                    |
          adapter (ground truth snapshot)                          action-risk gate
                     |                                              (BypassGate now)
      ObservationFeed per drone                                           |
  strip verdict -> attack hooks -> trust assessor --> Agent --> ActionPort per drone
  (sampled at a fixed rate,       (none yet)    (by observation_id)   |
   history kept per drone)                 <-- ActionFeedback (no ground truth)

  EvaluationRecorder (runs/*.jsonl): every observation, proposal, decision,
  result, correction, cancellation, shutdown, ground-truth pose and collision
```

## Who may see what

| Component | Ground truth | Its own drone's delivered observations | Raw payload | Clean (pre-attack) payload / attack details | Other drones' observations |
|---|---|---|---|---|---|
| Agent | No (feedback is ActionFeedback: status + fixed codes, no positions or measured errors) | Yes (its own deep copies) | Yes | No | No (explicit peer reports later) |
| Trust / provenance assessor | No | Yes (+ previous one) | Yes | No | No (peer reports later) |
| Action-risk gate | No | Yes, the feed's protected record (not the agent's copy), with its trust assessment | Yes | No | No (peer reports later) |
| Executor | **Yes** (preflight, settle, corrections) | No | No | No | No |
| Recorder / separation | Yes (full CommandResult, snapshots, reasons) | Yes | Yes | Yes | Yes |

The executor deliberately uses simulator ground truth: it plays the role of a
reliable flight controller. Consequence for experiments: when an agent's
observations are corrupted, its commands are still flown accurately, so an
attack shows up as the agent commanding the WRONG target, executed precisely
(see `test_attack_misleads_the_agent_and_executor_flies_the_wrong_target_accurately`).
Experiments that attack the flight controller's own sensing would need a
different executor.

How an agent refers to what it saw: proposals carry an `observation_id`. The
port resolves it through the agent's own feed to the feed's protected record and
rejects unknown ids (including other drones'), altered observation objects,
expired observations (`max_observation_age_s`, checked again after the gate),
observations received before the previous command ended, concurrent submissions
and replayed request ids. Observations are sampled at a fixed rate
(`ObservationFeed.sample`), independent of how often the agent decides; the
evaluator's 20 Hz ground-truth samples are not observations.

Placeholders are explicit, never silent: with no assessor every observation is
`not_assessed`, and with no gate every action is `bypassed` (`assessed=false`).
A run with `bypassed` decisions is an unprotected run, not one where actions
passed a risk check. Conflict decisions will later need permitted traffic
information (peer reports) on a modeled channel; the isolated missions here
don't.

## Status

| Pass | Scope | State |
|---|---|---|
| 1-3 | Connect, fly, explore telemetry | done |
| 4 | Telemetry adapter, snapshot model, validation | done |
| 4.1 | Review fixes: robust callbacks, reconnect reset, landed state, request lock, vendored config | done |
| 4.2 | Client pin 1.0.2, operational ground state, sim-progress staleness, action/result contract, completion predicates | done |
| 4.3 | Ground-state hardening: raw LANDED must be still, support-surface normals only, latch release on movement, reset on clock reset; strict action fields | done |
| 5.1 | Executor: `rotate_to_heading`, `change_altitude` (preflight, bounded task + settle waits, hover fallback, per-vehicle lock, cancellation) | done |
| 5.2 | `move_to_position` with bounded corrections (1.0 m tolerance: Simple Flight stops within ~0.5-0.75 m by design); three-drone scene and concurrent multi-drone flight | done |
| 5.3 | Pairwise separation + collision assessment; executor catch-all after dispatch; missions cancelled and awaited before shutdown | done |
| 5.4 | Cancellation survives repeated Ctrl+C and cancels each drone's task once (takeoff included); separation counts only fresh, in-sync samples and reports INCONCLUSIVE on poor coverage | done |
| 5.5 | Observation feed (verdict stripped, attack hooks before assessment, receiver-stamped ids), vehicle-bound action port (ownership, observation linkage, no replays, explicit gate bypass), JSONL evaluation recorder written during the run; three-drone demo runs through agents, with a sync point before descending; agents capture home on the ground before takeoff; altitude tolerance 0.5 m (Simple Flight accuracy) | done |
| 5.6a | Interface hardening for external (LLM) agents: agent-facing `ActionFeedback` (no ground truth, fixed codes; full `CommandResult` evaluation side only, executor `reason_code`), proposals by `observation_id` resolved to frozen, feed-protected records, freshness and supersession checks, one command in flight per drone, fixed-rate observation sampling with history | done (offline) |
| 5.6b | Gemini pilot for one drone through the same feed/port (provider-replaceable) | next |
| later | Trust assessor (scores + per-check results), action-risk gate with escalation, peer broadcasts, attack scenarios; `move_along_track`, `takeoff`, `land` | planned |

## Setup (Windows)

1. Download the Project AirSim **v1.0.1** Blocks environment and unzip it. Its release notes pair it with the Python client `projectairsim==1.0.2`.
2. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Start the simulator and wait for the Blocks window:
   ```powershell
   <path>\Blocks-Windows-1.0.1\Blocks.exe -windowed -ResX=1280 -ResY=720
   ```

Scene and robot configs are vendored in `sim_config/` (copied from the
Project AirSim v1.0.1 samples), so no sibling checkout is required.

## Running

All commands from the repo root.

| Command | Needs sim | What it checks |
|---|---|---|
| `python -m pytest tests` | no | Adapter, validation, ground detector, action contract, completion predicates, fail-safe |
| `python smoke_projectairsim_connection.py` | yes | Connect / disconnect |
| `python test_drone_creation.py` | yes | Scene load, drone creation, one telemetry read |
| `python test_navigation.py` | yes | Takeoff, rotate nose, move along tracks, return, land |
| `python test_adapter_live.py [--fly]` | yes | Snapshots at 2 Hz, optionally during a flight |
| `python test_executor_live.py` | yes | Executor: takeoff (setup), rotate +90 deg, climb 2 m, safe shutdown |
| `python test_multi_drone_live.py` | yes | Three drones (scene_three_drones.jsonc), each flown by a scripted agent through its own feed and port: climb to separate layers, fan out, turn, return, (wait for all) descend, land. Writes `runs/run_<id>.jsonl`. PASS also requires >= 2 m separation between every pair, no drone-to-drone collisions or impacts, and adequate telemetry coverage (otherwise the separation verdict is INCONCLUSIVE, which fails the run) |
| `python -m evaluation.summarize_run runs\run_<id>.jsonl` | no | Per-drone observation stream counts and chain observation (age) -> proposal -> gate decision -> result -> feedback code, shutdowns, separation; works on incomplete runs |
| `python telemetry_explorer.py` | yes | Dumps every non-camera topic to `telemetry_dump.json` |

Every script that arms the drone calls `utils.flight_safety.safe_shutdown()`
in a `finally` block: hover, land, and disarm only once the drone is confirmed
on the ground. If landing fails, times out or can't be confirmed, it does NOT
disarm (that would drop the drone); it reports UNRESOLVED and leaves the drone
armed and hovering. Every step is time-bounded end to end.

## Conventions

- **Frame:** NED (x north, y east, z down). Positive `vertical_speed_mps` = climbing.
- **Units:** SI inside the system; convert to ft / kt only for display.
- **Heading vs. track:** `heading_deg` is where the nose points (yaw);
  `track_deg` is where the drone moves. A multirotor can differ by any angle.
- **Ground state:** `landed_state` is Project AirSim's raw value and can stay
  `flying` for ~12 s after touchdown. `ground_state` is the operational
  decision. It requires the drone to be still, plus either raw `landed` or a
  slow contact with an upward-facing surface (collision normal z <= -0.7, so
  walls don't count) followed by >= 1 s of stillness near that contact.
  Decisions use `ground_state`, and only from a `valid` snapshot.
- **Freshness:** `telemetry_age_ms` is time since the last pose message;
  `sim_progress_age_ms` is time since the pose timestamp last advanced. Both
  must be under the limit for `valid`, so a frozen or replayed stream goes stale.
- **Validation:** snapshots are always built, even from bad data.
  `validation_status` is `valid`, `stale` or `invalid`; only `valid` should
  ever justify a command. Warnings (collision impacts, landed-state
  inconsistencies) don't change the status.

## Layout

```
adapters/projectairsim_adapter.py   connect, cache push topics, pull kinematics/geo/landed state
models/telemetry.py                 TelemetrySnapshot contract
models/action.py                    ProposedAction (per-type parameters) and CommandResult
executor/completion.py              pure settle predicates, dwell tracker, deadline
executor/projectairsim_executor.py  command lifecycle: preflight -> dispatch -> task wait -> settle -> result
validation/telemetry_validator.py   deterministic data-quality checks
utils/flight_safety.py              fail-safe shutdown for armed-flight scripts
utils/separation.py                 pairwise separation + collision assessment (evaluation side, ground truth); PASS / FAIL / INCONCLUSIVE
utils/fleet.py                      run one coroutine per drone (cancel + await the rest on failure); PhaseBarrier
models/observation.py               Observation, TrustAssessment, GateDecision
channel/observation_feed.py         per-drone observation channel (the agent's only input)
channel/attacks.py                  in-transit attack hooks (offset, replay, drop)
channel/action_port.py              per-drone action port (the agent's only output)
trust/assessor.py                   trust assessor interface + NoTrustAssessor placeholder
gate/action_gate.py                 action-risk gate interface + BypassGate placeholder
agent/scripted_agent.py             ScriptedMissionAgent (fixed steps, no decision logic)
evaluation/recorder.py              JSONL run log written during the run; read_run()
evaluation/summarize_run.py         readable run summary
sim_config/                         scene + robot configs
tests/                              offline tests with a fake projectairsim module
```
