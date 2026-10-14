"""
Per-vehicle action port: the ONLY way an agent can act.

    agent -> port.submit(action, based_on=observation)
          -> ownership + provenance checks  (port_rejected: never reaches gate or executor)
          -> action-risk gate               (gate_decision; BypassGate for now)
          -> executor                       (command_result)

Checks, in order (any failure -> CommandResult REFUSED, sent_to_simulator=False):
    1. the action is for THIS port's vehicle (ownership is bound to the port);
    2. `based_on` is an Observation this vehicle's feed actually delivered
       (every proposal is tied to the observation used to make it);
    3. the request_id has not been submitted before (no replayed proposals).

Every step is recorded with the request_id and observation_id, so a command
result can always be traced back to what the agent saw when it proposed it.
If the caller is cancelled during execution, a command_cancelled event is
recorded (the executor hovers the drone) and the cancellation propagates.
"""
import asyncio
import time

from gate.action_gate import BypassGate
from models.action import CommandResult, CommandStatus, ProposedAction


class ActionPort:
    def __init__(self, vehicle_id: str, executor, feed, gate=None, recorder=None,
                 clock=time.monotonic):
        if feed.vehicle_id != vehicle_id:
            raise ValueError(f"feed for {feed.vehicle_id!r} cannot back the port for {vehicle_id!r}")
        self.vehicle_id = vehicle_id
        self._executor = executor
        self._feed = feed
        self._gate = gate or BypassGate()
        self._recorder = recorder
        self._clock = clock
        self._seen_requests: set[str] = set()

    async def submit(self, action: ProposedAction, based_on) -> CommandResult:
        obs_id = getattr(based_on, "observation_id", None)
        self._record("proposal", request_id=action.request_id, observation_id=obs_id,
                     action=action.model_dump(mode="json"))

        problem = None
        if action.vehicle_id != self.vehicle_id:
            problem = (f"port for {self.vehicle_id} cannot submit an action for "
                       f"{action.vehicle_id}")
        elif not self._feed.issued(based_on):
            problem = ("proposal is not based on an observation delivered by "
                       f"{self.vehicle_id}'s feed")
        elif action.request_id in self._seen_requests:
            problem = f"request_id {action.request_id} was already submitted"
        if problem:
            self._record("port_rejected", request_id=action.request_id,
                         observation_id=obs_id, reason=problem)
            return self._refused(action, f"action port: {problem}")
        self._seen_requests.add(action.request_id)

        try:
            decision = self._gate.decide(action, based_on)
        except Exception as err:   # a broken gate blocks; it never waves actions through
            self._record("gate_error", request_id=action.request_id, observation_id=obs_id,
                         error=f"{type(err).__name__}: {err}")
            return self._refused(action, f"gate error: {type(err).__name__}: {err}")
        self._record("gate_decision", decision=decision.model_dump(mode="json"))
        if not decision.allows_execution:
            return self._refused(action, f"gate {decision.gate}: {decision.reason}")

        started = self._clock()
        try:
            result = await self._executor.execute(action)
        except BaseException as err:
            kind = "command_cancelled" if isinstance(err, asyncio.CancelledError) else "command_error"
            self._record(kind, request_id=action.request_id, observation_id=obs_id,
                         elapsed_s=round(self._clock() - started, 3),
                         error=f"{type(err).__name__}: {err}")
            raise
        self._record("command_result", observation_id=obs_id, decision_id=decision.decision_id,
                     result=summarize_result(result))
        return result

    # ------------------------------------------------------------ internal
    def _refused(self, action, reason) -> CommandResult:
        return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                             action_type=action.action_type, status=CommandStatus.REFUSED,
                             reason=reason, sent_to_simulator=False)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)


def summarize_result(r: CommandResult) -> dict:
    """CommandResult without the full snapshots, plus where the drone really ended up
    (ground truth from the executor's final snapshot; evaluation side)."""
    d = r.model_dump(mode="json", exclude={"start_snapshot", "final_snapshot"})
    for name in ("start_snapshot", "final_snapshot"):
        s = getattr(r, name)
        if s is not None:
            d[name.replace("snapshot", "truth")] = {
                "sim_time_ns": s.sim_time_ns,
                "north_m": round(s.position_ned_m.x, 3), "east_m": round(s.position_ned_m.y, 3),
                "altitude_m": round(s.altitude_local_m, 3), "heading_deg": round(s.heading_deg, 2),
                "validation_status": s.validation_status.value,
            }
    return d
