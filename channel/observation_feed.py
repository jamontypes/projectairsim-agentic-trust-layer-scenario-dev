"""
Per-vehicle observation channel: the ONLY way an agent learns about its drone.

    adapter snapshot (ground truth, simulator side)
        -> strip the adapter's validation verdict          (see models/observation.py)
        -> attack hooks, in order (may modify or drop)      (channel/attacks.py)
        -> receiver stamps id / channel / sequence / receipt time
        -> trust assessor (sees only what was delivered)    (trust/assessor.py)
        -> Observation handed to the agent
        -> recorder (evaluation side): delivered payload + assessment, and, only
           there, which hooks changed it and the clean payload

One feed per vehicle. A feed never serves another vehicle, and an agent holds
only its feed and its action port, never the adapter or a drone handle.

observe() returns None when there is nothing to deliver (adapter error or a
dropped message). Agents must treat None as "no information", not as "no change".

The source is simulator ground truth (Project AirSim kinematics), named in
`channel`. A sensor-based source (GPS/IMU topics) would be a different channel.
"""
import copy
import time
from collections import OrderedDict
from datetime import datetime, timezone

from channel.attacks import HookContext
from models.observation import Observation
from trust.assessor import NoTrustAssessor

# Adapter-computed verdict fields: computed on CLEAN data, so never delivered.
VERDICT_FIELDS = {"validation_status", "validation_errors", "validation_warnings"}


def _hook_name(hook) -> str:
    return getattr(hook, "name", None) or getattr(hook, "__name__", None) or type(hook).__name__


class ObservationFeed:
    def __init__(self, adapter, vehicle_id: str, hooks=(), assessor=None, recorder=None,
                 clock=time.monotonic, remember: int = 1000):
        if vehicle_id not in getattr(adapter, "vehicle_ids", [vehicle_id]):
            raise ValueError(f"unknown vehicle {vehicle_id!r}")
        self.vehicle_id = vehicle_id
        self.channel = f"projectairsim_ground_truth:{vehicle_id}"
        self._adapter = adapter
        self._hooks = list(hooks)
        self._assessor = assessor or NoTrustAssessor()
        self._recorder = recorder
        self._clock = clock
        self._seq = 0
        self._previous: Observation | None = None
        self._issued: OrderedDict[str, Observation] = OrderedDict()
        self._remember = remember

    # ------------------------------------------------------------ public
    def observe(self) -> Observation | None:
        self._seq += 1
        seq = self._seq
        try:
            snap = self._adapter.get_snapshot(self.vehicle_id)
        except Exception as err:
            self._record("observation_error", sequence=seq, error=f"{type(err).__name__}: {err}")
            return None
        received = self._clock()
        source_validation = {
            "status": snap.validation_status.value,
            "errors": list(snap.validation_errors),
        }
        clean = snap.model_dump(mode="json", exclude=VERDICT_FIELDS)

        payload = copy.deepcopy(clean)
        ctx = HookContext(vehicle_id=self.vehicle_id, sequence=seq, received_host_s=received)
        changed_by = []
        for hook in self._hooks:
            before = copy.deepcopy(payload)
            try:
                out = hook(payload, ctx)
            except Exception as err:   # a broken hook is a lost message, never a crash
                self._record("observation_dropped", sequence=seq, hook=_hook_name(hook),
                             reason=f"hook raised {type(err).__name__}: {err}",
                             source_validation=source_validation)
                return None
            if out is None:
                self._record("observation_dropped", sequence=seq, hook=_hook_name(hook),
                             reason="dropped by hook", source_validation=source_validation)
                return None
            if out != before:
                changed_by.append(_hook_name(hook))
            payload = out
        # anything a hook tried to add under a verdict name is not a verdict
        for f in VERDICT_FIELDS & set(payload):
            payload[f"claimed_{f}"] = payload.pop(f)

        obs = Observation(vehicle_id=self.vehicle_id, sequence=seq, channel=self.channel,
                          received_host_s=received, received_at_utc=datetime.now(timezone.utc),
                          payload=payload)
        try:
            obs.assessment = self._assessor.assess(obs, self._previous)
        except Exception as err:
            from models.observation import TrustAssessment, TrustStatus
            obs.assessment = TrustAssessment(
                observation_id=obs.observation_id, assessor=getattr(self._assessor, "name", "?"),
                assessed=False, status=TrustStatus.NOT_ASSESSED,
                reasons=[f"assessor raised {type(err).__name__}: {err}"])
        self._previous = obs
        self._issued[obs.observation_id] = obs
        while len(self._issued) > self._remember:
            self._issued.popitem(last=False)

        rec = {"observation": obs.model_dump(mode="json"),
               "source_validation": source_validation}   # evaluation side only
        if changed_by:
            rec["tampered_by"] = changed_by
            rec["clean_payload"] = clean
        self._record("observation", **rec)
        return obs

    def issued(self, observation) -> bool:
        """True only for the exact Observation object this feed delivered (and still
        remembers). A copy with the same id, or one built by hand, does not count."""
        return (isinstance(observation, Observation)
                and observation.vehicle_id == self.vehicle_id
                and self._issued.get(observation.observation_id) is observation)

    # ------------------------------------------------------------ internal
    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)
