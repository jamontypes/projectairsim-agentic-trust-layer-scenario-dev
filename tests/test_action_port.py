"""ActionPort ownership, provenance, gate handling and recording (no simulator)."""
import asyncio

import pytest

from channel.action_port import ActionPort
from channel.observation_feed import ObservationFeed
from models.action import ActionType, CommandResult, CommandStatus, ProposedAction
from models.observation import GateDecision, GateOutcome
from tests.channel_fakes import ListRecorder, SnapAdapter


class FakeExecutor:
    def __init__(self, block=False):
        self.calls = []
        self.block = block
        self.started = asyncio.Event() if block else None

    async def execute(self, action):
        self.calls.append(action)
        if self.block:
            self.started.set()
            await asyncio.sleep(3600)
        return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                             action_type=action.action_type, status=CommandStatus.SUCCEEDED,
                             reason="ok", sent_to_simulator=True)


def make(gate=None, block=False):
    adapter, rec = SnapAdapter(), ListRecorder()
    feed1 = ObservationFeed(adapter, "Drone1", recorder=rec)
    feed2 = ObservationFeed(adapter, "Drone2", recorder=rec)
    ex = FakeExecutor(block)
    port = ActionPort("Drone1", ex, feed1, gate=gate, recorder=rec)
    return port, feed1, feed2, ex, rec


def turn(vid="Drone1", **kw):
    return ProposedAction(vehicle_id=vid, action_type=ActionType.ROTATE_TO_HEADING,
                          heading_deg=90.0, reason="test", **kw)


def test_bypassed_action_executes_and_every_step_is_linked():
    port, feed, _, ex, rec = make()
    obs = feed.observe()
    a = turn()
    r = asyncio.run(port.submit(a, based_on=obs))
    assert r.status == CommandStatus.SUCCEEDED and ex.calls == [a]
    p, d, res = rec.of("proposal")[0], rec.of("gate_decision")[0], rec.of("command_result")[0]
    assert p["request_id"] == a.request_id and p["observation_id"] == obs.observation_id
    assert d["decision"]["outcome"] == "bypassed" and d["decision"]["assessed"] is False
    assert d["decision"]["observation_id"] == obs.observation_id
    assert res["result"]["request_id"] == a.request_id
    assert res["observation_id"] == obs.observation_id
    assert res["decision_id"] == d["decision"]["decision_id"]


def test_action_for_another_vehicle_is_rejected_before_gate_and_executor():
    port, feed, _, ex, rec = make()
    r = asyncio.run(port.submit(turn("Drone2"), based_on=feed.observe()))
    assert r.status == CommandStatus.REFUSED and not r.sent_to_simulator
    assert "cannot submit an action for Drone2" in r.reason
    assert ex.calls == [] and not rec.of("gate_decision")
    assert rec.of("port_rejected")


@pytest.mark.parametrize("which", ["other_vehicle", "copy", "none", "dict"])
def test_proposal_must_be_based_on_an_observation_from_this_vehicles_feed(which):
    port, feed, feed2, ex, rec = make()
    obs = feed.observe()
    based_on = {"other_vehicle": feed2.observe(), "copy": obs.model_copy(), "none": None,
                "dict": obs.model_dump()}[which]
    r = asyncio.run(port.submit(turn(), based_on=based_on))
    assert r.status == CommandStatus.REFUSED and "not based on an observation" in r.reason
    assert ex.calls == []


def test_replayed_request_id_is_rejected():
    port, feed, _, ex, _ = make()
    a = turn()
    first = asyncio.run(port.submit(a, based_on=feed.observe()))
    again = asyncio.run(port.submit(a, based_on=feed.observe()))
    assert first.status == CommandStatus.SUCCEEDED
    assert again.status == CommandStatus.REFUSED and "already submitted" in again.reason
    assert len(ex.calls) == 1


def test_rejected_request_id_is_not_burned():
    port, feed, feed2, ex, _ = make()
    a = turn()
    asyncio.run(port.submit(a, based_on=feed2.observe()))    # rejected: wrong observation
    r = asyncio.run(port.submit(a, based_on=feed.observe()))
    assert r.status == CommandStatus.SUCCEEDED


def test_gate_rejection_blocks_execution():
    class Deny:
        name = "deny"

        def decide(self, action, based_on):
            return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                                gate="deny", outcome=GateOutcome.REJECTED, assessed=True,
                                reason="too risky")
    port, feed, _, ex, rec = make(gate=Deny())
    r = asyncio.run(port.submit(turn(), based_on=feed.observe()))
    assert r.status == CommandStatus.REFUSED and "too risky" in r.reason and ex.calls == []
    assert rec.of("gate_decision")[0]["decision"]["outcome"] == "rejected"


def test_broken_gate_fails_closed():
    class Broken:
        name = "broken"

        def decide(self, action, based_on):
            raise RuntimeError("bug")
    port, feed, _, ex, rec = make(gate=Broken())
    r = asyncio.run(port.submit(turn(), based_on=feed.observe()))
    assert r.status == CommandStatus.REFUSED and ex.calls == [] and rec.of("gate_error")


def test_cancellation_during_execution_is_recorded_and_propagates():
    port, feed, _, ex, rec = make(block=True)

    async def go():
        t = asyncio.ensure_future(port.submit(turn(), based_on=feed.observe()))
        await ex.started.wait()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    asyncio.run(go())
    assert rec.of("command_cancelled") and not rec.of("command_result")


def test_port_and_feed_must_belong_to_the_same_vehicle():
    adapter = SnapAdapter()
    with pytest.raises(ValueError):
        ActionPort("Drone1", FakeExecutor(), ObservationFeed(adapter, "Drone2"))
