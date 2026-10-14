"""
Gemini pilot against a FAKE drone: checks the API key, the model, tool calling
and the whole agent loop WITHOUT Project AirSim. Run this before the live test.

    python .\\check_gemini.py [--model MODEL] [--slow-first SECONDS]

Needs: pip install google-genai, and GEMINI_API_KEY (environment or the repo's
.env; optionally GEMINI_MODEL). The fake drone (tests/test_executor.py fakes)
starts airborne at N=-1, E=8, altitude 5 and flies every command exactly.
The real ActionPort, executor logic, recorder and summarizer are used, so the
run log in runs/ looks like a live one (marked as a fake-drone check).

--slow-first N   delays the model's FIRST submit_action by N seconds after its
                 observation (N > 10 makes it expire), to check that the model
                 handles observation_expired by observing again.
"""
import argparse
import asyncio
import sys
import time

from agent.llm_pilot import FlightRules, LLMPilot, PilotLimits
from agent.providers import pick_gemini
from channel.action_port import ActionPort
from channel.observation_feed import ObservationFeed
from evaluation.recorder import EvaluationRecorder
from evaluation.summarize_run import summarize
from executor.projectairsim_executor import ProjectAirSimExecutor
from tests.test_executor import FAST, MultiAdapter

TASK = ("Climb to 3 m above your current altitude, fly to a point 10 m north of the launch "
        "point at that altitude, face east, then fly back above the launch point at the same "
        "altitude. Then give your final comment.")


class SlowFirstSubmit:
    """Wraps a provider: holds the first turn that contains submit_action for N seconds."""

    def __init__(self, provider, delay_s):
        self.inner, self.delay_s, self.name, self.model = provider, delay_s, provider.name, provider.model
        self.done = False

    def start(self, system, tools):
        s = self.inner.start(system, tools)
        outer = self

        class Session:
            async def _hold(self, turn):
                if not outer.done and any(c.name == "submit_action" for c in turn.tool_calls):
                    outer.done = True
                    print(f"  (holding the first submit_action for {outer.delay_s:g} s)")
                    await asyncio.sleep(outer.delay_s)
                return turn

            async def send_user(self, text):
                return await self._hold(await s.send_user(text))

            async def send_tool_results(self, results):
                return await self._hold(await s.send_tool_results(results))
        return Session()


async def main(args) -> int:
    print("checking model availability...")
    try:
        provider, tried = await pick_gemini(args.model)
    except RuntimeError as err:
        print(f"\nNO MODEL AVAILABLE: {err}")
        return 2
    if args.slow_first:
        provider = SlowFirstSubmit(provider, args.slow_first)
    print(f"provider {provider.name}, model {provider.model}")
    rec = EvaluationRecorder(run_config={
        "script": "check_gemini.py", "fake_drone": True, "model": provider.model,
        "preflight": tried,
        "slow_first_s": args.slow_first,
        "pipeline": {"observation_source": "FAKE drone (no simulator)", "attack_hooks": [],
                     "trust_assessor": "none", "action_gate": "none",
                     "agents": f"LLMPilot({provider.name}:{provider.model})"}})
    adapter = MultiAdapter(["Drone1"])
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None,
                               on_event=rec.recorder_for(source="executor"))
    feed = ObservationFeed(adapter, "Drone1", recorder=rec)
    port = ActionPort("Drone1", ex, feed, recorder=rec)

    def on_step(vid, label):
        print(f"  [{time.strftime('%H:%M:%S')}] {vid}: {label}")
    pilot = LLMPilot("Drone1", feed, port, provider, task=TASK, recorder=rec,
                     limits=PilotLimits(time_budget_s=300), rules=FlightRules(), on_step=on_step)
    outcome = "aborted"
    try:
        pilot.capture_home()
        stop = asyncio.Event()
        sampler = asyncio.create_task(feed.sample(5.0, stop))
        try:
            out = await pilot.run()
        finally:
            stop.set()
            await sampler
        print(f"\nended: {out.ended}; {out.model_turns} model turns; "
              f"{out.succeeded_actions}/{len(out.submissions)} actions succeeded")
        for f in out.submissions:
            print(f"  {f['action_type']}: {f['code']}")
        print(f"\nfinal comment:\n{out.comment}")
        outcome = "pass" if out.ended == "completed" and out.succeeded_actions >= 1 else "fail"
    finally:
        rec.close(outcome)
        print("\n================ RUN LOG SUMMARY ================")
        print(summarize(rec.path))
    print("\nRESULT:", outcome.upper())
    return 0 if outcome == "pass" else 1


def list_models() -> int:
    """Models this API key can use for text generation (to pick GEMINI_MODEL)."""
    from google import genai

    from agent.providers.gemini import api_key_from_env
    from utils.env import load_env_file
    load_env_file()
    client = genai.Client(api_key=api_key_from_env())
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            print(m.name.removeprefix("models/"))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--slow-first", type=float, default=0.0)
    ap.add_argument("--list-models", action="store_true",
                    help="print the models this API key can use, then exit")
    a = ap.parse_args()
    sys.exit(list_models() if a.list_models else asyncio.run(main(a)))
