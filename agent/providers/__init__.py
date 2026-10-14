"""Model providers for LLM agents. See base.py for the interface."""
import asyncio
import os

DEFAULT_GEMINI_MODELS = ["gemini-3.5-flash-lite"]
# Free-tier requests per minute seen in AI Studio (2026-10-14). We pace at ~80% of these.
# GEMINI_RPM overrides for every model (e.g. after enabling billing).
FREE_TIER_RPM = {"gemini-3.6-flash": 5, "gemini-3.5-flash-lite": 15}
UNKNOWN_MODEL_RPM = 5
RPM_HEADROOM = 0.8


def rpm_for(model: str) -> float:
    env = os.environ.get("GEMINI_RPM")
    if env:
        return float(env)
    return max(1.0, FREE_TIER_RPM.get(model, UNKNOWN_MODEL_RPM) * RPM_HEADROOM)


# HTTP codes that will not fix themselves by retrying (bad key, no access, unknown model)
_PERMANENT = {400, 401, 403, 404}


def gemini_models_from_env(model: str | None = None) -> list[str]:
    """Preference list: `model` (CLI) or GEMINI_MODEL, comma-separated, e.g.
    "gemini-3.5-flash-lite,gemini-3.6-flash". Loads the repo's .env first."""
    from utils.env import load_env_file
    load_env_file()
    raw = model or os.environ.get("GEMINI_MODEL") or ",".join(DEFAULT_GEMINI_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def gemini_from_env(model: str | None = None, temperature: float | None = 0.2):
    """GeminiProvider for the FIRST preferred model, without checking it is available."""
    from agent.providers.gemini import GeminiProvider
    m = gemini_models_from_env(model)[0]
    return GeminiProvider(model=m, temperature=temperature, rpm=rpm_for(m))


async def pick_gemini(model: str | None = None, temperature: float | None = 0.2,
                      attempts: int = 3, backoff_s: float = 3.0, log=print,
                      make=None, sleep=asyncio.sleep):
    """Preflight, BEFORE anything flies: ping each preferred model in order and return
    (provider, tried) for the first that answers. `tried` lists every attempt (for the
    run record). Raises RuntimeError if none answers.

    Overload (503/429) and network errors are retried `attempts` times with backoff;
    a permanent error (bad key, no access, unknown model) moves on at once."""
    if make is None:
        from agent.providers.gemini import GeminiProvider

        def make(m):
            return GeminiProvider(model=m, temperature=temperature, rpm=rpm_for(m))
    tried = []
    for m in gemini_models_from_env(model):
        try:
            provider = make(m)
        except Exception as err:            # e.g. SDK missing, no key: same for every model
            raise RuntimeError(f"cannot create the Gemini client: {err}") from err
        for attempt in range(attempts):
            try:
                latency = await provider.ping()
            except Exception as err:
                code = getattr(err, "code", None)
                msg = f"{type(err).__name__}" + (f" {code}" if code else "") + f": {str(err)[:160]}"
                tried.append({"model": m, "attempt": attempt + 1, "ok": False, "error": msg})
                log(f"  {m}: attempt {attempt + 1} failed: {msg}")
                if code in _PERMANENT:
                    break
                if attempt + 1 < attempts:
                    await sleep(backoff_s * 2 ** attempt)
                continue
            tried.append({"model": m, "attempt": attempt + 1, "ok": True,
                          "latency_s": round(latency, 3)})
            log(f"  {m}: available ({latency:.1f} s)")
            return provider, tried
    raise RuntimeError("no Gemini model answered: "
                       + "; ".join(f"{t['model']}#{t['attempt']} {t['error']}" for t in tried))
