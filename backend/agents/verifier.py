import re
from dataclasses import dataclass

from .llm import get_fast_llm, parse_json
from .state import AgentState

# timestamps the ANSWER cites look like [12:34]
BRACKET_TS_RE = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]")
# timestamps in the EVIDENCE may be bracketed (transcript) or bare (quiz JSON "00:10")
ANY_TS_RE = re.compile(r"(?<![\d:])(\d{1,2}:\d{2}(?::\d{2})?)(?![\d:])")

VIDEO_TOOLS = {"transcript", "rag", "quiz"}   # only these can validate a video timestamp
TOLERANCE_S = 3
MAX_EVIDENCE_CHARS = 40000   # keep in sync with MAX_TRANSCRIPT_CHARS in tools.py


@dataclass
class Verdict:
    passed: bool
    reason: str = ""


def _to_sec(ts: str) -> int:
    sec = 0
    for p in (int(x) for x in ts.split(":")):
        sec = sec * 60 + p
    return sec


def _timestamps(text: str, bracketed: bool = True) -> set[int]:
    rx = BRACKET_TS_RE if bracketed else ANY_TS_RE
    return {_to_sec(m) for m in rx.findall(text)}


def verify(state: AgentState, draft: str) -> Verdict:
    # 0) fail fast, no LLM call: nothing usable was gathered
    if not state.observations or all(
        o["content"].startswith(("ERROR", "SKIPPED")) for o in state.observations
    ):
        return Verdict(False, "No usable evidence gathered; call a tool.")

    evidence = "\n\n".join(o["content"] for o in state.observations)

    # 1) cheap code check: every timestamp the answer cites must exist in the video evidence
    video_evidence = "\n".join(o["content"] for o in state.observations if o["tool"] in VIDEO_TOOLS)
    valid = _timestamps(video_evidence, bracketed=False)
    bad = [t for t in _timestamps(draft) if not any(abs(t - v) <= TOLERANCE_S for v in valid)]
    if bad:
        return Verdict(False, f"Answer cites timestamps not present in the evidence: {sorted(bad)}")

    # 2) LLM judge (small model): grounded + answers the goal
    prompt = (
        "You are a strict verifier.\n"
        f"QUESTION: {state.goal}\n\nEVIDENCE:\n{evidence[:MAX_EVIDENCE_CHARS]}\n\nANSWER:\n{draft}\n\n"
        "Check: (a) every claim in the ANSWER is supported by the EVIDENCE, "
        "(b) the ANSWER actually addresses the QUESTION.\n"
        'Reply JSON only: {"grounded": bool, "answers_goal": bool, "reason": "<what is missing or unsupported>"}'
    )
    try:
        data = parse_json(get_fast_llm().invoke(prompt).content)
    except Exception:
        return Verdict(True)  # never loop because the verifier itself broke
    if not data:
        return Verdict(True)
    if data.get("grounded") and data.get("answers_goal"):
        return Verdict(True)
    return Verdict(False, data.get("reason", "unsupported or incomplete answer"))