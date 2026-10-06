import re

from .llm import get_llm, parse_json
from .state import AgentState
from .tools import tool_descriptions

SUMMARY_RE = re.compile(
    r"\b(summar\w*|overview|tl;?dr|main points|key takeaways|what is (this|the) video about)\b",
    re.I,
)
QUIZ_RE = re.compile(r"\b(quiz|test me|mcq|practice questions?|flashcards?)\b", re.I)


def default_steps(state: AgentState) -> list[dict]:
    """Deterministic plan used when the planner LLM is skipped or fails."""
    if QUIZ_RE.search(state.goal):
        return [{"tool": "quiz", "args": {}}]
    if SUMMARY_RE.search(state.goal):
        return [{"tool": "transcript", "args": {}}]
    return [{"tool": "rag", "args": {"query": state.goal}}]


PLAN_PROMPT = """You plan tool calls to answer a question about a YouTube video.

TOOLS:
{tools}

GOAL: {goal}

ALREADY GATHERED (tool names + sizes): {gathered}
TOOL CALLS ALREADY MADE (do not repeat): {tried}
{failure}
Return JSON only: {{"steps": [{{"tool": "<name>", "args": {{...}}}}]}}
Use at most 3 steps. Prefer the cheapest tool that can answer.
- Questions about the video itself: use "transcript" (whole video) or "rag" (specific facts).
- Questions needing outside information: use "web_search", then "web_page" on a returned URL,
  and also gather the video content with "transcript" if the question compares with the video.
If the evidence already gathered is enough, return {{"steps": []}}."""


def plan(state: AgentState) -> list[dict]:
    # Summaries and quizzes have an obvious first step: skip the planner LLM call.
    if not state.observations and (QUIZ_RE.search(state.goal) or SUMMARY_RE.search(state.goal)):
        return default_steps(state)

    gathered = [f"{o['tool']} ({len(o['content'])} chars)" for o in state.observations] or "nothing yet"
    failure = (
        f"LAST ATTEMPT FAILED VERIFICATION: {state.failure_reason}\n"
        "Choose different or additional tools to fix this.\n"
        if state.failure_reason else ""
    )
    prompt = PLAN_PROMPT.format(
        tools=tool_descriptions(),
        goal=state.goal,
        gathered=gathered,
        tried=[t[0] + t[1] for t in state.tools_tried] or "none",
        failure=failure,
    )
    data = parse_json(get_llm(0.0).invoke(prompt).content)
    steps = (data or {}).get("steps") if isinstance(data, dict) else None

    if steps is None:  # planner output unusable -> safe default
        return default_steps(state)
    if not steps and not state.observations:  # never answer with zero evidence
        return default_steps(state)
    return steps[:3]