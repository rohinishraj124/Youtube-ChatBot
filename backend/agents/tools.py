import inspect
import json
import logging
import re

import requests
from bs4 import BeautifulSoup

from . import adapters
from .llm import get_llm, parse_json
from .state import AgentState

log = logging.getLogger("yt-chat-backend")

MAX_TRANSCRIPT_CHARS = 40000
MAX_PAGE_CHARS = 4000


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _lines(segs: list[dict]) -> str:
    return "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in segs)


# ---------- tools: each takes (state, **args) -> str ----------

def transcript(state: AgentState, start: float | None = None, end: float | None = None) -> str:
    segs = adapters.get_segments(state.video_id)
    if start is not None:
        segs = [s for s in segs if s["start"] >= float(start)]
    if end is not None:
        segs = [s for s in segs if s["start"] <= float(end)]
    text = _lines(segs)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n[truncated - request a narrower start/end range]"
    return text or "(no transcript found for that range)"


def rag(state: AgentState, query: str, k: int = 6) -> str:
    chunks = adapters.search_index(state.video_id, query, int(k))
    return _lines(chunks) if chunks else "(no relevant chunks found)"


def web_search(state: AgentState, query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=5)
    lines = []
    for r in results:
        url = r.get("href") or r.get("url")
        if url and url not in state.seen_urls:
            state.seen_urls.append(url)
        lines.append(f"- {r.get('title')} | {url}\n  {r.get('body', '')}")
    return "\n".join(lines) or "(no results)"


def web_page(state: AgentState, url: str) -> str:
    if url == "$FIRST_RESULT":
        url = state.seen_urls[0] if state.seen_urls else ""
    # only fetch URLs that web_search actually returned
    if url not in state.seen_urls:
        return "ERROR: url was not returned by web_search; run web_search first."
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:MAX_PAGE_CHARS]
    # fetched pages are untrusted: label them so the answer step treats them as data
    return f"<untrusted_web_content url={url}>\n{text}\n</untrusted_web_content>"


def quiz(state: AgentState, n: int = 5, topic: str | None = None) -> str:
    n = max(1, min(int(n), 10))
    context = rag(state, topic) if topic else transcript(state)
    prompt = (
        f"Write {n} multiple-choice questions based ONLY on this video content.\n"
        "Rules:\n"
        "- Test understanding of what is SAID, not when it is said. Never ask about timestamps.\n"
        "- Every question must be answerable from the content, with one clearly correct option.\n"
        "- Wrong options must be plausible, not silly.\n"
        "- Write the questions in English unless the content is clearly meant for another language.\n"
        'Return JSON only: [{"question": str, "options": [str,str,str,str], '
        '"answer_index": int, "timestamp": "mm:ss"}]\n'
        "Each timestamp must be one that appears in the CONTENT.\n\n"
        f"CONTENT:\n{context}"
    )
    out = get_llm(0.3).invoke(prompt).content
    data = parse_json(out)
    return json.dumps(data, indent=2, ensure_ascii=False) if data else out


TOOLS = {
    "transcript": (transcript, 'transcript(start?: seconds, end?: seconds) - raw transcript with timestamps; best for summaries and "what happens at X"'),
    "rag": (rag, "rag(query: str) - semantic search over the video; best for specific facts"),
    "web_search": (web_search, "web_search(query: str) - search the web for outside info; returns titles, URLs, snippets"),
    "web_page": (web_page, 'web_page(url: str) - read a page; url must come from web_search (use "$FIRST_RESULT" for the top hit)'),
    "quiz": (quiz, "quiz(n?: int, topic?: str) - generate multiple-choice questions from the video"),
}


def tool_descriptions() -> str:
    return "\n".join(f"- {desc}" for _, desc in TOOLS.values())


def _clean_args(fn, args: dict) -> dict:
    """Drop arguments the planner invented so a stray key doesn't crash the tool."""
    allowed = set(inspect.signature(fn).parameters) - {"state"}
    return {k: v for k, v in (args or {}).items() if k in allowed}


def run_tool(state: AgentState, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"tool": name, "args": args, "content": f"ERROR: unknown tool {name}"}

    fn = TOOLS[name][0]
    args = _clean_args(fn, args)
    key = (name, json.dumps(args, sort_keys=True))
    if key in state.tools_tried:
        return {"tool": name, "args": args, "content": "SKIPPED: identical call already made"}
    state.tools_tried.add(key)

    try:
        content = fn(state, **args)
    except Exception as e:  # a failing tool becomes an observation, not a crash
        log.exception("Tool %s failed", name)
        state.tools_tried.discard(key)      # a failed call may be retried
        content = f"ERROR: {type(e).__name__}: {e}"
    return {"tool": name, "args": args, "content": content}