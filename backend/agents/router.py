import re

from .llm import get_fast_llm, parse_json

COMPLEX_HINTS = re.compile(
    r"\b(quiz|test me|summar\w*|overview|tl;?dr|main points|key takeaways|"
    r"what is (this|the) video about|"
    r"search|web|online|latest|compare|fact.?check|verify|"
    r"is (it|this|that) (true|correct|accurate)|outside|other sources)\b",
    re.I,
)


def route_query(question: str) -> str:
    """Return 'simple' (single RAG lookup) or 'complex' (full agent loop)."""
    if COMPLEX_HINTS.search(question):
        return "complex"
    prompt = (
        "Classify this question about a YouTube video.\n"
        '"simple" = answerable from one lookup in the video.\n'
        '"complex" = needs several steps, outside information, or generating a quiz.\n'
        f'Question: {question}\nReply JSON only: {{"route": "simple" | "complex"}}'
    )
    try:
        data = parse_json(get_fast_llm().invoke(prompt).content)
        return data["route"] if data and data.get("route") in ("simple", "complex") else "simple"
    except Exception:
        return "simple"