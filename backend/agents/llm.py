import json
import re
from langchain_groq import ChatGroq
import os

MAIN_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FAST_MODEL = os.environ.get("GROQ_FAST_MODEL", "openai/gpt-oss-20b")   # router + verifier

def get_llm(temperature: float = 0.2):
    return ChatGroq(model=MAIN_MODEL, temperature=temperature)


def get_fast_llm():
    return ChatGroq(model=FAST_MODEL, temperature=0.0)


def parse_json(text: str):
    """Pull the first JSON object/array out of an LLM reply. Returns None on failure."""
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
