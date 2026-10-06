import re
from functools import lru_cache

from youtube_transcript_api import YouTubeTranscriptApi


@lru_cache(maxsize=32)
def get_segments(video_id: str) -> list[dict]:
    """Transcript as [{"text", "start"}], works with old and new youtube-transcript-api."""
    if hasattr(YouTubeTranscriptApi, "fetch"):          # v1.x
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=["en", "hi"])
        except Exception:
            fetched = next(iter(api.list(video_id))).fetch()   # any available language
        return [{"text": s.text, "start": float(s.start)} for s in fetched]
    data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi"])  # older
    return [{"text": d["text"], "start": float(d["start"])} for d in data]


def search_index(video_id: str, query: str, k: int = 6) -> list[dict]:
    """Temporary keyword search over transcript windows (replace with your FAISS index)."""
    segs = get_segments(video_id)
    q = set(re.findall(r"\w+", query.lower()))
    scored = []
    for i in range(0, len(segs), 4):
        chunk = segs[i:i + 6]
        text = " ".join(s["text"] for s in chunk)
        score = len(q & set(re.findall(r"\w+", text.lower())))
        scored.append((score, {"text": text, "start": chunk[0]["start"]}))
    scored.sort(key=lambda x: -x[0])
    hits = [c for sc, c in scored[:k] if sc > 0]
    return hits or [c for _, c in scored[:k]]