"""
Flask server for the YouTube Chat extension.

Endpoints:
  GET  /api/health                     -> {"status": "ok"}
  POST /api/index   {"video_id": "..."} -> indexes (or loads cached index for) a video
  POST /api/chat     {"video_id", "question"} -> answers a question about an indexed video

Run with:  python app.py
"""

import os
import re
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from rag import build_index, ask, is_indexed, RagError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yt-chat-backend")

app = Flask(__name__)
# Chrome extension popups load from a chrome-extension:// origin, and this
# server is only ever meant to be called by your own extension/localhost tools.
CORS(app)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _clean_video_id(raw: str) -> str:
    """Accepts either a raw 11-char video ID or a full YouTube URL."""
    if not raw:
        return ""
    raw = raw.strip()
    if VIDEO_ID_RE.match(raw):
        return raw
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", raw)
    return match.group(1) if match else raw


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/status")
def status():
    video_id = _clean_video_id(request.args.get("video_id", ""))
    if not VIDEO_ID_RE.match(video_id):
        return jsonify({"error": "Invalid or missing video_id."}), 400
    return jsonify({"video_id": video_id, "indexed": is_indexed(video_id)})


@app.post("/api/index")
def index_video():
    data = request.get_json(silent=True) or {}
    video_id = _clean_video_id(data.get("video_id", ""))

    if not VIDEO_ID_RE.match(video_id):
        return jsonify({"error": "Invalid or missing video_id."}), 400

    try:
        chunk_count = build_index(video_id)
        return jsonify({"video_id": video_id, "chunks": chunk_count, "status": "indexed"})
    except RagError as e:
        log.warning("Indexing failed for %s: %s", video_id, e)
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        log.exception("Unexpected error indexing %s", video_id)
        return jsonify({"error": f"Unexpected server error: {e}"}), 500


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    video_id = _clean_video_id(data.get("video_id", ""))
    question = data.get("question", "")

    if not VIDEO_ID_RE.match(video_id):
        return jsonify({"error": "Invalid or missing video_id."}), 400

    if not is_indexed(video_id):
        return jsonify({"error": "Video hasn't been indexed yet. Index it first."}), 422

    try:
        answer = ask(video_id, question)
        return jsonify({"answer": answer})
    except RagError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        log.exception("Unexpected error answering question for %s", video_id)
        return jsonify({"error": f"Unexpected server error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=False
    )
