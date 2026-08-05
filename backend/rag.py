"""
Core RAG (Retrieval-Augmented Generation) logic for the YouTube Chat extension.
"""

import os
from pathlib import Path
import certifi
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_indexes"
INDEX_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

SYSTEM_PROMPT = """
You are a helpful assistant answering questions about a YouTube video.

Answer ONLY from the transcript context below.

If the answer is not present in the transcript,
reply:

"I don't know based on the transcript."

Be concise.
"""

_embeddings = None
_llm = None

_vector_store_cache = {}


class RagError(Exception):
    pass


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------
def get_llm():
    global _llm

    print("=" * 60)
    print("SSL_CERT_FILE:", repr(os.environ.get("SSL_CERT_FILE")))
    print("REQUESTS_CA_BUNDLE:", repr(os.environ.get("REQUESTS_CA_BUNDLE")))
    print("CURL_CA_BUNDLE:", repr(os.environ.get("CURL_CA_BUNDLE")))
    print("CERTIFI:", certifi.where())
    print("=" * 60)

    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RagError("GROQ_API_KEY not found")

        _llm = Groq(api_key=api_key)

    return _llm
# ---------------------------------------------------------------------
def _index_path(video_id):
    return INDEX_DIR / video_id


# ---------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------
def fetch_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(
            video_id,
            languages=["en", "hi"],
        )

        return " ".join(chunk.text for chunk in transcript)

    except (TranscriptsDisabled, NoTranscriptFound):
        raise RagError(
            "No transcript available for this video."
        )

    except VideoUnavailable:
        raise RagError(
            "Video unavailable or private."
        )

    except Exception as e:
        raise RagError(f"Could not fetch transcript: {e}")
# ---------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------
def build_index(video_id: str):

    if video_id in _vector_store_cache:
        return _vector_store_cache[video_id].index.ntotal

    embeddings = get_embeddings()

    path = _index_path(video_id)

    # already saved
    if path.exists():

        vector_store = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        _vector_store_cache[video_id] = vector_store

        return vector_store.index.ntotal

    transcript = fetch_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.create_documents([transcript])

    if len(docs) == 0:
        raise RagError("Transcript is empty.")

    vector_store = FAISS.from_documents(
        docs,
        embeddings,
    )

    vector_store.save_local(str(path))

    _vector_store_cache[video_id] = vector_store

    return len(docs)

# ---------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------
def ask(video_id, question):

    if not question.strip():
        raise RagError("Question cannot be empty.")

    build_index(video_id)

    retriever = _vector_store_cache[video_id].as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
{SYSTEM_PROMPT}

Transcript Context:

{context}

Question:

{question}
"""

    client = get_llm()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
# ---------------------------------------------------------------------
def is_indexed(video_id):

    return (
        video_id in _vector_store_cache
        or _index_path(video_id).exists()
    )