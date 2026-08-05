const SERVER_URL = "http://localhost:5000";

const els = {
  videoStatus: document.getElementById("videoStatus"),
  noVideoState: document.getElementById("noVideoState"),
  indexState: document.getElementById("indexState"),
  indexBtn: document.getElementById("indexBtn"),
  indexHint: document.getElementById("indexHint"),
  chatState: document.getElementById("chatState"),
  messages: document.getElementById("messages"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  serverStatus: document.getElementById("serverStatus"),
};

let currentVideoId = null;

function extractVideoId(url) {
  if (!url) return null;
  try {
    const u = new URL(url);
    if (u.hostname.includes("youtube.com") && u.pathname === "/watch") {
      return u.searchParams.get("v");
    }
    if (u.hostname === "youtu.be") {
      return u.pathname.slice(1);
    }
    if (u.pathname.startsWith("/shorts/")) {
      return u.pathname.split("/")[2];
    }
  } catch (e) {
    return null;
  }
  return null;
}

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

async function checkServer() {
  try {
    const res = await fetch(`${SERVER_URL}/api/health`);
    if (res.ok) {
      els.serverStatus.classList.add("online");
      els.serverStatus.classList.remove("offline");
      return true;
    }
  } catch (e) {
    // fall through
  }
  els.serverStatus.classList.add("offline");
  els.serverStatus.classList.remove("online");
  return false;
}

async function checkIndexed(videoId) {
  const res = await fetch(`${SERVER_URL}/api/status?video_id=${encodeURIComponent(videoId)}`);
  if (!res.ok) throw new Error("Status check failed");
  const data = await res.json();
  return data.indexed;
}

function showChat() {
  els.indexState.classList.add("hidden");
  els.chatState.classList.remove("hidden");
  els.questionInput.focus();
  if (els.messages.childElementCount === 0) {
    addMessage("Indexed! Ask me anything about this video.", "bot");
  }
}

async function handleIndex() {
  els.indexBtn.disabled = true;
  els.indexBtn.textContent = "Indexing… (first time takes ~30-60s)";
  els.indexHint.textContent = "";
  els.indexHint.className = "hint";

  try {
    const res = await fetch(`${SERVER_URL}/api/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Indexing failed.");
    }

    els.indexHint.textContent = `Indexed ${data.chunks} chunks.`;
    els.indexHint.className = "hint success";
    showChat();
  } catch (e) {
    els.indexHint.textContent = e.message;
    els.indexHint.className = "hint error";
    els.indexBtn.disabled = false;
    els.indexBtn.textContent = "Retry indexing";
  }
}

async function handleAsk(question) {
  addMessage(question, "user");
  const pending = addMessage("Thinking…", "bot pending");
  els.sendBtn.disabled = true;

  try {
    const res = await fetch(`${SERVER_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId, question }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    pending.textContent = data.answer;
    pending.className = "msg bot";
  } catch (e) {
    pending.textContent = e.message;
    pending.className = "msg bot error";
  } finally {
    els.sendBtn.disabled = false;
  }
}

async function init() {
  checkServer();

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const videoId = extractVideoId(tab?.url);

  if (!videoId) {
    els.videoStatus.textContent = "No YouTube video detected";
    els.indexState.classList.add("hidden");
    els.noVideoState.classList.remove("hidden");
    return;
  }

  currentVideoId = videoId;
  els.videoStatus.textContent = tab.title?.replace(" - YouTube", "") || videoId;

  const serverUp = await checkServer();
  if (!serverUp) {
    els.indexHint.textContent = "Backend not reachable at localhost:5000. Start it first (see README).";
    els.indexHint.className = "hint error";
    return;
  }

  try {
    const alreadyIndexed = await checkIndexed(videoId);
    if (alreadyIndexed) {
      showChat();
    }
  } catch (e) {
    // status check failed silently — user can still try indexing manually
  }
}

els.indexBtn.addEventListener("click", handleIndex);

els.chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = els.questionInput.value.trim();
  if (!question) return;
  els.questionInput.value = "";
  handleAsk(question);
});

init();
