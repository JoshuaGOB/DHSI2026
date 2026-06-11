const state = { collectionId: null, paperKey: null, pollTimer: null };

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").className = isError ? "error" : "";
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

// --- Collections & papers ------------------------------------------------

async function loadCollections() {
  try {
    const collections = await api("/collections");
    const ul = $("collections");
    ul.innerHTML = "";
    collections.forEach((c) => {
      const li = document.createElement("li");
      li.textContent = `${c.name} (${c.num_items})`;
      li.onclick = () => selectCollection(c.id, li);
      ul.appendChild(li);
    });
    setStatus("");
  } catch (err) {
    $("collections").innerHTML = "";
    setStatus(`Could not load collections: ${err.message}`, true);
  }
}

async function selectCollection(id, li) {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; $("ingest-btn").disabled = false; }
  state.collectionId = id;
  document.querySelectorAll("#collections li").forEach((el) => el.classList.remove("active"));
  li.classList.add("active");
  $("papers-section").hidden = false;
  await loadPapers();
}

async function loadPapers() {
  try {
    const papers = await api(`/collections/${state.collectionId}/papers`);
    const ul = $("papers");
    ul.innerHTML = papers.length ? "" : '<li class="muted">No papers yet — ingest first.</li>';
    papers.forEach((p) => {
      const li = document.createElement("li");
      li.innerHTML = `${p.title} <span class="badge ${p.ingest_status}">${p.ingest_status}</span>`;
      if (p.ingest_error) li.title = p.ingest_error;
      li.onclick = () => selectPaper(p, li);
      ul.appendChild(li);
    });
  } catch (err) {
    setStatus(`Could not load papers: ${err.message}`, true);
  }
}

// --- Ingest with job polling ----------------------------------------------

$("ingest-btn").onclick = async () => {
  if (!state.collectionId) return;
  $("ingest-btn").disabled = true;
  setStatus("Ingest running…");
  try {
    const { job_id } = await api(`/collections/${state.collectionId}/ingest`, { method: "POST" });
    state.pollTimer = setInterval(() => pollJob(job_id), 1500);
  } catch (err) {
    $("ingest-btn").disabled = false;
    setStatus(`Ingest failed to start: ${err.message}`, true);
  }
};

async function pollJob(jobId) {
  try {
    const job = await api(`/jobs/${jobId}`);
    await loadPapers();
    if (job.status === "running") return;
    clearInterval(state.pollTimer);
    $("ingest-btn").disabled = false;
    if (job.status === "done") setStatus("Ingest complete.");
    else setStatus(`Ingest failed: ${job.error}`, true);
  } catch (err) {
    clearInterval(state.pollTimer);
    $("ingest-btn").disabled = false;
    setStatus(`Lost ingest job: ${err.message}`, true);
  }
}

// --- Summaries --------------------------------------------------------------

async function selectPaper(paper, li) {
  state.paperKey = paper.zotero_key;
  document.querySelectorAll("#papers li").forEach((el) => el.classList.remove("active"));
  li.classList.add("active");
  $("summary-actions").hidden = paper.ingest_status !== "indexed";
  $("summary-content").textContent = "Loading…";
  try {
    const cached = await api(`/papers/${paper.zotero_key}/summary`);
    $("summary-content").textContent = cached
      ? cached.content
      : (paper.ingest_status === "indexed"
          ? "No summary yet — generate one."
          : `Paper is ${paper.ingest_status}${paper.ingest_error ? ": " + paper.ingest_error : ""}.`);
  } catch (err) {
    $("summary-content").textContent = `Error: ${err.message}`;
  }
}

$("summarize-btn").onclick = async () => {
  $("summarize-btn").disabled = true;
  $("summary-content").textContent = "Generating summary…";
  try {
    const summary = await api(`/papers/${state.paperKey}/summary`, { method: "POST" });
    $("summary-content").textContent = summary.content;
  } catch (err) {
    $("summary-content").textContent = `Summary failed: ${err.message}`;
  } finally {
    $("summarize-btn").disabled = false;
  }
};

// --- Chat --------------------------------------------------------------------

function appendMsg(who, text, citations = []) {
  const div = document.createElement("div");
  div.className = "msg";
  const cites = citations
    .map((c) => {
      const where = c.page != null ? `p. ${c.page}` : "section unknown";
      return `<div class="citation">— ${c.title} (${c.authors}, ${c.year}), ${where}</div>`;
    })
    .join("");
  const span = document.createElement("span");
  span.textContent = text;
  div.innerHTML = `<span class="who">${who}:</span> ${span.innerHTML}${cites}`;
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

$("chat-form").onsubmit = async (e) => {
  e.preventDefault();
  const query = $("chat-input").value.trim();
  if (!query) return;
  if (!state.collectionId) { setStatus("Pick a collection first.", true); return; }
  $("chat-input").value = "";
  appendMsg("You", query);
  appendMsg("…", "thinking");
  try {
    const result = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, collection_id: state.collectionId }),
    });
    $("chat-log").lastChild.remove();
    appendMsg("Answer", result.answer, result.citations);
  } catch (err) {
    $("chat-log").lastChild.remove();
    appendMsg("Error", err.message);
  }
};

loadCollections();
