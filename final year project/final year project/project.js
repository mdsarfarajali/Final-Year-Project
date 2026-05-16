document.addEventListener("DOMContentLoaded", () => {
  // --- DOM ---
  const $ = (id) => document.getElementById(id);
  const els = {
    startBtn: $("startBtn"),
    stopBtn: $("stopBtn"),
    denoiseBtn: $("denoiseBtn"),
    originalAudio: $("originalAudio"),
    denoisedAudio: $("denoisedAudio"),
    fileInput: $("fileInput"),
    backendImg: $("backendImg"),
    backendText: $("backendText"),
    checkBackend: $("checkBackend"),
  };

  // --- state ---
  let mediaRecorder = null;
  let audioChunks = [];
  let originalBlob = null;
  let uploadedFile = null;
  let discoveredApiUrl = null;

  // --- utilities ---
  const setText = (el, t) => (el.textContent = t);
  const setBackendStatus = (ok, text) => {
    const green =
      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><circle cx='9' cy='9' r='8' fill='%2300c853'/></svg>";
    const red =
      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><circle cx='9' cy='9' r='8' fill='%23ff4d4d'/></svg>";
    if (els.backendImg) els.backendImg.src = ok ? green : red;
    if (els.backendText) setText(els.backendText, text);
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // probe a few candidate backends, return first reachable /denoise URL
  async function probeBackendCandidates() {
    if (
      location.host === "localhost:5000" ||
      location.host === "127.0.0.1:5000"
    ) {
      discoveredApiUrl = "/denoise";
      setBackendStatus(true, "Backend connected");
      return discoveredApiUrl;
    }
    const candidates = ["http://localhost:5000", "http://127.0.0.1:5000"];
    for (const base of candidates) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000);
        const resp = await fetch(base + "/", {
          method: "GET",
          mode: "cors",
          signal: controller.signal,
        });
        clearTimeout(timeout);
        if (resp.ok || resp.status === 404) {
          discoveredApiUrl = base + "/denoise";
          setBackendStatus(true, "Backend connected");
          return discoveredApiUrl;
        }
      } catch {}
    }
    discoveredApiUrl = null;
    setBackendStatus(false, "Backend not reachable. Start Flask on :5000.");
    return null;
  }

  function getApiUrl() {
    if (discoveredApiUrl) return discoveredApiUrl;
    if (location.protocol === "file:" || location.origin === "null")
      return "http://localhost:5000/denoise";
    return "http://localhost:5000/denoise";
  }

  function makeAbortController(timeoutMs = 45000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    return { controller, timeout };
  }

  async function apiPostForm(url, formData) {
    const { controller, timeout } = makeAbortController();
    const resp = await fetch(url, {
      method: "POST",
      body: formData,
      mode: "cors",
      signal: controller.signal,
    });
    clearTimeout(timeout);
    return resp;
  }

  function setAudioSrc(el, blobOrUrl) {
    if (!el) return;
    if (blobOrUrl instanceof Blob) el.src = URL.createObjectURL(blobOrUrl);
    else el.src = blobOrUrl;
  }

  // --- event handlers / core logic ---

  // file upload
  els.fileInput?.addEventListener("change", (ev) => {
    const f = ev.target.files && ev.target.files[0];
    uploadedFile = f || null;
    originalBlob = null; // prefer uploaded file
    if (uploadedFile) {
      setAudioSrc(els.originalAudio, URL.createObjectURL(uploadedFile));
      els.denoiseBtn.disabled = false;
    } else {
      els.denoiseBtn.disabled = !originalBlob;
    }
  });

  // recording helpers
  async function startRecording() {
    try {
      els.startBtn.disabled = true;
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error("getUserMedia not supported");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      try {
        els.originalAudio.srcObject = stream;
      } catch {}
      const options = {};
      const pref = "audio/webm;codecs=opus";
      if (
        window.MediaRecorder &&
        MediaRecorder.isTypeSupported &&
        MediaRecorder.isTypeSupported(pref)
      )
        options.mimeType = pref;
      else if (
        window.MediaRecorder &&
        MediaRecorder.isTypeSupported &&
        MediaRecorder.isTypeSupported("audio/webm")
      )
        options.mimeType = "audio/webm";

      mediaRecorder = new MediaRecorder(stream, options);
      audioChunks = [];
      uploadedFile = null;
      els.fileInput && (els.fileInput.value = "");

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size) audioChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const blobType = options.mimeType || "audio/webm";
        originalBlob = new Blob(audioChunks, { type: blobType });
        if (els.originalAudio?.srcObject) {
          els.originalAudio.srcObject.getTracks().forEach((t) => t.stop());
          els.originalAudio.srcObject = null;
        }
        setAudioSrc(els.originalAudio, originalBlob);
        els.denoiseBtn.disabled = false;
      };
      mediaRecorder.start();
      els.stopBtn.disabled = false;
    } catch (err) {
      console.error("Start recording failed:", err);
      alert("Microphone error: " + (err.message || err));
      els.startBtn.disabled = false;
    }
  }

  function stopRecording() {
    try {
      if (mediaRecorder && mediaRecorder.state !== "inactive")
        mediaRecorder.stop();
      els.startBtn.disabled = false;
      els.stopBtn.disabled = true;
    } catch (err) {
      console.warn("Stop failed", err);
    }
  }

  // send selected file (uploaded or recorded) to backend
  async function sendToDenoise(fileLike) {
    if (!fileLike) return alert("No audio selected or recorded.");
    const form = new FormData();
    const filename =
      fileLike.name ||
      (fileLike.type
        ? `input.${fileLike.type.split("/")[1] || "webm"}`
        : "input.webm");
    form.append("file", fileLike, filename);

    const apiUrl = getApiUrl();
    if (!preflightChecks(apiUrl)) return;

    try {
      const resp = await apiPostForm(apiUrl, form);
      if (!resp.ok) {
        const txt = await resp.text().catch(() => resp.statusText);
        throw new Error("Server error " + resp.status + ": " + txt);
      }
      const buf = await resp.arrayBuffer();
      const type = resp.headers.get("Content-Type") || "audio/wav";
      const blob = new Blob([buf], { type });
      setAudioSrc(els.denoisedAudio, blob);
      setBackendStatus(true, "Denoised");
    } catch (err) {
      if (err.name === "AbortError")
        alert("Request timed out. Is the backend running?");
      else if (
        err.message &&
        err.message.toLowerCase().includes("failed to fetch")
      )
        alert(
          "Network error: failed to reach backend. Start Flask (python server.py) and ensure flask-cors is installed."
        );
      else alert("Failed to call denoise API: " + (err.message || err));
      console.error("Denoise request failed:", err);
      setBackendStatus(false, "Request failed. See console.");
    }
  }

  // preflight checks reused
  function preflightChecks(apiUrl) {
    if (location.protocol === "file:") {
      alert(
        "Do not open index.html directly. Serve the frontend over http(s) or open http://localhost:5000/ if using Flask."
      );
      return false;
    }
    if (location.protocol === "https:" && apiUrl.startsWith("http:")) {
      alert("Mixed content blocked: frontend is HTTPS but backend uses HTTP.");
      return false;
    }
    return true;
  }

  // --- wire UI ---
  els.startBtn?.addEventListener("click", startRecording);
  els.stopBtn?.addEventListener("click", stopRecording);
  els.checkBackend?.addEventListener("click", async () => {
    setBackendStatus(false, "Checking...");
    await probeBackendCandidates();
  });

  els.denoiseBtn?.addEventListener("click", async () => {
    const fileToSend = uploadedFile || originalBlob;
    // optional confirm removed to keep UX simple
    await sendToDenoise(fileToSend);
  });

  // probe backend on load (small delay to allow server warmup)
  (async () => {
    await sleep(120);
    await probeBackendCandidates();
  })();
});
