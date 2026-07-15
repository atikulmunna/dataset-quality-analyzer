(() => {
  "use strict";

  const config = window.DQA_CONFIG || { mode: "preview" };
  const state = {
    file: null,
    view: "overview",
    runFilter: "all",
    progressTimers: [],
    tokens: null,
    pollTimer: null,
  };

  let jobs = [
    { name: "street-scenes-v8", id: "8f2c1a7b4d90", preset: "Detection", images: "18,420", findings: 27, status: "succeeded", updated: "14 Jul, 16:42", ext: "YOLO" },
    { name: "warehouse-pallets-v2", id: "bc91e2a70f44", preset: "Segmentation", images: "7,816", findings: 11, status: "succeeded", updated: "10 Jul, 09:18", ext: "COCO" },
    { name: "retail-shelf-v5", id: "d32a09ef81c5", preset: "Low-noise", images: "4,205", findings: 0, status: "running", updated: "Just now", ext: "COCO" },
    { name: "road-signs-v3", id: "41d8c9bb0e73", preset: "Detection", images: "2,980", findings: 43, status: "failed", updated: "04 Jul, 21:07", ext: "YOLO" },
    { name: "drone-survey-v6", id: "77a31d45c20f", preset: "Detection", images: "11,602", findings: 19, status: "succeeded", updated: "29 Jun, 12:34", ext: "YOLO" },
    { name: "factory-defects-v1", id: "ae209d1178f2", preset: "Segmentation", images: "1,488", findings: 8, status: "cancelled", updated: "25 Jun, 18:01", ext: "COCO" },
  ];

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const isLive = config.mode === "live";

  function base64Url(bytes) {
    let binary = "";
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function randomValue(size = 32) {
    return base64Url(crypto.getRandomValues(new Uint8Array(size)));
  }

  async function digestBase64(value) {
    const input = typeof value === "string" ? new TextEncoder().encode(value) : value;
    return base64Url(new Uint8Array(await crypto.subtle.digest("SHA-256", input)));
  }

  async function fileChecksum(file) {
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()));
    let binary = "";
    digest.forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary);
  }

  function tokenPayload(token) {
    try {
      const value = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
      return JSON.parse(atob(value.padEnd(Math.ceil(value.length / 4) * 4, "=")));
    }
    catch { return {}; }
  }

  function saveTokens(payload) {
    const tokens = { ...payload, expires_at: Date.now() + Number(payload.expires_in || 3600) * 1000 };
    sessionStorage.setItem("dqa-tokens", JSON.stringify(tokens));
    state.tokens = tokens;
  }

  function restoreTokens() {
    try {
      const tokens = JSON.parse(sessionStorage.getItem("dqa-tokens") || "null");
      if (tokens && tokens.access_token && tokens.expires_at > Date.now() + 30000) state.tokens = tokens;
    } catch { sessionStorage.removeItem("dqa-tokens"); }
  }

  async function signIn() {
    const verifier = randomValue(64);
    const authState = randomValue(24);
    sessionStorage.setItem("dqa-pkce-verifier", verifier);
    sessionStorage.setItem("dqa-auth-state", authState);
    const challenge = await digestBase64(verifier);
    const parameters = new URLSearchParams({
      response_type: "code",
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirectUri,
      scope: "openid email dqa/jobs",
      code_challenge_method: "S256",
      code_challenge: challenge,
      state: authState,
    });
    location.assign(`${config.cognitoDomain}/oauth2/authorize?${parameters}`);
  }

  async function handleAuthCallback() {
    const parameters = new URLSearchParams(location.search);
    const code = parameters.get("code");
    if (!code) return;
    const expectedState = sessionStorage.getItem("dqa-auth-state");
    const verifier = sessionStorage.getItem("dqa-pkce-verifier");
    if (!expectedState || parameters.get("state") !== expectedState || !verifier) throw new Error("Sign-in response could not be verified.");
    const body = new URLSearchParams({
      grant_type: "authorization_code",
      client_id: config.cognitoClientId,
      code,
      redirect_uri: config.cognitoRedirectUri,
      code_verifier: verifier,
    });
    const response = await fetch(`${config.cognitoDomain}/oauth2/token`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!response.ok) throw new Error("Sign-in token exchange failed.");
    saveTokens(await response.json());
    sessionStorage.removeItem("dqa-auth-state");
    sessionStorage.removeItem("dqa-pkce-verifier");
    history.replaceState({}, "", location.pathname);
  }

  function signOut() {
    sessionStorage.removeItem("dqa-tokens");
    state.tokens = null;
    const parameters = new URLSearchParams({ client_id: config.cognitoClientId, logout_uri: `${location.origin}/` });
    location.assign(`${config.cognitoDomain}/logout?${parameters}`);
  }

  async function apiFetch(path, options = {}) {
    if (!state.tokens) throw new Error("Sign in to continue.");
    const headers = new Headers(options.headers || {});
    headers.set("authorization", `Bearer ${state.tokens.access_token}`);
    if (options.body && !(options.body instanceof FormData)) headers.set("content-type", "application/json");
    const response = await fetch(`${config.apiBaseUrl}${path}`, { ...options, headers });
    const payload = response.status === 204 ? {} : await response.json().catch(() => ({}));
    if (response.status === 401) {
      sessionStorage.removeItem("dqa-tokens");
      state.tokens = null;
      updateAccount();
    }
    if (!response.ok) throw new Error(payload.message || payload.error || `Request failed (${response.status}).`);
    return payload;
  }

  function updateAccount() {
    if (!isLive) return;
    const label = $("#environmentLabel");
    label.lastChild.textContent = " Live private alpha";
    $("#sampleDataset").hidden = true;
    if (!state.tokens) {
      $("#accountName").textContent = "Sign in";
      $("#accountStatus").textContent = "Required to run audits";
      $("#accountButton").setAttribute("aria-label", "Sign in");
      return;
    }
    const claims = tokenPayload(state.tokens.id_token || state.tokens.access_token);
    const email = String(claims.email || "Private alpha user");
    $("#accountName").textContent = email;
    $("#accountStatus").textContent = "Select to sign out";
    $("#accountButton").setAttribute("aria-label", "Sign out");
    $(".avatar").textContent = email.slice(0, 2).toUpperCase();
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]);
  }

  function savedDatasetNames() {
    try { return JSON.parse(localStorage.getItem("dqa-job-names") || "{}"); }
    catch { return {}; }
  }

  function normalizeJob(job) {
    const names = savedDatasetNames();
    const presetNames = { detection: "Detection", segmentation: "Segmentation", segmentation_low_noise: "Low-noise" };
    return {
      name: names[job.job_id] || `dataset-${String(job.job_id).slice(0, 8)}`,
      id: job.job_id,
      preset: presetNames[job.preset] || job.preset,
      images: "—",
      findings: "—",
      status: job.status,
      updated: job.updated_at ? new Date(job.updated_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—",
      ext: "ZIP",
    };
  }

  function populateComparisonInputs() {
    const completed = jobs.filter((job) => job.status === "succeeded");
    [$("#compareOld"), $("#compareNew")].forEach((select, index) => {
      select.replaceChildren(...completed.map((job, jobIndex) => {
        const option = document.createElement("option");
        option.value = job.id;
        option.textContent = `${job.name} · ${job.updated}`;
        option.selected = jobIndex === Math.min(index, Math.max(0, completed.length - 1));
        return option;
      }));
    });
  }

  async function loadJobs() {
    if (!isLive || !state.tokens) return;
    const payload = await apiFetch("/jobs?limit=50");
    jobs = payload.jobs.map(normalizeJob);
    renderTables();
    populateComparisonInputs();
  }

  function renderRow(job, compact = false) {
    const dataset = `<div class="dataset-cell"><span class="dataset-thumb">${escapeHtml(job.ext)}</span><span><strong>${escapeHtml(job.name)}</strong><small>${compact ? `job · ${escapeHtml(job.id)}` : `${escapeHtml(job.images)} images`}</small></span></div>`;
    if (compact) {
      return `<tr data-status="${job.status}" data-search="${escapeHtml(`${job.name} ${job.id}`.toLowerCase())}"><td>${dataset}</td><td><span class="job-id">${escapeHtml(job.id)}</span></td><td>${escapeHtml(job.preset)}</td><td><span class="finding-count">${job.findings}</span></td><td><span class="status ${job.status}">${job.status}</span></td><td>${escapeHtml(job.updated)}</td><td><button class="row-action" type="button" data-run-action="${escapeHtml(job.id)}" aria-label="Open ${escapeHtml(job.name)}">•••</button></td></tr>`;
    }
    return `<tr><td>${dataset}</td><td>${escapeHtml(job.preset)}</td><td>${escapeHtml(job.images)}</td><td><span class="finding-count">${job.findings}</span></td><td><span class="status ${job.status}">${job.status}</span></td><td>${escapeHtml(job.updated)}</td><td><button class="row-action" type="button" data-run-action="${escapeHtml(job.id)}" aria-label="Open ${escapeHtml(job.name)}">→</button></td></tr>`;
  }

  function renderTables() {
    $("#recentRunsBody").innerHTML = jobs.slice(0, 4).map((job) => renderRow(job)).join("");
    $("#allRunsBody").innerHTML = jobs.map((job) => renderRow(job, true)).join("");
    bindRunActions();
  }

  function bindRunActions() {
    $$('[data-run-action]').forEach((button) => {
      button.addEventListener("click", async () => {
        if (!isLive) {
          showToast("Run selected", `Job ${button.dataset.runAction} is ready for the artifact API connection.`);
          return;
        }
        try {
          const payload = await apiFetch(`/jobs/${encodeURIComponent(button.dataset.runAction)}/artifacts`);
          const dialog = $("#infoDialog");
          $("#dialogEyebrow").textContent = "Audit artifacts";
          $("#dialogTitle").textContent = `Job ${button.dataset.runAction}`;
          const body = $("#dialogBody");
          body.replaceChildren();
          if (!payload.artifacts.length) {
            body.textContent = "Artifacts are not available until this audit succeeds.";
          } else {
            const list = document.createElement("ul");
            payload.artifacts.forEach((artifact) => {
              const item = document.createElement("li");
              const link = document.createElement("a");
              link.href = artifact.download_url;
              link.textContent = `${artifact.name} (${formatBytes(artifact.size)})`;
              link.rel = "noopener";
              item.append(link);
              list.append(item);
            });
            body.append(list);
          }
          dialog.showModal();
        } catch (error) { showToast("Artifacts unavailable", error.message, true); }
      });
    });
  }

  function navigate(view) {
    state.view = view;
    $$("[data-view-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.viewPanel === view));
    $$(".nav-item").forEach((item) => {
      const active = item.dataset.view === view;
      item.classList.toggle("active", active);
      if (active) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current");
    });
    $(".sidebar").classList.remove("open");
    $("#mobileMenu").setAttribute("aria-expanded", "false");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function formatBytes(bytes) {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KiB`;
  }

  function selectFile(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      showToast("ZIP archive required", "Choose a .zip file containing a supported YOLO or COCO export.", true);
      return;
    }
    if (file.size > 2 * 1024 ** 3) {
      showToast("Archive is too large", "The hosted alpha accepts compressed archives up to 2 GiB.", true);
      return;
    }
    state.file = file;
    $("#fileName").textContent = file.name;
    $("#fileMeta").textContent = `${formatBytes(file.size)} · ready for secure upload`;
    $("#dropzone").hidden = true;
    $("#selectedFile").hidden = false;
  }

  function clearFile() {
    state.file = null;
    $("#datasetFile").value = "";
    $("#dropzone").hidden = false;
    $("#selectedFile").hidden = true;
  }

  let toastTimer;
  function showToast(title, message, error = false) {
    const toast = $("#toast");
    $("#toastTitle").textContent = title;
    $("#toastMessage").textContent = message;
    toast.firstElementChild.textContent = error ? "!" : "✓";
    toast.firstElementChild.style.background = error ? "var(--red-soft)" : "var(--emerald-soft)";
    toast.firstElementChild.style.color = error ? "var(--red)" : "var(--emerald)";
    toast.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("visible"), 4200);
  }

  function showDialog(type) {
    const dialog = $("#infoDialog");
    const content = type === "formats"
      ? {
          eyebrow: "Supported inputs",
          title: "One clear dataset entry point",
          body: "<ul><li><strong>YOLO detection or segmentation:</strong> one data.yaml or data.yml plus its referenced images and labels.</li><li><strong>COCO:</strong> annotation JSON files and referenced image directories.</li><li>Archives with traversal paths, links, encryption, unsupported methods, or unsafe expansion are rejected before audit.</li></ul>",
        }
      : {
          eyebrow: "Private alpha",
          title: "Retention & privacy",
          body: "<ul><li>Source archives expire after processing, normally within 24–48 hours.</li><li>Successful reports remain available for seven days.</li><li>Job metadata expires after 30 days.</li><li>Workers are isolated, source files are read-only, and another account cannot access your jobs.</li></ul>",
        };
    $("#dialogEyebrow").textContent = content.eyebrow;
    $("#dialogTitle").textContent = content.title;
    $("#dialogBody").innerHTML = content.body;
    dialog.showModal();
  }

  function resetProgress() {
    state.progressTimers.forEach(clearTimeout);
    state.progressTimers = [];
    $$("[data-progress-step]").forEach((item) => {
      item.classList.remove("active", "complete");
      item.querySelector("b").textContent = "—";
    });
    $('[data-progress-step="upload"]').classList.add("active");
    $('[data-progress-step="upload"] b').textContent = "0%";
    $("#progressBar").style.width = "3%";
  }

  function scheduleProgress(delay, step, percent, label) {
    state.progressTimers.push(setTimeout(() => {
      const current = $(`[data-progress-step="${step}"]`);
      const previous = current.previousElementSibling;
      if (previous) {
        previous.classList.remove("active");
        previous.classList.add("complete");
        previous.querySelector("b").textContent = "Done";
      }
      current.classList.add("active");
      current.querySelector("b").textContent = label;
      $("#progressBar").style.width = `${percent}%`;
    }, delay));
  }

  function startPreviewAudit() {
    resetProgress();
    const drawer = $("#progressDrawer");
    $("#progressDataset").textContent = state.file.name;
    drawer.hidden = false;
    $('[data-progress-step="upload"] b').textContent = "38%";
    $("#progressBar").style.width = "12%";
    scheduleProgress(900, "queue", 31, "Queued");
    scheduleProgress(1900, "audit", 62, "Running");
    scheduleProgress(4100, "report", 88, "Building");
    state.progressTimers.push(setTimeout(() => {
      const report = $('[data-progress-step="report"]');
      report.classList.remove("active");
      report.classList.add("complete");
      report.querySelector("b").textContent = "Done";
      $("#progressBar").style.width = "100%";
      jobs.unshift({ name: state.file.name.replace(/\.zip$/i, ""), id: "preview7a91c2", preset: $("input[name=preset]:checked").nextElementSibling.nextElementSibling.textContent, images: "2,416", findings: 16, status: "succeeded", updated: "Just now", ext: "YOLO" });
      renderTables();
      showToast("Preview audit completed", "The production flow will persist this report and secure its artifacts.");
      state.progressTimers.push(setTimeout(() => { drawer.hidden = true; navigate("runs"); }, 2400));
    }, 5600));
  }

  function setLiveProgress(step, percent, label) {
    const order = ["upload", "queue", "audit", "report"];
    order.forEach((name, index) => {
      const item = $(`[data-progress-step="${name}"]`);
      item.classList.toggle("complete", index < order.indexOf(step));
      item.classList.toggle("active", name === step);
      if (index < order.indexOf(step)) item.querySelector("b").textContent = "Done";
    });
    $(`[data-progress-step="${step}"] b`).textContent = label;
    $("#progressBar").style.width = `${percent}%`;
  }

  async function pollJob(jobId) {
    clearTimeout(state.pollTimer);
    const payload = await apiFetch(`/jobs/${encodeURIComponent(jobId)}`);
    const job = payload.job;
    if (job.status === "queued") setLiveProgress("queue", 38, "Queued");
    if (job.status === "running") setLiveProgress("audit", 68, "Running");
    if (["failed", "cancelled", "expired"].includes(job.status)) {
      $("#progressDrawer").hidden = true;
      await loadJobs();
      showToast("Audit did not complete", job.error_code || job.status, true);
      return;
    }
    if (job.status === "succeeded") {
      setLiveProgress("report", 100, "Done");
      await loadJobs();
      showToast("Audit completed", "Your reports are ready in audit history.");
      state.pollTimer = setTimeout(() => { $("#progressDrawer").hidden = true; navigate("runs"); }, 1600);
      return;
    }
    state.pollTimer = setTimeout(() => pollJob(jobId).catch((error) => showToast("Status check failed", error.message, true)), 5000);
  }

  async function startLiveAudit() {
    if (!state.tokens) {
      await signIn();
      return;
    }
    resetProgress();
    $("#progressDataset").textContent = state.file.name;
    $("#progressDrawer").hidden = false;
    setLiveProgress("upload", 8, "Hashing");
    const checksum = await fileChecksum(state.file);
    setLiveProgress("upload", 16, "Preparing");
    const intentPayload = await apiFetch("/uploads", {
      method: "POST",
      body: JSON.stringify({ filename: state.file.name, size_bytes: state.file.size, checksum_sha256: checksum }),
    });
    const intent = intentPayload.upload;
    const form = new FormData();
    Object.entries(intent.post.fields).forEach(([key, value]) => form.append(key, value));
    form.append("file", state.file);
    setLiveProgress("upload", 23, "Uploading");
    const upload = await fetch(intent.post.url, { method: "POST", body: form });
    if (!upload.ok) throw new Error(`Secure upload failed (${upload.status}).`);
    setLiveProgress("queue", 32, "Submitting");
    const jobPayload = await apiFetch("/jobs", {
      method: "POST",
      headers: { "idempotency-key": randomValue(24) },
      body: JSON.stringify({
        dataset_key: intent.object_key,
        preset: $("input[name=preset]:checked").value,
        fail_on: $("#failOn").value,
        near_duplicates: $("#nearDuplicates").checked,
      }),
    });
    const names = savedDatasetNames();
    names[jobPayload.job.job_id] = state.file.name.replace(/\.zip$/i, "");
    localStorage.setItem("dqa-job-names", JSON.stringify(names));
    await loadJobs();
    await pollJob(jobPayload.job.job_id);
  }

  function applyRunFilters() {
    const query = $("#runSearch").value.trim().toLowerCase();
    $$("#allRunsBody tr").forEach((row) => {
      const statusMatch = state.runFilter === "all" || (state.runFilter === "running" ? ["running", "queued"].includes(row.dataset.status) : row.dataset.status === state.runFilter);
      const searchMatch = !query || row.dataset.search.includes(query);
      row.hidden = !(statusMatch && searchMatch);
    });
  }

  function initializeTheme() {
    const saved = localStorage.getItem("dqa-theme");
    const theme = saved || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = theme;
    $("#themeToggle").setAttribute("aria-label", theme === "dark" ? "Use light theme" : "Use dark theme");
  }

  renderTables();
  initializeTheme();

  async function initializeLive() {
    if (!isLive) return;
    if (![config.apiBaseUrl, config.cognitoDomain, config.cognitoClientId, config.cognitoRedirectUri].every(Boolean)) {
      throw new Error("Live configuration is incomplete.");
    }
    await handleAuthCallback();
    restoreTokens();
    updateAccount();
    if (state.tokens) await loadJobs();
  }

  initializeLive().catch((error) => {
    updateAccount();
    showToast("Live connection unavailable", error.message, true);
  });

  $$(".nav-item").forEach((item) => item.addEventListener("click", () => navigate(item.dataset.view)));
  $$('[data-go-to]').forEach((button) => button.addEventListener("click", () => navigate(button.dataset.goTo)));

  $("#themeToggle").addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("dqa-theme", theme);
    $("#themeToggle").setAttribute("aria-label", theme === "dark" ? "Use light theme" : "Use dark theme");
  });

  $("#mobileMenu").addEventListener("click", (event) => {
    const open = $(".sidebar").classList.toggle("open");
    event.currentTarget.setAttribute("aria-expanded", String(open));
  });

  $("#accountButton").addEventListener("click", () => {
    if (!isLive) return;
    if (state.tokens) signOut(); else signIn().catch((error) => showToast("Sign-in unavailable", error.message, true));
  });

  const fileInput = $("#datasetFile");
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) selectFile(fileInput.files[0]); });
  $("#removeFile").addEventListener("click", clearFile);
  $("#sampleDataset").addEventListener("click", () => selectFile({ name: "street-scenes-sample.zip", size: 48.7 * 1024 ** 2 }));

  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => { if (event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]); });

  $$(".preset-card input").forEach((radio) => radio.addEventListener("change", () => {
    $$(".preset-card").forEach((card) => card.classList.toggle("selected", card.contains($("input[name=preset]:checked"))));
  }));

  $("#auditForm").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!state.file) {
      showToast("Choose a dataset first", "Upload a ZIP or use the sample dataset to preview the audit flow.", true);
      dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    if (isLive) {
      startLiveAudit().catch((error) => {
        $("#progressDrawer").hidden = true;
        showToast("Audit could not start", error.message, true);
      });
      return;
    }
    startPreviewAudit();
  });

  $("#runSearch").addEventListener("input", applyRunFilters);
  $$("[data-filter]").forEach((button) => button.addEventListener("click", () => {
    state.runFilter = button.dataset.filter;
    $$("[data-filter]").forEach((item) => item.classList.toggle("active", item === button));
    applyRunFilters();
  }));

  $("#compareButton").addEventListener("click", () => {
    const summary = $("#comparisonSummary");
    summary.animate([{ opacity: .35, transform: "translateY(5px)" }, { opacity: 1, transform: "none" }], { duration: 280, easing: "ease-out" });
    showToast("Comparison refreshed", "Three meaningful changes were found between these runs.");
  });

  $("#retentionButton").addEventListener("click", () => showDialog("retention"));
  $("#formatHelp").addEventListener("click", () => showDialog("formats"));
  $(".dialog-close").addEventListener("click", () => $("#infoDialog").close());
  $("#infoDialog").addEventListener("click", (event) => { if (event.target === $("#infoDialog")) $("#infoDialog").close(); });
})();
