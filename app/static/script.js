(() => {
  const chamber = document.getElementById("chamber");
  const chamberEmpty = document.getElementById("chamberEmpty");
  const chamberImage = document.getElementById("chamberImage");
  const fileInput = document.getElementById("fileInput");
  const logPanel = document.getElementById("logPanel");
  const logLines = document.getElementById("logLines");
  const statusPill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");
  const resultPanel = document.getElementById("resultPanel");
  const stamp = document.getElementById("stamp");
  const errorMsg = document.getElementById("errorMsg");
  const footerHeldOut = document.getElementById("footerHeldOut");

  const statClass = document.getElementById("statClass");
  const statConfidence = document.getElementById("statConfidence");
  const statEnergy = document.getElementById("statEnergy");
  const statThreshold = document.getElementById("statThreshold");
  const statMsp = document.getElementById("statMsp");
  const statProto = document.getElementById("statProto");
  const gaugeFill = document.getElementById("gaugeFill");
  const viewToggle = document.getElementById("viewToggle");
  const rescanBtn = document.getElementById("rescanBtn");
  const tileEnergy = document.getElementById("tileEnergy");
  const chipProto = document.getElementById("chipProto");

  const SCAN_LOG_LINES = [
    "INITIALIZING SENSOR ARRAY...",
    "NORMALIZING INPUT (224×224)...",
    "EXTRACTING FEATURES / RESNET-50 LAYER4...",
    "COMPUTING ENERGY SCORE...",
    "COMPARING AGAINST KNOWN-CLASS THRESHOLD...",
  ];
  const MIN_SCAN_MS = 2600;

  let lastResult = null;
  let objectUrl = null;

  fetch("/api/meta")
    .then((r) => r.json())
    .then((m) => {
      footerHeldOut.textContent = m.held_out_class;
    })
    .catch(() => {
      footerHeldOut.textContent = "unavailable";
    });

  function setChamberState(state) {
    chamber.dataset.state = state;
  }

  function setStatus(state, text) {
    statusPill.dataset.state = state;
    statusText.textContent = text;
  }

  function resetToIdle() {
    setChamberState("idle");
    setStatus("idle", "LINE IDLE");
    chamberEmpty.hidden = false;
    chamberImage.hidden = true;
    chamberImage.src = "";
    logPanel.hidden = true;
    logLines.innerHTML = "";
    resultPanel.hidden = true;
    errorMsg.hidden = true;
    stamp.textContent = "";
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    fileInput.value = "";
    lastResult = null;
    viewToggle.setAttribute("aria-pressed", "false");
    viewToggle.textContent = "SHOW INSPECTION HEATMAP";
    tileEnergy.classList.remove("triggered");
    chipProto.classList.remove("triggered");
  }

  function openPicker() {
    fileInput.click();
  }

  chamber.addEventListener("click", () => {
    if (chamber.dataset.state === "idle" || chamber.dataset.state === "armed") openPicker();
  });
  chamber.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && (chamber.dataset.state === "idle" || chamber.dataset.state === "armed")) {
      e.preventDefault();
      openPicker();
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    chamber.addEventListener(evt, (e) => {
      e.preventDefault();
      if (chamber.dataset.state === "idle") setChamberState("armed");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    chamber.addEventListener(evt, (e) => {
      e.preventDefault();
      if (chamber.dataset.state === "armed") setChamberState("idle");
    })
  );
  chamber.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (f) handleFile(f);
  });

  function runLogSequence() {
    logPanel.hidden = false;
    logLines.innerHTML = "";
    const step = MIN_SCAN_MS / (SCAN_LOG_LINES.length + 0.4);
    SCAN_LOG_LINES.forEach((line, i) => {
      const span = document.createElement("span");
      span.textContent = "> " + line;
      span.style.animationDelay = `${i * step}ms`;
      logLines.appendChild(span);
      setTimeout(() => span.classList.add("done"), i * step + step);
    });
  }

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("That doesn't look like an image file. Try a JPG or PNG.");
      return;
    }

    errorMsg.hidden = true;
    resultPanel.hidden = true;
    stamp.textContent = "";

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    chamberEmpty.hidden = true;
    chamberImage.src = objectUrl;
    chamberImage.hidden = false;

    setChamberState("scanning");
    setStatus("scanning", "SCANNING…");
    runLogSequence();

    const form = new FormData();
    form.append("file", file);

    const minDelay = new Promise((res) => setTimeout(res, MIN_SCAN_MS));
    const request = fetch("/api/scan", { method: "POST", body: form }).then(async (r) => {
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || "The scan failed. Try again.");
      }
      return r.json();
    });

    Promise.all([request, minDelay])
      .then(([result]) => renderResult(result))
      .catch((err) => showError(err.message || "The scan failed. Try again."));
  }

  function renderResult(result) {
    lastResult = result;
    const accepted = result.decision === "ACCEPTED";

    chamberImage.src = result.model_view_image;
    setChamberState(accepted ? "accepted" : "rejected");
    stamp.textContent = accepted ? "ACCEPTED" : "FOREIGN OBJECT";
    setStatus(accepted ? "accepted" : "rejected", accepted ? "ACCEPTED" : "FOREIGN OBJECT DETECTED");

    statClass.textContent = result.predicted_class;
    statConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    statEnergy.textContent = result.energy_score.toFixed(3);
    statThreshold.textContent = result.energy_threshold.toFixed(3);
    statMsp.textContent = result.msp_score.toFixed(3);
    statProto.textContent = result.proto_score.toFixed(1);

    const delta = result.energy_score - result.energy_threshold;
    const percent = Math.min(100, Math.max(0, 50 + delta * 15));
    gaugeFill.style.width = `${percent}%`;

    const triggered = result.triggered_by || [];
    tileEnergy.classList.toggle("triggered", triggered.includes("energy"));
    chipProto.classList.toggle("triggered", triggered.includes("proto"));

    viewToggle.setAttribute("aria-pressed", "false");
    viewToggle.textContent = "SHOW INSPECTION HEATMAP";
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function showError(message) {
    setChamberState("error");
    setStatus("idle", "LINE IDLE");
    logPanel.hidden = true;
    errorMsg.textContent = message;
    errorMsg.hidden = false;
  }

  viewToggle.addEventListener("click", () => {
    if (!lastResult) return;
    const showingHeatmap = viewToggle.getAttribute("aria-pressed") === "true";
    if (showingHeatmap) {
      chamberImage.src = lastResult.model_view_image;
      viewToggle.setAttribute("aria-pressed", "false");
      viewToggle.textContent = "SHOW INSPECTION HEATMAP";
    } else {
      chamberImage.src = lastResult.heatmap_image;
      viewToggle.setAttribute("aria-pressed", "true");
      viewToggle.textContent = "SHOW ORIGINAL VIEW";
    }
  });

  rescanBtn.addEventListener("click", resetToIdle);
})();
