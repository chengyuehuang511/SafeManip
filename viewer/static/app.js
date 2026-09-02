// SafeManip monitor viewer front-end. No frameworks, just fetch + DOM.

const state = {
  task: null,
  episode: null,
  detail: null,
};

// eval root and training-data root are two different directory trees (see
// server.py's ROOT vs. TRAINING_DATASET_ROOT) -- tracked separately so the
// header can show whichever one is actually relevant to the active tab,
// instead of always showing the eval root even on the Training Data tab.
const roots = { eval: null, training: null };

const el = (sel) => document.querySelector(sel);
const taskSelect = el("#task-select");
const episodeList = el("#episode-list");
const emptyState = el("#empty-state");
const episodeView = el("#episode-view");
const video = el("#video");
const reconVideo = el("#recon-video");

// "Training Data" tab state/elements -- independent screen, own task/episode
// picker, no monitor/violations (just ground-truth reconstructed video + the
// recorded language instruction). See server.py's api_training_* /
// replay/official_playback/README.md.
const tdState = { task: null, episode: null, loaded: false, monitorMethod: null, property: null };
const tdTaskTree = el("#td-task-tree");
const tdPropertyTree = el("#td-property-tree");
const tdEpisodeList = el("#td-episode-list");
const tdEmptyState = el("#td-empty-state");
const tdEpisodeView = el("#td-episode-view");
const tdVideo = el("#td-video");
const tdOriginalVideo = el("#td-original-video");
const tdSyncState = { wired: false };
// The original dataset video is native 20 fps with no frame skip, so its
// frame index equals the monitor frame index 1:1 (see selectTrainingEpisode,
// which passes ratio=1 for it).
const TRAINING_ORIGINAL_FPS = 20;

// Plain `v == null ? fallback : v` instead of `??` -- confirmed via `node
// --check` that `??` breaks parsing in at least one JS runtime this viewer
// needs to run in (nullish coalescing requires a newer engine than that
// one has), and the whole page silently fails to boot (stuck on the static
// "loading…" placeholder text) if *any* top-level syntax in this file is
// rejected, not just the one expression using it.
function orUnknown(v) {
  return v == null ? "?" : v;
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function initTheme() {
  const stored = localStorage.getItem("safemanip-theme");
  const theme = stored || "dark";
  document.documentElement.setAttribute("data-theme", theme);
  const btn = el("#theme-toggle");
  const setLabel = () => {
    const current = document.documentElement.getAttribute("data-theme");
    btn.textContent = current === "light" ? "🌙 dark mode" : "☀️ light mode";
  };
  setLabel();
  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("safemanip-theme", next);
    setLabel();
  });
}

function initTabs() {
  const tabEval = el("#tab-eval");
  const tabTraining = el("#tab-training");
  const screenEval = el("#screen-eval");
  const screenTraining = el("#screen-training");

  tabEval.addEventListener("click", () => {
    tabEval.classList.add("active");
    tabTraining.classList.remove("active");
    screenEval.classList.remove("hidden");
    screenTraining.classList.add("hidden");
    el("#root-path").textContent = roots.eval || "";
  });
  tabTraining.addEventListener("click", () => {
    tabTraining.classList.add("active");
    tabEval.classList.remove("active");
    screenTraining.classList.remove("hidden");
    screenEval.classList.add("hidden");
    el("#root-path").textContent = roots.training || "loading…";
    if (!tdState.loaded) {
      tdState.loaded = true;
      initTrainingData();
    }
  });
}

async function init() {
  initTheme();
  initTabs();
  const data = await fetchJSON("/api/tasks");
  roots.eval = data.root;
  el("#root-path").textContent = data.root;
  taskSelect.innerHTML = "";
  for (const t of data.tasks) {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    taskSelect.appendChild(opt);
  }
  taskSelect.addEventListener("change", () => loadEpisodes(taskSelect.value));
  if (data.tasks.length) {
    // ?task=<name>&episode=<n> deep-links straight to a specific episode
    // (also just a normal, shareable way to point someone at one -- not
    // only a debugging aid).
    const params = new URLSearchParams(location.search);
    const wantTask = params.get("task");
    const task = wantTask && data.tasks.includes(wantTask) ? wantTask : data.tasks[0];
    taskSelect.value = task;
    loadEpisodes(task, params.get("episode"));
  }
}

// --------------------------------------------------------------------------
// Training Data tab
// --------------------------------------------------------------------------

// Raw task list (from /api/td_tasks) and property name list (from
// /api/training_ltl_properties) -- kept around so the Task/LTL trees can be
// re-rendered (e.g. after a method change re-scopes the violation counts)
// without re-fetching either list.
let tdTasksList = [];
let tdPropertiesList = [];
// { by_task: {task: {total, by_property: {prop: count}}},
//   by_property: {prop: {total, by_task: {task: count}}} } for whichever
// method is currently selected -- see server.py's training_violation_counts.
let tdViolationCounts = { by_task: {}, by_property: {} };

async function initTrainingData() {
  const data = await fetchJSON("/api/td_tasks");
  roots.training = data.dataset_root;
  el("#root-path").textContent = data.dataset_root;
  tdTasksList = data.tasks;
  await ensureTrainingMonitorMethods();  // need a method selected before the trees can show violation counts
  await initTrainingLtlPropertyList();
  await refreshViolationCounts();
  renderTaskTree();
  renderPropertyTree();
  if (tdTasksList.length) {
    loadTrainingEpisodes(tdTasksList[0].task);
  }
}

// LTL property list: scopes both the left-column per-episode violation
// badges (server-side, via /api/td_episodes?property=...) and the main
// detail panel's violations/satisfied lists (client-side filter in
// loadTrainingMonitor) to a single named property instead of the
// whole-episode aggregate / all 19 properties. "All properties" (null)
// restores the unfiltered view in both places. See server.py's
// list_training_episodes' property_filter param / _property_status_for.
async function initTrainingLtlPropertyList() {
  try {
    const data = await fetchJSON("/api/training_ltl_properties");
    tdPropertiesList = data.properties;
  } catch (e) {
    tdPropertiesList = [];  // non-fatal -- "All properties" still works, just no per-property entries
  }
}

function selectTrainingProperty(property) {
  tdState.property = property || null;
  renderPropertyTree();
  if (tdState.task) loadTrainingEpisodes(tdState.task);
}

// Fetches the violation-count breakdown for the currently-selected
// postprocess method (tdState.monitorMethod) -- called on init and whenever
// the method picker changes, since counts are scoped to one method at a
// time (see the "Method scope" decision: currently-selected method only,
// not summed across methods).
async function refreshViolationCounts() {
  if (!tdState.monitorMethod) return;
  try {
    tdViolationCounts = await fetchJSON(
      `/api/training_violation_counts?method=${encodeURIComponent(tdState.monitorMethod)}`
    );
  } catch (e) {
    tdViolationCounts = { by_task: {}, by_property: {} };
  }
}

// Generic expandable tree-list row: `label` is the clickable selector text,
// `total` the badge count shown next to it, `children` an array of
// {label, count} shown (read-only) in a collapsible nested list, `isActive`
// highlights it as the current selection, `onSelect` fires on a label/count
// click (not on the caret, which only toggles the nested breakdown).
function buildTreeRow(label, total, isActive, onSelect, children) {
  const wrap = document.createElement("div");
  wrap.className = "tree-item";

  const row = document.createElement("div");
  row.className = "tree-row" + (isActive ? " active" : "");

  const hasChildren = children && children.length > 0;
  const caret = document.createElement("span");
  caret.className = "tree-caret";
  caret.textContent = hasChildren ? "▸" : "";

  const labelSpan = document.createElement("span");
  labelSpan.className = "tree-label";
  labelSpan.textContent = label;
  labelSpan.title = label;

  const countSpan = document.createElement("span");
  countSpan.className = "tree-count" + (total ? " nonzero" : "");
  countSpan.textContent = total == null ? "" : `${total}`;

  row.append(caret, labelSpan, countSpan);
  wrap.appendChild(row);

  if (hasChildren) {
    const childList = document.createElement("div");
    childList.className = "tree-children hidden";
    for (const c of children) {
      const cRow = document.createElement("div");
      cRow.className = "tree-child-row";
      cRow.innerHTML = `<span>${c.label}</span><span>${c.count}</span>`;
      childList.appendChild(cRow);
    }
    wrap.appendChild(childList);
    caret.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const collapsed = childList.classList.toggle("hidden");
      caret.textContent = collapsed ? "▸" : "▾";
    });
  }

  const select = () => onSelect();
  labelSpan.addEventListener("click", select);
  countSpan.addEventListener("click", select);
  return wrap;
}

function renderTaskTree() {
  tdTaskTree.innerHTML = "";
  if (!tdTasksList.length) {
    tdTaskTree.innerHTML = "<div class='muted'>no tasks found</div>";
    return;
  }
  for (const t of tdTasksList) {
    const counts = tdViolationCounts.by_task[t.task] || { total: 0, by_property: {} };
    const children = Object.entries(counts.by_property)
      .sort((a, b) => b[1] - a[1])
      .map(([prop, count]) => ({ label: prop, count }));
    const label = `${t.task} (${t.n_reconstructed} reconstructed)`;
    tdTaskTree.appendChild(
      buildTreeRow(label, counts.total, t.task === tdState.task, () => loadTrainingEpisodes(t.task), children)
    );
  }
}

function renderPropertyTree() {
  tdPropertyTree.innerHTML = "";
  const allTotal = Object.values(tdViolationCounts.by_property).reduce((s, p) => s + p.total, 0);
  tdPropertyTree.appendChild(
    buildTreeRow("All properties", allTotal, tdState.property == null, () => selectTrainingProperty(null), null)
  );
  for (const prop of tdPropertiesList) {
    const counts = tdViolationCounts.by_property[prop] || { total: 0, by_task: {} };
    const children = Object.entries(counts.by_task)
      .sort((a, b) => b[1] - a[1])
      .map(([task, count]) => ({ label: task, count }));
    tdPropertyTree.appendChild(
      buildTreeRow(prop, counts.total, prop === tdState.property, () => selectTrainingProperty(prop), children)
    );
  }
}

async function loadTrainingEpisodes(task) {
  // captured before tdState.task/episode get overwritten below -- used to
  // re-select the same episode after a property-filter change reloads this
  // same task's list (see the bottom of this function).
  const previousTask = tdState.task;
  const previousEpisode = tdState.episode;
  tdState.task = task;
  renderTaskTree();  // update active highlighting in the sidebar tree
  tdEpisodeList.innerHTML = "<div class='loading'>loading episodes…</div>";
  await ensureTrainingMonitorMethods();  // so tdMethodLabel() has short labels ready for the badges below
  const propertyParam = tdState.property ? `&property=${encodeURIComponent(tdState.property)}` : "";
  const data = await fetchJSON(`/api/td_episodes?task=${encodeURIComponent(task)}${propertyParam}`);
  tdEpisodeList.innerHTML = "";
  if (!data.episodes.length) {
    tdEpisodeList.innerHTML = "<div class='muted'>no reconstructed episodes yet for this task"
      + " -- see replay/official_playback/submit_training_data.sh</div>";
    return;
  }
  for (const ep of data.episodes) {
    const row = document.createElement("button");
    // ep.success/num_violations are null (not the eval tab's guaranteed
    // true/false/int) until SafeManip/monitor/extract_privileged_from_dataset.py
    // has actually been run for this episode -- shown as a neutral "not
    // analyzed" badge rather than misleadingly rendering as failure/0-viol.
    const analyzed = ep.success != null;
    row.className = "ep-row" + (analyzed ? (ep.success ? " success" : " failure") : "");
    const successBadge = analyzed
      ? `<span class="mini-badge ${ep.success ? "s-ok" : "s-fail"}">${ep.success ? "success" : "fail"}</span>`
      : `<span class="mini-badge">not analyzed</span>`;
    // One violation-count badge per postprocess method that's actually been
    // run for this episode (ep.methods -- see server.py's
    // list_training_episodes), not just the default method, so both are
    // visible at a glance without opening the episode.
    const methodEntries = Object.entries(ep.methods || {});
    // When a single LTL property is selected (tdState.property), m.num_violations
    // is 1/0/null (violated/satisfied/not-evaluated-for-this-episode) instead of
    // an aggregate count -- worded as such rather than "N viol" for clarity.
    const violBadges = methodEntries.length
      ? methodEntries.map(([key, m]) => {
          const label = tdMethodLabel(key);
          if (tdState.property) {
            if (m.num_violations == null) {
              return `<span class="mini-badge" title="${key}">${label}: n/a</span>`;
            }
            return m.num_violations
              ? `<span class="mini-badge viol" title="${key}">${label}: ✗</span>`
              : `<span class="mini-badge ok" title="${key}">${label}: ✓</span>`;
          }
          return m.num_violations
            ? `<span class="mini-badge viol" title="${key}">${label}: ${m.num_violations} viol</span>`
            : `<span class="mini-badge ok" title="${key}">${label}: 0 viol</span>`;
        }).join("\n      ")
      : "";
    row.innerHTML = `<span class="ep-num">#${ep.episode}</span>
      ${successBadge}
      ${violBadges}
      <span class="mini-badge">${orUnknown(ep.n_frames)} frames</span>`;
    row.addEventListener("click", () => selectTrainingEpisode(task, ep, row));
    tdEpisodeList.appendChild(row);
  }
  // Re-select whichever episode was already open if this reload is for the
  // *same* task (e.g. the property filter just changed) and that episode
  // still exists in the list; otherwise fall back to the first episode
  // (task actually changed, or first load).
  const rows = tdEpisodeList.querySelectorAll(".ep-row");
  let keepIdx = 0;
  if (previousTask === task && previousEpisode != null) {
    const idx = data.episodes.findIndex((e) => e.episode === previousEpisode);
    if (idx !== -1) keepIdx = idx;
  }
  selectTrainingEpisode(task, data.episodes[keepIdx], rows[keepIdx]);
}

function selectTrainingEpisode(task, ep, rowEl) {
  tdEpisodeList.querySelectorAll(".ep-row").forEach((r) => r.classList.remove("active"));
  if (rowEl) rowEl.classList.add("active");

  tdState.episode = ep.episode;
  tdEmptyState.classList.add("hidden");
  tdEpisodeView.classList.remove("hidden");

  el("#td-ep-title").textContent = `${task} — episode ${ep.episode}`;
  el("#td-ep-lang").textContent = ep.lang || "";
  const cams = (ep.camera_names || []).join(", ");
  el("#td-ep-meta").textContent =
    `fps=${orUnknown(ep.fps)} · frames=${orUnknown(ep.n_frames)} · cameras: ${cams || "?"}`;

  tdVideo.src = `/td_video?task=${encodeURIComponent(task)}&episode=${ep.episode}`;
  tdVideo.load();

  // "monitor frame" here means the raw-simulation-frame index the
  // postprocess monitor pipeline actually indexes by (one per states.npz
  // row, ~20fps -- see SafeManip/monitor/extract_privileged_from_dataset.py).
  // The reconstructed video is rendered at video_skip=2 by default (10fps),
  // i.e. 1 video frame = 2 raw/monitor frames -- ratio = fps/20 makes
  // wireFrameReadoutFor's `monitorFrame = videoFrame / ratio` come out to
  // videoFrame * 2, matching that skip exactly. The *original* dataset video
  // is native 20fps (no skip), so ratio=1 there (1:1, no doubling).
  const reconRatio = (ep.fps || 10) / 20;
  wireFrameReadoutFor(tdVideo, "td-frame-readout", ep.fps || 10, reconRatio, null, ep.n_frames);

  const tdSyncRow = el("#td-sync-row");
  if (ep.original_video_url) {
    // lazily ffmpeg-concatenated server-side on first request (see
    // server.py's ensure_original_concat) -- first load of a given episode
    // can take a couple seconds, cached forever after.
    tdOriginalVideo.src = ep.original_video_url;
    tdOriginalVideo.load();
    tdSyncRow.classList.remove("hidden");
    wireSync(tdOriginalVideo, tdVideo, "#td-sync-play-btn", tdSyncState);
    wireFrameReadoutFor(tdOriginalVideo, "td-original-frame-readout", 20, 1, ep.n_frames, ep.n_frames);
  } else {
    tdOriginalVideo.removeAttribute("src");
    tdOriginalVideo.load();
    tdSyncRow.classList.add("hidden");
  }

  loadTrainingMonitor(task, ep.episode, tdState.monitorMethod);
}

// --------------------------------------------------------------------------
// Training Data tab -- postprocess symbolic-monitor panel (violations/
// satisfied properties computed by SafeManip/monitor/extract_privileged_
// from_dataset.py + run_monitor_on_privileged.py from this same ground-truth
// training episode -- see server.py's api_training_monitor). Reuses the
// exact same renderViolation/renderSatisfied/predicateBreakdown/
// renderMissedPanel functions the eval tab uses; only the target DOM ids and
// the fetch URL differ.
// --------------------------------------------------------------------------

let tdMethodsLoaded = false;
// method key (e.g. "v0_2026-08-27_baseline_upstream_predicates") -> short
// label (e.g. "v0"), populated by ensureTrainingMonitorMethods(). Used so
// the per-episode violation-count badges in the left column show the same
// short label as the method dropdown instead of the full directory name.
const tdMethodLabels = {};
function tdMethodLabel(key) {
  return tdMethodLabels[key] || key;
}

async function ensureTrainingMonitorMethods() {
  if (tdMethodsLoaded) return;
  tdMethodsLoaded = true;
  const select = el("#td-method-select");
  try {
    const data = await fetchJSON("/api/training_monitor_methods");
    tdState.monitorMethod = tdState.monitorMethod || data.default;
    select.innerHTML = "";
    for (const [key, info] of Object.entries(data.methods)) {
      tdMethodLabels[key] = info.label;
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = info.label;
      if (key === tdState.monitorMethod) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener("change", async () => {
      tdState.monitorMethod = select.value;
      // Violation counts in the Task/LTL trees are scoped to one method at
      // a time (see refreshViolationCounts) -- re-fetch and re-render both
      // whenever the selected method changes.
      await refreshViolationCounts();
      renderTaskTree();
      renderPropertyTree();
      if (tdState.task && tdState.episode != null) {
        loadTrainingMonitor(tdState.task, tdState.episode, tdState.monitorMethod);
      }
    });
  } catch (e) {
    select.innerHTML = "<option>failed to load methods</option>";
  }
}

async function loadTrainingMonitor(task, episode, method) {
  await ensureTrainingMonitorMethods();
  method = method || tdState.monitorMethod;
  const missing = el("#td-monitor-missing");
  const body = el("#td-monitor-body");
  missing.classList.add("hidden");
  body.classList.add("hidden");
  missing.textContent = "loading monitor results…";
  missing.classList.remove("hidden");

  let detail;
  try {
    detail = await fetchJSON(
      `/api/training_monitor?task=${encodeURIComponent(task)}&episode=${episode}&method=${encodeURIComponent(method)}`
    );
  } catch (e) {
    missing.textContent = `failed to load monitor results: ${e}`;
    return;
  }
  if (detail.error) {
    missing.textContent = detail.error;
    missing.classList.remove("hidden");
    body.classList.add("hidden");
    return;
  }
  // When a single LTL property is selected in the left-column picker, the
  // main detail panel shows only that property's violation/satisfied entry
  // (if any) instead of all 19 -- client-side filter, server still returns
  // the full set (so switching properties doesn't need a re-fetch).
  if (tdState.property) {
    detail = {
      ...detail,
      violations: detail.violations.filter((v) => v.property_name === tdState.property),
      satisfied: detail.satisfied.filter((s) => s.property_name === tdState.property),
    };
  }
  missing.classList.add("hidden");
  body.classList.remove("hidden");
  renderTrainingMonitor(detail);
}

function renderTrainingMonitor(detail) {
  setAnnotationContext(detail.annotation_task_key || `training__${detail.task}`, detail.episode);
  // Two videos on this tab, and both should follow a marker click. The
  // reconstruction is `primary` (its marker time_s is computed against its own
  // fps/ratio); the *original* dataset video goes in the second slot, which
  // seeks by monitor_frame / fps -- correct here because the original is native
  // 20 fps with no frame skip, so 1 monitor frame == 1 video frame (the same
  // ratio=1 that wireFrameReadoutFor uses for it in selectTrainingEpisode).
  // Registering it matters: seekTo sets suppressSeekSync, which deliberately
  // stops wireSync from mirroring the jump, so a video left out of these slots
  // simply does not move.
  const tdOriginalActive = tdOriginalVideo && tdOriginalVideo.getAttribute("src")
    ? tdOriginalVideo
    : null;
  setActiveVideos(tdVideo, tdOriginalActive, TRAINING_ORIGINAL_FPS);

  el("#td-monitor-meta").textContent =
    `fps=${detail.fps} · video frames≈${orUnknown(detail.video_frame_count)} · ` +
    `monitor frames=${orUnknown(detail.monitor_num_frames)} · ` +
    `ratio=${detail.ratio ? detail.ratio.toFixed(3) : "?"} · ` +
    `violated=${orUnknown(detail.num_violated_instances)} · satisfied=${orUnknown(detail.num_satisfied_instances)}`;

  el("#td-viol-count").textContent = detail.violations.length;
  el("#td-sat-count").textContent = detail.satisfied.length;

  const vlist = el("#td-violations-list");
  vlist.innerHTML = "";
  if (!detail.violations.length) {
    vlist.innerHTML = "<div class='muted'>No violations flagged by the monitor.</div>";
  }
  for (const v of detail.violations) vlist.appendChild(renderViolation(v, detail.annotations));

  const slist = el("#td-satisfied-list");
  slist.innerHTML = "";
  for (const s of detail.satisfied) slist.appendChild(renderSatisfied(s, detail.annotations));

  resolveMarkCollisions(el("#td-monitor-body"));
  renderMissedPanel(detail, el("#td-monitor-body"), "td-missed-panel");
}

async function loadEpisodes(task, autoEpisode) {
  state.task = task;
  episodeList.innerHTML = "<div class='loading'>loading episodes…</div>";
  const data = await fetchJSON(`/api/episodes?task=${encodeURIComponent(task)}`);
  episodeList.innerHTML = "";
  const hint = document.createElement("div");
  hint.className = "rollout-dir";
  hint.textContent = `latest rollout: ${data.rollout_dir}`;
  episodeList.appendChild(hint);

  for (const ep of data.episodes) {
    const row = document.createElement("button");
    row.className = "ep-row" + (ep.success ? " success" : " failure");
    const violBadge = ep.num_violations
      ? `<span class="mini-badge viol">${ep.num_violations} viol</span>`
      : `<span class="mini-badge ok">0 viol</span>`;
    row.innerHTML = `<span class="ep-num">#${ep.episode}</span>
      <span class="mini-badge ${ep.success ? "s-ok" : "s-fail"}">${ep.success ? "success" : "fail"}</span>
      ${violBadge}`;
    row.addEventListener("click", () => selectEpisode(task, ep.episode, row));
    episodeList.appendChild(row);
    if (autoEpisode != null && String(ep.episode) === String(autoEpisode)) {
      selectEpisode(task, ep.episode, row);
    }
  }
}

function fmtTime(s) {
  if (s == null) return "?";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

// monitor frame number is what the raw privileged_information_*.json / LTL
// trace actually indexes by, so prefer showing that over wall-clock seconds
// wherever we're pointing at a specific predicate-trace frame.
function fmtFrame(marker) {
  if (!marker || marker.monitor_frame == null) return "?";
  return `f${marker.monitor_frame}`;
}

async function selectEpisode(task, episode, rowEl) {
  document.querySelectorAll(".ep-row").forEach((r) => r.classList.remove("active"));
  if (rowEl) rowEl.classList.add("active");

  state.episode = episode;
  emptyState.classList.add("hidden");
  episodeView.classList.remove("hidden");

  const detail = await fetchJSON(
    `/api/episode?task=${encodeURIComponent(task)}&episode=${episode}`
  );
  state.detail = detail;
  render(detail);
}

function verdictControls(group, index, current) {
  const wrap = document.createElement("div");
  wrap.className = "verdict-controls";
  const verdicts = [
    ["confirmed", "✓ confirmed"],
    ["disputed", "✗ disputed"],
    ["unsure", "? unsure"],
    ["unverifiable", "⦸ unverifiable"],
  ];
  for (const [val, label] of verdicts) {
    const b = document.createElement("button");
    b.className = "verdict-btn" + (current && current.verdict === val ? " active" : "");
    b.textContent = label;
    b.dataset.verdict = val;
    b.addEventListener("click", () => {
      wrap.querySelectorAll(".verdict-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      saveAnnotation(group, index, { verdict: val });
    });
    wrap.appendChild(b);
  }
  return wrap;
}

const DRAFT_VERDICT_LABEL = {
  confirmed: "✓ confirmed",
  disputed: "✗ disputed",
  unsure: "? unsure",
  unverifiable: "⦸ unverifiable",
};

function aiDraftBlock(current) {
  if (!current || !current.ai_draft) return null;
  const wrap = document.createElement("div");
  wrap.className = "ai-draft";
  const label = DRAFT_VERDICT_LABEL[current.ai_draft_verdict] || current.ai_draft_verdict || "";
  wrap.innerHTML = `<div class="ai-draft-head">Claude's draft take${
    label ? ` — <span class="ai-draft-chip">${label}</span>` : ""
  }</div>`;
  const body = document.createElement("div");
  body.className = "ai-draft-body";
  body.textContent = current.ai_draft;
  wrap.appendChild(body);
  return wrap;
}

function noteBox(group, index, current) {
  const ta = document.createElement("textarea");
  ta.className = "note-box";
  ta.placeholder = "reviewer notes (what you actually see in the video)…";
  ta.value = (current && current.note) || "";
  let debounce;
  ta.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => saveAnnotation(group, index, { note: ta.value }), 500);
  });
  return ta;
}

// Which (task, episode) key annotation writes go to -- swapped by whichever
// tab/episode is currently rendered (see setAnnotationContext), so the
// shared renderViolation/renderSatisfied/renderMissedPanel/verdictControls/
// noteBox functions below work for both tabs without a task/episode
// parameter threaded through every one of them. The eval tab uses the plain
// task name; the training tab uses a "training__<task>" key (matching
// server.py's api_training_monitor) so notes on a training episode never
// collide with an eval episode of the same task name.
const annotationContext = { task: null, episode: null };
function setAnnotationContext(task, episode) {
  annotationContext.task = task;
  annotationContext.episode = episode;
}

async function saveAnnotation(group, index, patch) {
  // group === null means an episode-level field (missed_notes, overall_verdict)
  await fetchJSON("/api/annotate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task: annotationContext.task,
      episode: annotationContext.episode,
      ...(group ? { group, index } : {}),
      ...patch,
    }),
  });
}

// While true, the mirrored-seek listeners in wireVideoSync() ignore "seeked"
// events -- set around any programmatic currentTime write that already
// positions *both* videos correctly on its own (jump-to-frame buttons), so
// the generic fraction-based mirroring doesn't clobber the precise
// monitor-frame-based position with a naive proportional guess.
let suppressSeekSync = false;
const evalSyncState = { wired: false };

// Which video(s) predicate-timeline/chip clicks seek -- swapped by whichever
// tab/episode is currently rendered (see setActiveVideos), same rationale as
// annotationContext above: keyFrameChip/fillTimeline/markTimeline all funnel
// through the single seekTo() below, so only this one function needed to
// stop hardcoding the eval tab's `video`/`reconVideo` elements.
const activeVideos = { primary: video, recon: null, reconFps: null };
function setActiveVideos(primaryEl, reconEl, reconFps) {
  activeVideos.primary = primaryEl;
  activeVideos.recon = reconEl || null;
  activeVideos.reconFps = reconFps || null;
}

function seekTo(marker) {
  if (!marker) return;
  suppressSeekSync = true;
  if (marker.time_s != null) {
    activeVideos.primary.currentTime = Math.max(0, marker.time_s - 0.5); // small pre-roll so onset is visible
    activeVideos.primary.play();
  }
  if (activeVideos.recon && marker.monitor_frame != null) {
    activeVideos.recon.currentTime = Math.max(0, marker.monitor_frame / activeVideos.reconFps - 0.2);
    activeVideos.recon.play();
  }
  setTimeout(() => { suppressSeekSync = false; }, 100);
}

// Generalized version of the original two-video sync (originally hardcoded
// to `video`/`reconVideo`) -- also used by the Training Data tab's
// original/reconstructed pair. `syncState` is a plain `{wired: false}` bag
// so each pair gets its own "already wired" latch instead of sharing one
// global flag (which would silently no-op the second pair's wiring).
function wireSync(primary, secondary, btnSel, syncState) {
  if (syncState.wired) return;
  syncState.wired = true;

  const btn = el(btnSel);
  const updateBtnLabel = () => {
    const playing = !primary.paused && !primary.ended;
    btn.textContent = playing ? "⏸ Pause both" : "▶ Play both";
  };

  btn.addEventListener("click", () => {
    if (primary.paused) {
      primary.play();
      secondary.play();
    } else {
      primary.pause();
      secondary.pause();
    }
  });

  function syncTo(source, target, threshold) {
    if (!isFinite(source.duration) || !isFinite(target.duration) || source.duration === 0) return;
    const fraction = source.currentTime / source.duration;
    const targetTime = fraction * target.duration;
    if (Math.abs(target.currentTime - targetTime) > threshold) {
      suppressSeekSync = true;
      target.currentTime = targetTime;
      setTimeout(() => {
        suppressSeekSync = false;
      }, 100);
    }
  }

  function mirror(source, target) {
    source.addEventListener("play", () => {
      target.play();
      updateBtnLabel();
    });
    source.addEventListener("pause", () => {
      target.pause();
      updateBtnLabel();
    });
    source.addEventListener("seeked", () => {
      if (suppressSeekSync) return;
      syncTo(source, target, 0.05);
    });
  }

  mirror(primary, secondary);
  mirror(secondary, primary);

  // Discrete play/pause/seek mirroring alone isn't enough: once both are
  // actually playing, each <video> runs its own independent decode clock,
  // and a lower-fps side (e.g. the reconstructed video's ~1.26fps long,
  // chunky frames vs. the original's 10fps in the eval tab) visibly drifts
  // apart within seconds even though they started in sync. Continuously
  // re-anchor secondary to primary (not the reverse -- primary has more
  // frames/updates, so it's the more reliable clock) during playback. The
  // 0.15s threshold keeps this from fighting the natural per-frame jitter of
  // a <video> element or fighting the discrete mirroring above.
  primary.addEventListener("timeupdate", () => {
    if (!suppressSeekSync && !secondary.paused) {
      syncTo(primary, secondary, 0.15);
    }
  });

  updateBtnLabel();
}

function keyFrameChip(chip) {
  const btn = document.createElement("button");
  btn.className = "chip";
  btn.title = `video frame ${chip.video_frame} · t=${fmtTime(chip.time_s)}`;
  btn.innerHTML = `<span class="chip-label">${chip.label}</span><span class="chip-time">${fmtFrame(chip)}</span>`;
  btn.addEventListener("click", () => seekTo(chip));
  return btn;
}

function predicateRow(p, indent) {
  const row = document.createElement("div");
  row.className = "predicate-row" + (indent ? " sub" : "") + (p.is_ltl_summary ? " ltl-summary-row" : "");

  const label = document.createElement("div");
  label.className = "predicate-label" + (p.is_decomposed_extra ? " extra" : "");
  label.textContent = (indent ? "└ " : "") + p.label;
  // Hover tooltip: the real key plus its human-readable description (if
  // any) -- the visible text is always the real key (see server.py's
  // node_for), the description is supplementary, not a substitute.
  label.title = p.description ? `${p.key} — ${p.description}` : p.key;
  row.appendChild(label);

  const timeline = document.createElement("div");
  timeline.className = "predicate-timeline";
  row.appendChild(timeline);
  return { row, timeline };
}

// One stable color per distinct categorical value (e.g. object name), not
// tied to true/false -- picked from a fixed palette by a simple string
// hash so the same object always gets the same color within one episode
// (and typically across episodes too, though that's not guaranteed).
const _CATEGORICAL_COLORS = [
  "#5b7fff", "#3bb273", "#e0a13a", "#d64545", "#9b6bd6", "#3ab0c9", "#c9843a", "#7a9e3b",
];
function _categoricalColor(value) {
  if (value == null) return "var(--bg-alt2)";
  let h = 0;
  for (let i = 0; i < value.length; i++) h = (h * 31 + value.charCodeAt(i)) >>> 0;
  return _CATEGORICAL_COLORS[h % _CATEGORICAL_COLORS.length];
}

function fillTimeline(timeline, p, span) {
  for (const r of p.runs) {
    const width = ((r.end_frame - r.start_frame + 1) / span) * 100;
    const seg = document.createElement("button");
    if (p.is_categorical) {
      // Which object (e.g. "bread" vs "basket") a predicate was actually
      // about at each frame -- object identity, not a boolean -- so shown
      // as a colored-by-value segment with the name itself as visible
      // text (not just true/false coloring) whenever there's room.
      seg.className = "predicate-run run-categorical";
      seg.style.width = `${width}%`;
      seg.style.background = _categoricalColor(r.value);
      seg.textContent = r.value || "";
    } else {
      seg.className =
        "predicate-run " +
        (r.value === true ? "run-true" : r.value === false ? "run-false" : "run-unknown");
      seg.style.width = `${width}%`;
    }
    seg.title =
      `${p.label}: ${r.value === null ? "n/a" : r.value} · ` +
      `monitor frames ${r.start_frame}-${r.end_frame} · t=${fmtTime(r.start.time_s)}-${fmtTime(r.end.time_s)}`;
    seg.addEventListener("click", () => seekTo(r.start));
    timeline.appendChild(seg);
  }
}

const MARK_GLYPH = { start: "▸", violated: "✗", end: "◆" };

// Line-height constants for the mark labels (kept in one place since the
// layout math below needs to reproduce them exactly -- see the matching
// font-size/line-height rules in style.css's .bar-mark-object/-frame/-glyph).
const MARK_LINE = { object: 10, frame: 10, glyph: 9 };
const MARK_TIER_GAP = 6; // vertical breathing room between stacked tiers
const BAR_HEIGHT = 16; // must match .predicate-timeline's CSS height

// Bottom-to-top stacking order. A property/card only reserves space for the
// tiers it actually uses -- e.g. a satisfied property (or a violation
// instance whose window never violates) has no "violated" marks at all, so
// that tier -- and the gap it would've needed -- is skipped entirely rather
// than leaving a blank reserved gap.
const TIER_ORDER = ["end", "violated", "start"];

// Figures out, from the actual marks a card has, how tall each present tier
// really needs to be (a "start" mark only grows a 3rd line when it actually
// carries an object/fixture name) and stacks only the tiers that are
// present -- so both the mark position *and* the row spacing above the bar
// adapt to what's actually being drawn, instead of a fixed guess.
function computeMarkTierLayout(marks) {
  const present = TIER_ORDER.filter((k) => marks.some((m) => m.kind === k));
  const bottoms = {};
  let cursor = BAR_HEIGHT; // next tier's bottom offset, measured from the bar's own top edge
  for (const kind of present) {
    const hasObject = kind === "start" && marks.some((m) => m.kind === "start" && m.object);
    const height = (hasObject ? MARK_LINE.object : 0) + MARK_LINE.frame + MARK_LINE.glyph;
    bottoms[kind] = cursor;
    cursor += height + MARK_TIER_GAP;
  }
  // margin needed above the bar to fit every present tier, plus a small
  // buffer so the topmost label doesn't butt right up against whatever is
  // above the whole predicate-breakdown block
  const marginTop = present.length ? cursor - MARK_TIER_GAP + 6 : 4;
  return { bottoms, marginTop };
}

// Overlay dashed tick marks on top of an (already-filled) bar — same set of
// marks on every row so they line up vertically like a shared time axis.
// Consecutive violated frames were already collapsed server-side to one mark
// at the transition, so this never draws more than one tick per event.
function markTimeline(timeline, marks, window_start, window_end, span, tierBottoms) {
  for (const m of marks) {
    if (m.frame < window_start || m.frame > window_end) continue; // outside this bar's window
    const left = ((m.frame - window_start) / span) * 100;
    const bottom = tierBottoms[m.kind];
    if (bottom == null) continue; // shouldn't happen -- layout is built from these same marks
    const tick = document.createElement("button");
    tick.className = "bar-mark bar-mark-" + m.kind;
    tick.style.left = `${left}%`;
    tick.style.bottom = `${bottom}px`;
    // the dashed line always reaches down to the bar's own bottom edge, so
    // its length is exactly this tier's "bottom" offset from that edge.
    tick.style.setProperty("--dash-height", `${bottom}px`);
    const obj = m.object ? `${m.object}: ` : "";
    tick.title = `${obj}${m.label} @ f${m.frame}` + (m.reason ? ` — ${m.reason}` : "");
    if (m.marker) tick.addEventListener("click", (e) => { e.stopPropagation(); seekTo(m.marker); });

    // object identity only needs stating once per occurrence -- shown on the
    // "start" mark (where the occurrence begins), since violated/end marks
    // further right on the same bar are implicitly about that same object.
    if (m.kind === "start" && m.object) {
      const objLabel = document.createElement("span");
      objLabel.className = "bar-mark-object";
      objLabel.textContent = m.object;
      tick.appendChild(objLabel);
    }

    const frameLabel = document.createElement("span");
    frameLabel.className = "bar-mark-frame";
    frameLabel.textContent = `f${m.frame}`;
    tick.appendChild(frameLabel);

    const glyph = document.createElement("span");
    glyph.className = "bar-mark-glyph";
    glyph.textContent = MARK_GLYPH[m.kind] || "•";
    tick.appendChild(glyph);

    timeline.appendChild(tick);
  }
}

// margin-top for a decomposed atom/sub-predicate row -- these only ever show
// a single small tier (frame number + a rise/fall glyph, no stacking, no
// object-name line), unlike the whole-LTL summary row's occurrence-level
// marks, so this stays a constant instead of running through
// computeMarkTierLayout().
const OWN_TRANSITION_MARGIN_TOP = 40;

// Every bar below the whole-LTL summary shows *its own* true/false
// transitions -- where this specific atom/sub-predicate's own value flips --
// rather than the shared occurrence-level start/violated/end marks (that's
// what the top summary bar is for). Derived straight from this predicate's
// own `runs` (already computed server-side), so it's always exactly this
// bar's own transitions, never another bar's. The first run's boundary is
// the edge of the display window, not a real transition, so it's skipped.
function markOwnTransitions(timeline, p, span, window_start) {
  if (!p.runs || p.runs.length < 2) return;
  for (let i = 1; i < p.runs.length; i++) {
    const run = p.runs[i];
    const frame = run.start_frame;
    const left = ((frame - window_start) / span) * 100;
    const cls = run.value === true ? "bar-mark-rise" : run.value === false ? "bar-mark-fall" : "bar-mark-unknown";
    const tick = document.createElement("button");
    tick.className = "bar-mark " + cls;
    tick.style.left = `${left}%`;
    tick.style.bottom = `${BAR_HEIGHT}px`;
    tick.style.setProperty("--dash-height", `${BAR_HEIGHT}px`);
    const verb = run.value === true ? "became true" : run.value === false ? "became false" : "became n/a";
    tick.title = `${p.label} ${verb} @ f${frame}`;
    tick.addEventListener("click", (e) => { e.stopPropagation(); seekTo(run.start); });

    const frameLabel = document.createElement("span");
    frameLabel.className = "bar-mark-frame";
    frameLabel.textContent = `f${frame}`;
    tick.appendChild(frameLabel);

    const glyph = document.createElement("span");
    glyph.className = "bar-mark-glyph";
    glyph.textContent = run.value === true ? "▲" : run.value === false ? "▼" : "•";
    tick.appendChild(glyph);

    timeline.appendChild(tick);
  }
}

// Vertical step used to bump a mark up by one collision-avoidance lane,
// measured (not guessed) from the same line-height constants the marks'
// own CSS uses.
const MARK_COLLISION_STEP = MARK_LINE.frame + MARK_LINE.glyph + MARK_TIER_GAP;

// Own-transition marks (and, less often, several same-kind LTL-summary
// marks) can land close enough together that their frame-number labels
// visually run into each other -- how close depends on the actual rendered
// pixel width of the bar and of each label, which varies with the window's
// physical width and isn't something CSS/position-math alone can predict.
// So instead of guessing a minimum frame gap, this is a real post-layout
// pass: once every card is in the live document (so getBoundingClientRect
// reflects actual on-screen geometry), group each bar's marks by which tier
// they're already sitting at, and for any two that actually overlap
// horizontally, push the later one up by a lane until it doesn't.
function resolveMarkCollisions(container) {
  container.querySelectorAll(".predicate-timeline").forEach((timeline) => {
    const marks = Array.from(timeline.querySelectorAll(".bar-mark"));
    if (marks.length < 2) return;
    const byBase = new Map();
    for (const m of marks) {
      const base = parseFloat(m.style.bottom) || 0;
      if (!byBase.has(base)) byBase.set(base, []);
      byBase.get(base).push(m);
    }
    for (const group of byBase.values()) {
      if (group.length < 2) continue;
      group.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);
      const laneRightEdge = [];
      for (const m of group) {
        const rect = m.getBoundingClientRect();
        let lane = 0;
        while (laneRightEdge[lane] != null && rect.left < laneRightEdge[lane]) lane++;
        laneRightEdge[lane] = rect.right;
        if (lane > 0) {
          const base = parseFloat(m.style.bottom) || 0;
          const bumped = base + lane * MARK_COLLISION_STEP;
          m.style.bottom = `${bumped}px`;
          m.style.setProperty("--dash-height", `${bumped}px`);
        }
      }
    }

    // 3+ marks crowding together can need more lanes than the row's default
    // margin-top budgeted for (that default only assumes each mark's normal,
    // uncollided tier) -- grow the row to fit however tall this bar's marks
    // actually ended up, using each mark's own real rendered height rather
    // than a guessed constant.
    let neededMarginTop = 0;
    for (const m of marks) {
      const bottom = parseFloat(m.style.bottom) || 0;
      neededMarginTop = Math.max(neededMarginTop, bottom + m.getBoundingClientRect().height + 8);
    }
    const row = timeline.closest(".predicate-row");
    if (row && neededMarginTop > (parseFloat(row.style.marginTop) || 0)) {
      row.style.marginTop = `${neededMarginTop}px`;
    }
  });
}

// The whole-LTL bar (green/red) comes first, its decomposition follows
// indented underneath. The LTL bar carries the shared occurrence-level
// start/violated/end marks; every bar below it instead highlights its own
// true/false transitions (see markOwnTransitions).
function predicateBreakdown(pb) {
  if (!pb || !pb.predicates.length) return null;
  const wrap = document.createElement("div");
  wrap.className = "predicate-breakdown";

  const { start_frame, end_frame } = pb.window;
  const span = Math.max(1, end_frame - start_frame + 1);
  const marks = pb.marks || [];
  const { bottoms, marginTop } = computeMarkTierLayout(marks);

  for (const p of pb.predicates) {
    const { row, timeline } = predicateRow(p, false);
    fillTimeline(timeline, p, span);
    if (p.is_ltl_summary) {
      if (pb.pattern_blurb) row.title = pb.pattern_blurb;
      row.style.marginTop = `${marginTop}px`;
      markTimeline(timeline, marks, start_frame, end_frame, span, bottoms);
    } else {
      row.style.marginTop = `${OWN_TRANSITION_MARGIN_TOP}px`;
      // Categorical rows (active_object/settle_obj_name) already show their
      // value as visible text on each segment (see fillTimeline) -- skip
      // the true/false/n/a transition ticks, which only make sense for
      // boolean predicates.
      if (!p.is_categorical) markOwnTransitions(timeline, p, span, start_frame);
    }
    wrap.appendChild(row);
    for (const sub of p.subs || []) {
      const { row: subRow, timeline: subTimeline } = predicateRow(sub, true);
      subRow.style.marginTop = `${OWN_TRANSITION_MARGIN_TOP}px`;
      fillTimeline(subTimeline, sub, span);
      markOwnTransitions(subTimeline, sub, span, start_frame);
      wrap.appendChild(subRow);
    }
  }
  return wrap;
}

function ltlLine(ltl) {
  if (!ltl) return null;
  const line = document.createElement("div");
  line.className = "ltl-line";
  line.innerHTML = `<span class="ltl-tag">LTL</span><code>${ltl}</code>`;
  return line;
}

function renderViolation(v, ann) {
  const current = ann.violations[String(v.index)];
  const card = document.createElement("div");
  card.className = "card violation-card";

  const head = document.createElement("div");
  head.className = "card-head";
  head.innerHTML = `<strong>${v.property_name}</strong>`;
  card.appendChild(head);

  const desc = document.createElement("div");
  desc.className = "card-desc";
  desc.textContent = v.property_description;
  card.appendChild(desc);

  const ltl = ltlLine(v.ltl);
  if (ltl) card.appendChild(ltl);

  if (v.key_frames && v.key_frames.length) {
    const chips = document.createElement("div");
    chips.className = "chip-row";
    for (const c of v.key_frames) chips.appendChild(keyFrameChip(c));
    card.appendChild(chips);
  }

  const pb = predicateBreakdown(v.predicate_breakdown);
  if (pb) card.appendChild(pb);

  const details = document.createElement("details");
  details.className = "raw-explanation";
  const summary = document.createElement("summary");
  summary.textContent = "raw monitor explanation text";
  details.appendChild(summary);
  const expl = document.createElement("div");
  expl.className = "card-explanation";
  expl.textContent = v.explanation;
  details.appendChild(expl);
  card.appendChild(details);

  const draft = aiDraftBlock(current);
  if (draft) card.appendChild(draft);

  card.appendChild(verdictControls("violations", v.index, current));
  card.appendChild(noteBox("violations", v.index, current));
  return card;
}

function renderSatisfied(s, ann) {
  const card = document.createElement("div");
  card.className = "card satisfied-card";
  const current = ann.satisfied[String(s.index)];

  const head = document.createElement("div");
  head.className = "card-head";
  head.innerHTML = `<strong>${s.property_name}</strong>`;
  card.appendChild(head);

  const desc = document.createElement("div");
  desc.className = "card-desc";
  desc.textContent = s.property_description;
  card.appendChild(desc);

  const ltl = ltlLine(s.ltl);
  if (ltl) card.appendChild(ltl);

  const note = document.createElement("div");
  note.className = "card-hint";
  note.textContent =
    "Monitor says this held for the whole episode — the breakdown below covers the full " +
    "trace, not just one moment. Scrub through it (and the video) to confirm it never breaks.";
  card.appendChild(note);

  const pb = predicateBreakdown(s.predicate_breakdown);
  if (pb) card.appendChild(pb);

  const draft = aiDraftBlock(current);
  if (draft) card.appendChild(draft);

  card.appendChild(verdictControls("satisfied", s.index, current));
  card.appendChild(noteBox("satisfied", s.index, current));
  return card;
}

function wireFrameReadoutFor(videoEl, readoutId, fps, ratio, frameCount, monitorFrameCount) {
  let readout = el(`#${readoutId}`);
  if (!readout) {
    readout = document.createElement("div");
    readout.id = readoutId;
    readout.className = "frame-readout";
    videoEl.insertAdjacentElement("afterend", readout);
  }
  const update = () => {
    const t = videoEl.currentTime || 0;
    const videoFrame = Math.round(t * fps);
    const monitorFrame = Math.round(videoFrame / ratio);
    readout.textContent =
      `video frame ${videoFrame}` +
      (frameCount ? ` / ${frameCount}` : "") +
      ` · t=${t.toFixed(2)}s · ≈ monitor frame ${monitorFrame}` +
      (monitorFrameCount ? ` / ${monitorFrameCount}` : "");
  };
  videoEl.ontimeupdate = update;
  videoEl.onseeked = update;
  videoEl.onloadedmetadata = update;
  update();
}

function wireFrameReadout(detail) {
  wireFrameReadoutFor(
    video, "frame-readout",
    detail.fps || 10, detail.ratio || 8,
    detail.video_frame_count, detail.monitor_num_frames
  );
  if (detail.reconstruction) {
    // reconstruction has exactly one rendered frame per monitor frame, at its own fps
    // (derived from the original video's duration by reconstruct_video.py, so both
    // videos play the same real-time length -- see replay/README.md)
    wireFrameReadoutFor(
      reconVideo, "recon-frame-readout",
      detail.reconstruction.fps, 1,
      detail.monitor_num_frames, detail.monitor_num_frames
    );
  }
}

function renderComparisonSummary(comparison) {
  if (!comparison) {
    return `<div class="card-hint">Reconstructed video found, but no comparison report
      (run replay/compare_frames.py to generate one).</div>`;
  }
  const skipped = comparison.num_skipped_corrupted_original || 0;
  const madAvg = comparison.mean_abs_diff_avg != null ? comparison.mean_abs_diff_avg : "?";
  const ssimAvg = comparison.ssim_avg != null ? comparison.ssim_avg : "n/a";
  return `
    <div class="recon-summary-head">Reconstruction vs. original (frame-by-frame)</div>
    <div class="recon-stats">
      <span class="recon-stat"><strong>${madAvg}</strong> mean abs pixel diff (0-255)</span>
      <span class="recon-stat"><strong>${ssimAvg}</strong> mean SSIM</span>
      <span class="recon-stat"><strong>${comparison.num_compared}</strong> frames compared</span>
      ${skipped ? `<span class="recon-stat warn"><strong>${skipped}</strong> skipped — original frame corrupted (noise)</span>` : ""}
    </div>
    <p class="card-hint">
      Reconstruction is posed directly from the recorded per-frame state (robot/object/fixture
      poses), not replayed from actions — so mismatch here is camera/calibration noise, not
      accumulated drift. See replay/README.md for what's known to cause the residual diff.
    </p>
  `;
}

function render(detail) {
  setAnnotationContext(detail.task, detail.episode);

  el("#ep-title").textContent = `${detail.task} — episode ${detail.episode}`;
  const badge = el("#ep-badge");
  badge.textContent = detail.success ? "SUCCESS" : "FAILURE";
  badge.className = "badge " + (detail.success ? "badge-ok" : "badge-fail");
  el("#ep-description").textContent = detail.task_description || "";
  el("#ep-meta").textContent =
    `rollout: ${detail.rollout_dir} · fps=${detail.fps} · ` +
    `video frames≈${detail.video_frame_count} · monitor frames=${detail.monitor_num_frames} · ` +
    `ratio=${detail.ratio ? detail.ratio.toFixed(3) : "?"} · ` +
    `violated=${detail.num_violated_instances} · satisfied=${detail.num_satisfied_instances}`;

  video.src = detail.video_url;
  video.load();

  const reconCol = el("#recon-col");
  const reconPanel = el("#recon-panel");
  const syncRow = el("#sync-row");
  if (detail.reconstruction) {
    reconCol.classList.remove("hidden");
    reconVideo.src = detail.reconstruction.video_url;
    reconVideo.load();
    reconPanel.classList.remove("hidden");
    reconPanel.innerHTML = renderComparisonSummary(detail.reconstruction.comparison);
    syncRow.classList.remove("hidden");
    wireSync(video, reconVideo, "#sync-play-btn", evalSyncState);
    setActiveVideos(video, reconVideo, detail.reconstruction.fps);
  } else {
    reconCol.classList.add("hidden");
    reconPanel.classList.add("hidden");
    reconPanel.innerHTML = "";
    syncRow.classList.add("hidden");
    setActiveVideos(video, null, null);
  }
  wireFrameReadout(detail);

  el("#viol-count").textContent = detail.violations.length;
  el("#sat-count").textContent = detail.satisfied.length;

  const vlist = el("#violations-list");
  vlist.innerHTML = "";
  if (!detail.violations.length) {
    vlist.innerHTML = "<div class='muted'>No violations flagged by the monitor.</div>";
  }
  for (const v of detail.violations) vlist.appendChild(renderViolation(v, detail.annotations));

  const slist = el("#satisfied-list");
  slist.innerHTML = "";
  for (const s of detail.satisfied) slist.appendChild(renderSatisfied(s, detail.annotations));

  resolveMarkCollisions(episodeView);
  renderMissedPanel(detail, episodeView, "missed-panel");
}

// `containerEl`/`panelId` let this be reused by the training tab's own
// episode-view container with its own panel id, instead of always appending
// to the eval tab's #episode-view (see renderTrainingMonitor).
function renderMissedPanel(detail, containerEl, panelId) {
  let panel = el(`#${panelId}`);
  if (!panel) {
    panel = document.createElement("section");
    panel.id = panelId;
    panel.className = "missed-panel";
    containerEl.appendChild(panel);
  }
  panel.innerHTML = `
    <h3>Reviewer: anything the monitor missed?</h3>
    <p class="card-hint">
      Watch the whole video (not just the jump buttons above) and note any safety issue
      you can see that is <em>not</em> in the violations list, or any listed violation/satisfied
      claim that looks wrong once you actually look at the footage.
    </p>
  `;
  const ta = document.createElement("textarea");
  ta.className = "note-box missed-box";
  ta.placeholder = "e.g. \"gripper clips the microwave door at ~1:12, not flagged\"";
  ta.value = detail.annotations.missed_notes || "";
  let debounce;
  ta.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(
      () => saveAnnotation(null, null, { missed_notes: ta.value }),
      500
    );
  });
  panel.appendChild(ta);

  const overallWrap = document.createElement("div");
  overallWrap.className = "overall-verdict";
  const overallLabel = document.createElement("span");
  overallLabel.textContent = "Overall episode verdict: ";
  overallWrap.appendChild(overallLabel);
  for (const [val, label] of [
    ["matches", "monitor output matches video"],
    ["mismatches", "monitor output does not match video"],
    ["partial", "partially matches"],
  ]) {
    const b = document.createElement("button");
    b.className =
      "verdict-btn" +
      (detail.annotations.overall_verdict === val ? " active" : "");
    b.textContent = label;
    b.addEventListener("click", async () => {
      overallWrap.querySelectorAll(".verdict-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      await saveAnnotation(null, null, { overall_verdict: val });
    });
    overallWrap.appendChild(b);
  }
  panel.appendChild(overallWrap);
}

init();
