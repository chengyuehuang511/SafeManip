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
// Which simulator's training data the tab is currently showing ("robocasa" |
// "libero") -- see server.py's sim= query param on /api/td_tasks,
// /api/td_episodes, /api/training_monitor, /api/training_monitor_methods.
// LIBERO has no ground-truth video reconstruction yet (see server.py's
// PREDICATES_PY_PATH_LIBERO comment) and no violation-count tree endpoint
// yet (/api/training_violation_counts stays RoboCasa-only) -- both are
// handled by branching on tdSim below, not by pretending LIBERO has them.
let tdSim = "robocasa";
function tdSimQS() {
  return `&sim=${encodeURIComponent(tdSim)}`;
}

// Multi-annotator support (2026-09-10): who's currently "logged in" as far
// as annotation reads/writes go -- persisted in localStorage so it survives
// reloads/new tabs on the same browser, but is otherwise just a plain-text
// identity (no real auth; this is an internal review tool). Threaded into
// every annotation-touching fetch as &annotator=<name> (GET) or
// {annotator: <name>} (POST /api/annotate) -- see server.py's
// register_annotator/annotation_path, which give each annotator their own
// subdirectory under viewer/annotations/, so two people's verdicts on the
// same episode never overwrite each other. null until loadAnnotators()
// resolves on page load (falls back to the server's DEFAULT_ANNOTATOR,
// "chengyue", the pre-existing owner of all annotations made before this
// feature existed).
let currentAnnotator = localStorage.getItem("safemanip-annotator") || null;
function annotatorQS() {
  return currentAnnotator ? `&annotator=${encodeURIComponent(currentAnnotator)}` : "";
}
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

const ALL_ANNOTATORS_VALUE = "__all__";

// Populates the header's Annotator select from /api/annotators, wires the
// "register new" flow, and reacts to a change by re-fetching whatever's
// currently on screen under the newly-selected identity/filter. Called
// once on page load; the select itself persists across tab/sim switches
// (it's outside both #screen-eval/#screen-training).
async function initAnnotatorPicker() {
  const select = el("#annotator-select");
  const registerBtn = el("#annotator-register-btn");

  async function refreshOptions(selectValue) {
    let data;
    try {
      data = await fetchJSON("/api/annotators");
    } catch (e) {
      data = { annotators: [], default: "chengyue" };
    }
    if (!currentAnnotator) {
      currentAnnotator = data.default;
      localStorage.setItem("safemanip-annotator", currentAnnotator);
    }
    const names = data.annotators.length ? data.annotators : [data.default];
    select.innerHTML = "";
    const allOpt = document.createElement("option");
    allOpt.value = ALL_ANNOTATORS_VALUE;
    allOpt.textContent = "All annotators (view only)";
    select.appendChild(allOpt);
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    }
    select.value = selectValue || currentAnnotator;
    if (select.value !== (selectValue || currentAnnotator)) {
      // requested value isn't a real <option> (e.g. a stale localStorage
      // name from before a corpus reset) -- fall back rather than silently
      // showing the browser's own "nothing selected" default.
      select.value = currentAnnotator;
    }
  }

  await refreshOptions(currentAnnotator);

  select.addEventListener("change", async () => {
    currentAnnotator = select.value;
    // "All annotators" is a view-only filter, never a save identity -- but
    // there's nothing to *save* here, just re-render the current screen
    // scoped to the new selection, same as switching to a real name.
    if (currentAnnotator !== ALL_ANNOTATORS_VALUE) {
      localStorage.setItem("safemanip-annotator", currentAnnotator);
    }
    if (el("#tab-training").classList.contains("active")) {
      await refreshViolationCounts();
      renderTaskTree();
      renderPropertyTree();
      if (tdState.task && tdState.episode != null) {
        await loadTrainingMonitor(tdState.task, tdState.episode, tdState.monitorMethod);
      }
    } else if (state.task && state.episode != null) {
      await selectEpisode(state.task, state.episode);
    }
  });

  registerBtn.addEventListener("click", async () => {
    const name = (prompt("New annotator name:") || "").trim();
    if (!name) return;
    const data = await fetchJSON("/api/register_annotator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    currentAnnotator = data.annotator;
    localStorage.setItem("safemanip-annotator", currentAnnotator);
    await refreshOptions(currentAnnotator);
    select.dispatchEvent(new Event("change"));
  });
}

// URL sync (2026-09-10): reflects which top-level tab (eval/training) and,
// for the training tab, which sim (robocasa/libero) is active in
// `?tab=&sim=` query params -- via history.pushState/popstate, not a real
// server-side route (single-page app stays as one page/one app.js; see the
// explicit scope decision this was built to). Fixes a real, reported
// pain point: with everything living in one URL and only in-memory JS
// state, two people (or two tabs) sharing the same link always land on
// the same default view, and there was no way to bookmark/share "the
// LIBERO training tab" specifically -- easy for different sims'/tabs'
// state to visually "overwrite" each other with no address-bar cue which
// one you're looking at. Deliberately scoped to tab+sim only (not task/
// episode/property/method) -- the existing ?task=&episode= deep link for
// the eval tab is untouched/orthogonal.
function currentTabSimParams() {
  const params = new URLSearchParams(location.search);
  params.set("tab", el("#tab-training").classList.contains("active") ? "training" : "eval");
  if (el("#tab-training").classList.contains("active")) {
    params.set("sim", tdSim);
  } else {
    params.delete("sim");
  }
  return params;
}

function syncUrl() {
  const params = currentTabSimParams();
  const next = `${location.pathname}?${params.toString()}`;
  if (next !== `${location.pathname}${location.search}`) {
    history.pushState(null, "", next);
  }
}

function activateEvalTab() {
  el("#tab-eval").classList.add("active");
  el("#tab-training").classList.remove("active");
  el("#screen-eval").classList.remove("hidden");
  el("#screen-training").classList.add("hidden");
  el("#root-path").textContent = roots.eval || "";
}

function activateTrainingTab() {
  el("#tab-training").classList.add("active");
  el("#tab-eval").classList.remove("active");
  el("#screen-training").classList.remove("hidden");
  el("#screen-eval").classList.add("hidden");
  el("#root-path").textContent = roots.training || "loading…";
  if (!tdState.loaded) {
    tdState.loaded = true;
    initTrainingData();
  }
}

// Wires the header's "📖 guide" button to the in-page annotator guide modal
// (2026-09-10) -- see index.html's #annotator-guide-backdrop and style.css's
// .modal-backdrop/.modal-window.guide rules. Closes on the ✕ button, on a
// click outside the window (backdrop itself), or on Escape while open.
function initAnnotatorGuideModal() {
  const backdrop = el("#annotator-guide-backdrop");
  const openBtn = el("#annotator-guide-btn");
  const closeBtn = el("#annotator-guide-close");
  if (!backdrop || !openBtn || !closeBtn) return;
  const open = () => backdrop.classList.remove("hidden");
  const close = () => backdrop.classList.add("hidden");
  openBtn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) close();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !backdrop.classList.contains("hidden")) close();
  });
}

function initTabs() {
  el("#tab-eval").addEventListener("click", () => {
    activateEvalTab();
    syncUrl();
  });
  el("#tab-training").addEventListener("click", () => {
    activateTrainingTab();
    syncUrl();
  });
  // Browser back/forward: re-derive tab+sim from the URL we just
  // navigated to and apply it, instead of leaving the page showing
  // whatever it happened to already be showing (pushState alone doesn't
  // do this -- only a real navigation event, which popstate is). Toggles
  // the tab DOM directly (not via activateTrainingTab, whose own
  // "!tdState.loaded" check means something narrower here -- "has *any*
  // sim's training data ever been loaded", not "does the *current* sim
  // need loading", which is what a sim change via back/forward needs).
  window.addEventListener("popstate", () => {
    const params = new URLSearchParams(location.search);
    const wantTab = params.get("tab") || "eval";
    const wantSim = params.get("sim") || "robocasa";
    if (wantTab !== "training") {
      activateEvalTab();
      return;
    }
    const needsLoad = !tdState.loaded || wantSim !== tdSim;
    if (wantSim !== tdSim) {
      tdSim = wantSim;
      el("#td-sim-select").value = wantSim;
      tdState.task = null;
      tdState.episode = null;
      tdState.monitorMethod = null;
      tdMethodsLoaded = false;
    }
    tdState.loaded = true;
    el("#tab-training").classList.add("active");
    el("#tab-eval").classList.remove("active");
    el("#screen-training").classList.remove("hidden");
    el("#screen-eval").classList.add("hidden");
    el("#root-path").textContent = roots.training || "loading…";
    if (needsLoad) {
      initTrainingData();
    }
  });
}

async function init() {
  initTheme();
  await initAnnotatorPicker();
  initAnnotatorGuideModal();
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

  // ?tab=training&sim=libero deep-links straight into the Training Data
  // tab, already scoped to the requested sim -- same bookmark/share intent
  // as ?task=&episode= above, just for the other tab. Applied last (after
  // the eval tab's own always-loaded default state above), and via
  // history.replaceState (not pushState -- this is establishing the
  // *initial* URL/state pairing, not a new navigation entry a "back"
  // button should ever land on).
  const initialParams = new URLSearchParams(location.search);
  if ((initialParams.get("tab") || "eval") === "training") {
    const wantSim = initialParams.get("sim") || "robocasa";
    tdSim = wantSim;
    el("#td-sim-select").value = wantSim;
    activateTrainingTab();
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
// Property names whose trigger mechanism never fires anywhere in the
// current sim's corpus (LIBERO only, e.g. rc_press_preconditions_safe --
// no push-button fixture in any of the 40 tasks) -- see server.py's
// LIBERO_INACTIVE_PROPERTIES for why this is a maintained list, not
// computed live. Empty for RoboCasa.
let tdInactiveProperties = new Set();
// { by_task: {task: {total, by_property: {prop: count}}},
//   by_property: {prop: {total, by_task: {task: count}}} } for whichever
// method is currently selected -- see server.py's training_violation_counts.
let tdViolationCounts = { by_task: {}, by_property: {} };

// Bumped at the start of every initTrainingData() call (initial tab load AND
// every sim switch); each call captures its own snapshot at entry and
// re-checks it after every await, bailing out silently if a newer call has
// since started. Fixes a real, confirmed race: /api/training_violation_counts
// for RoboCasa's 50-task corpus can take ~30s (no caching, scans every
// episode's monitor json fresh each time) -- switching to LIBERO well before
// that resolves used to leave the correct (empty) LIBERO render on screen
// only until the stale RoboCasa fetch finally completed and unconditionally
// overwrote tdViolationCounts/tdTasksList/etc. + re-rendered with RoboCasa's
// real numbers, ~30s after the switch, with no visible cause -- confirmed via
// a scripted repro (switch sim at t=0.5s, RoboCasa's real 119-violation tree
// reappeared at t=35s). A plain tdSim-at-call-time comparison isn't quite
// enough on its own (two rapid switches back to the same sim string would
// look "not stale" by that check alone) -- a monotonic generation counter
// is the standard fix for this class of async race.
let tdLoadGeneration = 0;

async function initTrainingData() {
  const myGeneration = ++tdLoadGeneration;
  const data = await fetchJSON(`/api/td_tasks?sim=${encodeURIComponent(tdSim)}`);
  if (myGeneration !== tdLoadGeneration) return;  // superseded by a newer tab-load/sim-switch
  roots.training = data.dataset_root;
  el("#root-path").textContent = data.dataset_root;
  tdTasksList = data.tasks;
  await ensureTrainingMonitorMethods();  // need a method selected before the trees can show violation counts
  if (myGeneration !== tdLoadGeneration) return;
  await initTrainingLtlPropertyList();
  if (myGeneration !== tdLoadGeneration) return;
  await refreshViolationCounts();
  if (myGeneration !== tdLoadGeneration) return;
  renderTaskTree();
  renderPropertyTree();
  if (tdTasksList.length) {
    loadTrainingEpisodes(tdTasksList[0].task);
  } else {
    tdEmptyState.textContent = tdSim === "libero"
      ? "No LIBERO training-data output found yet under monitor/output/ (run monitor/extract_privileged_from_dataset_libero.py)."
      : "Pick a task, then an episode, to view its ground-truth reconstruction.";
    tdEmptyState.classList.remove("hidden");
    tdEpisodeView.classList.add("hidden");
  }
}

// Sim toggle: re-fetch everything scoped to the newly-selected simulator.
// Method list is sim-specific (RoboCasa's vN_.../ vs LIBERO's vN_..._libero_.../
// under the same monitor/output/ dir), so it must reload too, not just the
// task list.
el("#td-sim-select").addEventListener("change", async (e) => {
  tdSim = e.target.value;
  tdState.task = null;
  tdState.episode = null;
  tdState.monitorMethod = null;
  tdMethodsLoaded = false;
  syncUrl();
  await initTrainingData();
});

// LTL property list: scopes both the left-column per-episode violation
// badges (server-side, via /api/td_episodes?property=...) and the main
// detail panel's violations/satisfied lists (client-side filter in
// loadTrainingMonitor) to a single named property instead of the
// whole-episode aggregate / all 19 properties. "All properties" (null)
// restores the unfiltered view in both places. See server.py's
// list_training_episodes' property_filter param / _property_status_for.
async function initTrainingLtlPropertyList() {
  try {
    const data = await fetchJSON(`/api/training_ltl_properties?sim=${encodeURIComponent(tdSim)}`);
    tdPropertiesList = data.properties;
    tdInactiveProperties = new Set(data.inactive || []);
  } catch (e) {
    tdPropertiesList = [];  // non-fatal -- "All properties" still works, just no per-property entries
    tdInactiveProperties = new Set();
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
  if (!tdState.monitorMethod) {
    tdViolationCounts = { by_task: {}, by_property: {} };
    return;
  }
  try {
    tdViolationCounts = await fetchJSON(
      `/api/training_violation_counts?sim=${encodeURIComponent(tdSim)}&method=${encodeURIComponent(tdState.monitorMethod)}${annotatorQS()}`
    );
  } catch (e) {
    tdViolationCounts = { by_task: {}, by_property: {} };
  }
}

// Generic expandable tree-list row: `label` is the clickable selector text,
// `total` the badge count shown next to it, `children` an array of
// {label, count, onSelect} in a collapsible nested list -- clickable
// (jumps straight to that child's `onSelect`, e.g. a violating episode,
// 2026-09-08) when a child supplies one, plain read-only text otherwise
// (unchanged from before), `isActive` highlights the row as the current
// selection, `onSelect` (the row's own, not a child's) fires on a
// label/count click (not on the caret, which only toggles the nested
// breakdown). `annotated` (optional, 2026-09-08): if given, renders as
// "total (N✎)" alongside the raw count, without affecting the "nonzero"
// styling check below (which stays keyed off the numeric `total`, not the
// formatted text) -- a child's own `c.annotated` works the same way.
// `inactive` (optional, 2026-09-09): tags a property whose trigger
// mechanism never fires anywhere in the current sim's corpus (LIBERO only
// -- see tdInactiveProperties/server.py's LIBERO_INACTIVE_PROPERTIES) --
// dimmed row + "(N/A for this sim)" suffix, so a 0-violation count doesn't
// read as "exercised and always passed" when it actually never triggered
// at all.
function buildTreeRow(label, total, isActive, onSelect, children, annotated, inactive) {
  const wrap = document.createElement("div");
  wrap.className = "tree-item";

  const row = document.createElement("div");
  row.className = "tree-row" + (isActive ? " active" : "") + (inactive ? " tree-row-inactive" : "");

  const hasChildren = children && children.length > 0;
  const caret = document.createElement("span");
  caret.className = "tree-caret";
  caret.textContent = hasChildren ? "▸" : "";

  const labelSpan = document.createElement("span");
  labelSpan.className = "tree-label";
  const displayLabel = inactive ? `${label} (N/A for this sim)` : label;
  labelSpan.textContent = displayLabel;
  labelSpan.title = inactive
    ? `${label} -- never triggers anywhere in this corpus (no matching fixture/object), not "exercised and always passed"`
    : label;

  const countSpan = document.createElement("span");
  countSpan.className = "tree-count" + (total ? " nonzero" : "");
  countSpan.textContent = total == null ? "" : (annotated != null ? `${total} (${annotated}✎)` : `${total}`);

  row.append(caret, labelSpan, countSpan);
  wrap.appendChild(row);

  if (hasChildren) {
    const childList = document.createElement("div");
    childList.className = "tree-children hidden";
    for (const c of children) {
      const cRow = document.createElement("div");
      cRow.className = "tree-child-row" + (c.onSelect ? " clickable" : "");
      const cCountText = c.annotated != null ? `${c.count} (${c.annotated}✎)` : `${c.count}`;
      cRow.innerHTML = `<span>${c.label}</span><span>${cCountText}</span>`;
      if (c.onSelect) {
        cRow.title = "jump to an example violating episode";
        cRow.addEventListener("click", (ev) => {
          ev.stopPropagation();
          c.onSelect();
        });
      }
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

// Fixed display order for LIBERO's 4 in-scope benchmark suites (not
// alphabetical -- matches the order extract_privileged_from_dataset_libero.py
// processes them in). "Other" catches a task whose source hdf5 wasn't found
// (suite is None -- e.g. LIBERO_DATASET_ROOT not populated on this machine),
// so a task is never silently dropped from the tree just because its suite
// couldn't be determined.
const LIBERO_SUITE_ORDER = ["libero_10", "libero_goal", "libero_object", "libero_spatial"];
const LIBERO_SUITE_LABELS = {
  libero_10: "LIBERO-10", libero_goal: "LIBERO-Goal",
  libero_object: "LIBERO-Object", libero_spatial: "LIBERO-Spatial",
};

function taskTreeRow(t) {
  const counts = tdViolationCounts.by_task[t.task] || { total: 0, by_property: {} };
  const children = Object.entries(counts.by_property)
    .sort((a, b) => b[1] - a[1])
    .map(([prop, count]) => ({ label: prop, count }));
  // n_reconstructed (RoboCasa: reconstructed-video count) vs. n_extracted/
  // n_monitored (LIBERO: no video-reconstruction step -- see
  // ensure_libero_original_video's docstring -- so "extracted"/"monitored"
  // episode counts are the meaningful progress numbers instead).
  const label = tdSim === "libero"
    ? `${t.task} (${t.n_monitored}/${t.n_extracted} monitored)`
    : `${t.task} (${t.n_reconstructed} reconstructed)`;
  return buildTreeRow(label, counts.total, t.task === tdState.task, () => loadTrainingEpisodes(t.task), children);
}

function renderTaskTree() {
  tdTaskTree.innerHTML = "";
  if (!tdTasksList.length) {
    tdTaskTree.innerHTML = "<div class='muted'>no tasks found</div>";
    return;
  }
  if (tdSim !== "libero") {
    for (const t of tdTasksList) {
      tdTaskTree.appendChild(taskTreeRow(t));
    }
    return;
  }
  // LIBERO: grouped by benchmark suite with a divider header per group,
  // instead of one flat 40-task list.
  const bySuite = {};
  for (const t of tdTasksList) {
    const key = t.suite || "other";
    (bySuite[key] = bySuite[key] || []).push(t);
  }
  const suiteKeys = [...LIBERO_SUITE_ORDER.filter((s) => bySuite[s]), ...Object.keys(bySuite).filter((s) => !LIBERO_SUITE_ORDER.includes(s))];
  for (const suite of suiteKeys) {
    const header = document.createElement("div");
    header.className = "tree-divider";
    header.textContent = `${LIBERO_SUITE_LABELS[suite] || suite} (${bySuite[suite].length})`;
    tdTaskTree.appendChild(header);
    for (const t of bySuite[suite]) {
      tdTaskTree.appendChild(taskTreeRow(t));
    }
  }
}

function renderPropertyTree() {
  tdPropertyTree.innerHTML = "";
  const allTotal = Object.values(tdViolationCounts.by_property).reduce((s, p) => s + p.total, 0);
  tdPropertyTree.appendChild(
    buildTreeRow("All properties", allTotal, tdState.property == null, () => selectTrainingProperty(null), null)
  );
  for (const prop of tdPropertiesList) {
    const counts = tdViolationCounts.by_property[prop] || { total: 0, annotated: 0, by_task: {} };
    // by_task[task] is {count, annotated, example_episode} (2026-09-08) --
    // example_episode backs the clickable jump-to-a-violating-episode
    // shortcut below, so a violation doesn't have to be found by hand, task
    // by task, episode by episode. "annotated" (also 2026-09-08) is how
    // many of those violations already have a real human annotation --
    // shown alongside the raw count, not as a separate row, so the two
    // numbers read together at a glance ("N violations, M annotated so
    // far") instead of requiring a second lookup.
    const children = Object.entries(counts.by_task)
      .sort((a, b) => b[1].count - a[1].count)
      .map(([task, info]) => ({
        label: task,
        count: info.count,
        annotated: info.annotated,
        onSelect: () => jumpToViolatingEpisode(prop, task, info.example_episode),
      }));
    tdPropertyTree.appendChild(
      buildTreeRow(
        prop,
        counts.total,
        prop === tdState.property,
        () => selectTrainingProperty(prop),
        children,
        counts.annotated,
        tdInactiveProperties.has(prop)
      )
    );
  }
}

// Jumps straight to one example violating episode for a (property, task)
// pair, from the property tree's per-task breakdown (2026-09-08) -- selects
// the property filter too, so the episode view opens already scoped to the
// property that made this a "violation" in the first place, instead of
// landing on the unfiltered whole-episode view.
function jumpToViolatingEpisode(property, task, episode) {
  tdState.property = property;
  renderPropertyTree();
  loadTrainingEpisodes(task, episode);
}

async function loadTrainingEpisodes(task, targetEpisode) {
  // captured before tdState.task/episode get overwritten below -- used to
  // re-select the same episode after a property-filter change reloads this
  // same task's list (see the bottom of this function), unless a specific
  // `targetEpisode` was requested instead (jumpToViolatingEpisode above).
  const previousTask = tdState.task;
  const previousEpisode = targetEpisode != null ? targetEpisode : tdState.episode;
  tdState.task = task;
  renderTaskTree();  // update active highlighting in the sidebar tree
  tdEpisodeList.innerHTML = "<div class='loading'>loading episodes…</div>";
  await ensureTrainingMonitorMethods();  // so tdMethodLabel() has short labels ready for the badges below
  const propertyParam = tdState.property ? `&property=${encodeURIComponent(tdState.property)}` : "";
  // method= (2026-09-09): LIBERO's api_libero_training_episodes reads this
  // to resolve which method dir to scope entry.methods[...]/entry.
  // annotated[...] to -- without it, the server silently falls back to
  // LIBERO's own default method regardless of the dropdown's actual
  // selection (harmless while only one LIBERO method exists, but wrong in
  // general). RoboCasa's api_training_episodes has no such param (it
  // already reports every known method's counts unconditionally), so this
  // is a no-op query param there.
  const methodParam = tdState.monitorMethod ? `&method=${encodeURIComponent(tdState.monitorMethod)}` : "";
  const data = await fetchJSON(`/api/td_episodes?task=${encodeURIComponent(task)}${propertyParam}${methodParam}${tdSimQS()}${annotatorQS()}`);
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
    // Only the currently-selected/latest postprocess method's badge (not one
    // per version -- with several vN_ dirs now kept around for history, a
    // badge-per-method row got noisy fast; the left column should read as
    // "is the latest design good," not a version comparison table).
    const methodEntries = Object.entries(ep.methods || {}).filter(
      ([key]) => key === tdState.monitorMethod
    );
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
    // Per-episode "have I (a human) annotated this yet" indicator
    // (2026-09-08) -- scoped to the currently-selected method, same as
    // ep.annotated's own per-method shape (server.py's
    // list_training_episodes/has_human_annotation). Deliberately excludes
    // Claude-authored content (ai_draft/ai_draft_verdict, the structured
    // entry["claude"] block) -- only counts verdict/note/entry["human"]/
    // missed_notes/overall_verdict, all only ever set by the reviewer's own
    // UI actions. A button, not just a badge, so it can jump straight into
    // that episode without requiring a second click on the row first.
    const isAnnotated = !!(ep.annotated || {})[tdState.monitorMethod];
    const annotatedBtn = document.createElement("button");
    annotatedBtn.type = "button";
    annotatedBtn.className = "mini-badge annotate-btn" + (isAnnotated ? " annotated" : "");
    annotatedBtn.title = isAnnotated ? "human-annotated -- click to open" : "not yet human-annotated -- click to open";
    annotatedBtn.textContent = isAnnotated ? "✎ annotated" : "✎ annotate";
    annotatedBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      selectTrainingEpisode(task, ep, row);
    });

    // "Who's already annotated this episode" (2026-09-10) -- server.py's
    // ep.annotated_by[method] is the full list of registered annotator
    // names with real content here, regardless of the currently-selected
    // annotator filter. Split into "you" (folded into the annotate button's
    // own state above) vs. everyone else, so a reviewer can tell at a
    // glance whether someone already covered this episode before deciding
    // whether it's worth their own time too.
    const annotatedByAll = (ep.annotated_by || {})[tdState.monitorMethod] || [];
    const others = annotatedByAll.filter((a) => a !== currentAnnotator);
    const othersBadge = others.length
      ? `<span class="mini-badge others-annotated" title="${others.join(", ")} also annotated this episode">by: ${others.join(", ")}</span>`
      : "";

    row.innerHTML = `<span class="ep-num">#${ep.episode}</span>
      ${successBadge}
      ${violBadges}
      <span class="mini-badge">${orUnknown(ep.n_frames)} frames</span>
      ${othersBadge}`;
    row.appendChild(annotatedBtn);
    row.addEventListener("click", () => selectTrainingEpisode(task, ep, row));
    tdEpisodeList.appendChild(row);
  }
  // Re-select whichever episode was already open if this reload is for the
  // *same* task (e.g. the property filter just changed) and that episode
  // still exists in the list; otherwise fall back to the first episode
  // (task actually changed, or first load). An explicit `targetEpisode`
  // (jumpToViolatingEpisode) always wins, even across a task switch --
  // unlike the property-filter-change case, that's the whole point of the
  // jump, not an incidental same-task preservation.
  const rows = tdEpisodeList.querySelectorAll(".ep-row");
  let keepIdx = 0;
  if (targetEpisode != null || (previousTask === task && previousEpisode != null)) {
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

  const tdVideoRow = el("#td-video-row");
  const tdVideoUnavailable = el("#td-video-unavailable");
  const tdSyncRow = el("#td-sync-row");
  const tdReconCol = el("#td-recon-video-col");
  const tdOriginalLabel = el("#td-original-video-label");
  if (tdSim === "libero") {
    // No re-rendered "reconstruction" for LIBERO yet (see server.py's
    // ensure_libero_original_video docstring) -- hide just the reconstructed
    // <video>'s column and the original/reconstructed sync row (nothing to
    // sync against), NOT the whole video row (that would also hide the
    // Original column). The ORIGINAL (ground-truth demonstration) video IS
    // available, straight from the LIBERO hdf5's own recorded camera
    // frames -- show it whenever ep.original_video_url is present.
    tdVideo.removeAttribute("src");
    tdReconCol.classList.add("hidden");
    tdSyncRow.classList.add("hidden");
    // Reconstructed column is hidden, so the Original column would
    // otherwise stretch to fill the whole row (flex:1 with no sibling to
    // share it with) -- pin it to the same ~half-row width RoboCasa's
    // side-by-side pair uses instead.
    el("#td-original-video-col").classList.add("video-col-half");
    tdOriginalLabel.textContent = "Original (from the LIBERO demo hdf5 — agentview | eye_in_hand)";
    if (ep.original_video_url) {
      tdVideoRow.classList.remove("hidden");
      tdVideoUnavailable.classList.add("hidden");
      tdOriginalVideo.src = ep.original_video_url;
      tdOriginalVideo.load();
      wireFrameReadoutFor(tdOriginalVideo, "td-original-frame-readout", 20, 1, ep.n_frames, ep.n_frames);
    } else {
      tdVideoRow.classList.add("hidden");
      tdVideoUnavailable.classList.remove("hidden");
      tdOriginalVideo.removeAttribute("src");
    }
    loadTrainingMonitor(task, ep.episode, tdState.monitorMethod);
    return;
  }
  tdVideoRow.classList.remove("hidden");
  tdVideoUnavailable.classList.add("hidden");
  tdReconCol.classList.remove("hidden");
  el("#td-original-video-col").classList.remove("video-col-half");
  tdOriginalLabel.textContent = "Original (from the training dataset — robot0_agentview_left | robot0_agentview_right | robot0_eye_in_hand)";

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

// Attached exactly once (module load) rather than inside
// ensureTrainingMonitorMethods() itself -- that function re-runs on every
// sim switch (tdMethodsLoaded gets reset), and re-attaching a listener each
// time on the same persistent <select> would stack up N duplicate handlers
// after N switches, each independently re-fetching/re-rendering on the next
// method change. Reads tdSim/tdState fresh at fire time regardless, so a
// single, permanently-attached listener is both correct and sufficient.
el("#td-method-select").addEventListener("change", async () => {
  const select = el("#td-method-select");
  tdState.monitorMethod = select.value;
  // Same generation-guard pattern as initTrainingData() (see its comment) --
  // a method change and a sim switch can race the same way a fast sim
  // double-switch can.
  const myGeneration = ++tdLoadGeneration;
  await refreshViolationCounts();
  if (myGeneration !== tdLoadGeneration) return;
  renderTaskTree();
  renderPropertyTree();
  if (tdState.task && tdState.episode != null) {
    loadTrainingMonitor(tdState.task, tdState.episode, tdState.monitorMethod);
  }
});

async function ensureTrainingMonitorMethods() {
  if (tdMethodsLoaded) return;
  tdMethodsLoaded = true;
  const select = el("#td-method-select");
  try {
    const data = await fetchJSON(`/api/training_monitor_methods?sim=${encodeURIComponent(tdSim)}`);
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
      `/api/training_monitor?task=${encodeURIComponent(task)}&episode=${episode}&method=${encodeURIComponent(method)}${tdSimQS()}${annotatorQS()}`
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
  for (const v of detail.violations) vlist.appendChild(renderViolation(v, detail.annotations, detail.other_annotations));

  const slist = el("#td-satisfied-list");
  slist.innerHTML = "";
  for (const s of detail.satisfied) slist.appendChild(renderSatisfied(s, detail.annotations, detail.other_annotations));

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
    `/api/episode?task=${encodeURIComponent(task)}&episode=${episode}${annotatorQS()}`
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

// Foldable "other annotators' takes" reference (2026-09-10) -- other
// registered annotators' verdict/note on this SAME (group, index) instance,
// read-only (you can only ever edit your own selected identity's
// annotation, never someone else's -- see currentAnnotator/saveAnnotation).
// `otherAnn` is detail.other_annotations: {annotatorName: full annotation
// dict}, already pre-filtered server-side to annotators with *some* real
// content for this episode -- but that doesn't mean they annotated *this
// specific* instance, so still checked per-entry here. Collapsed by
// default (a <details> element, not always-visible) so it's available as
// a reference without cluttering the common case (no one else has looked
// at this episode yet).
function otherAnnotatorsBlock(group, index, otherAnn) {
  if (!otherAnn) return null;
  const rows = [];
  for (const [name, data] of Object.entries(otherAnn)) {
    const entry = (data[group] || {})[String(index)];
    if (!entry || (!entry.verdict && !(entry.note || "").trim())) continue;
    rows.push({ name, entry });
  }
  if (!rows.length) return null;
  const details = document.createElement("details");
  details.className = "other-annotators";
  const summary = document.createElement("summary");
  summary.textContent = `other annotators' takes (${rows.length})`;
  details.appendChild(summary);
  for (const { name, entry } of rows) {
    const row = document.createElement("div");
    row.className = "other-annotator-row";
    const label = entry.verdict ? (DRAFT_VERDICT_LABEL[entry.verdict] || entry.verdict) : "(no verdict)";
    row.innerHTML = `<strong>${name}</strong>: <span class="other-annotator-verdict">${label}</span>`;
    if ((entry.note || "").trim()) {
      const note = document.createElement("div");
      note.className = "other-annotator-note";
      note.textContent = entry.note;
      row.appendChild(note);
    }
    details.appendChild(row);
  }
  return details;
}

// Same idea as otherAnnotatorsBlock, but for the episode-level missed_notes/
// overall_verdict fields instead of a single violation/satisfied instance.
function otherAnnotatorsEpisodeBlock(otherAnn) {
  if (!otherAnn) return null;
  const rows = [];
  for (const [name, data] of Object.entries(otherAnn)) {
    const missed = (data.missed_notes || "").trim();
    const overall = data.overall_verdict;
    if (!missed && !overall) continue;
    rows.push({ name, missed, overall });
  }
  if (!rows.length) return null;
  const details = document.createElement("details");
  details.className = "other-annotators";
  const summary = document.createElement("summary");
  summary.textContent = `other annotators' episode-level notes (${rows.length})`;
  details.appendChild(summary);
  for (const { name, missed, overall } of rows) {
    const row = document.createElement("div");
    row.className = "other-annotator-row";
    row.innerHTML = `<strong>${name}</strong>${overall ? `: <span class="other-annotator-verdict">${overall}</span>` : ""}`;
    if (missed) {
      const note = document.createElement("div");
      note.className = "other-annotator-note";
      note.textContent = missed;
      row.appendChild(note);
    }
    details.appendChild(row);
  }
  return details;
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
      annotator: currentAnnotator,
      ...(group ? { group, index } : {}),
      ...patch,
    }),
  });
  // Refresh the training-data sidebar's LTL-property/task trees (2026-09-08)
  // -- their "N annotated" counts otherwise go stale the instant a verdict
  // is saved, since tdViolationCounts is only ever fetched once (on load /
  // method change), never re-pulled after an annotation write. Only when
  // we're actually in the training tab (annotationContext.task is
  // "training__<task>", see setAnnotationContext's own comment) --
  // saveAnnotation is shared with the eval tab, which has no such tree.
  // Cheap: refreshViolationCounts() is one aggregate fetch, not a full
  // episode-list/detail-panel reload.
  if (annotationContext.task && annotationContext.task.startsWith("training__")) {
    await refreshViolationCounts();
    renderPropertyTree();
    renderTaskTree();
    // The currently-open episode's own sidebar "annotate" button, updated
    // in place (no network round-trip) -- any verdict/note/human-block
    // patch means this episode now counts as annotated for the current
    // method.
    const activeRow = tdEpisodeList.querySelector(".ep-row.active .annotate-btn");
    if (activeRow) {
      activeRow.classList.add("annotated");
      activeRow.title = "human-annotated -- click to open";
      activeRow.textContent = "✎ annotated";
    }
  }
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

function repeatedViolationsBlock(rep, window) {
  // repeated_violation_episodes -- RepeatedViolationMonitor's own, separate
  // recovery_ltl-driven bookkeeping, distinct from the predicate_breakdown
  // occurrences table above it (which re-simulates the *main* formula's own
  // until/escape directly, independent of recovery_ltl entirely -- see
  // viewer/annotations/ltl_debugging_guides/README.md's "two separate
  // mechanisms" note, and that property's own guide for what recovery_ltl
  // does/doesn't mean for it specifically).
  if (!rep || !rep.recovery_ltl) return null;
  const details = document.createElement("details");
  details.className = "repeated-violations";
  const summary = document.createElement("summary");
  const count = rep.repeated_violation_count || 0;
  summary.textContent = `repeated_violation_episodes (recovery_ltl bookkeeping) — ${count} episode(s)`;
  details.appendChild(summary);

  const ltl = document.createElement("div");
  ltl.className = "ltl-line";
  ltl.innerHTML = `<span class="ltl-tag">recovery_ltl</span><code>${rep.recovery_ltl}</code>`;
  details.appendChild(ltl);

  if (window && rep.episodes && rep.episodes.length) {
    const { start_frame, end_frame } = window;
    const span = Math.max(1, end_frame - start_frame + 1);
    // Reuses the same run-segment machinery every other predicate row uses
    // (fillTimeline/predicateRow) -- "true" (green) = in an episode that
    // recovered, "false" (red) = in an episode still unrecovered
    // (including the still-open one at episode end), "unknown" (gray) =
    // not currently tracking any episode at all. Gaps between/around
    // episodes are filled with unknown runs so the bar always spans the
    // full window with no missing segments (fillTimeline assumes a
    // contiguous run list).
    const sorted = [...rep.episodes].sort((a, b) => a.start_frame - b.start_frame);
    const runs = [];
    let cursor = start_frame;
    for (const ep of sorted) {
      if (ep.start_frame > cursor) {
        runs.push({ start_frame: cursor, end_frame: ep.start_frame - 1, value: null,
          start: { frame: cursor }, end: { frame: ep.start_frame - 1 } });
      }
      const epEnd = ep.end_frame != null ? ep.end_frame : end_frame;
      runs.push({
        start_frame: ep.start_frame, end_frame: epEnd, value: ep.recovered,
        start: ep.start_marker || { frame: ep.start_frame },
        end: ep.end_marker || { frame: epEnd },
      });
      cursor = epEnd + 1;
    }
    if (cursor <= end_frame) {
      runs.push({ start_frame: cursor, end_frame, value: null,
        start: { frame: cursor }, end: { frame: end_frame } });
    }
    const { row, timeline } = predicateRow({ label: "recovery status", key: "recovery_ltl",
      description: "green = recovered, red = still unrecovered, gray = not tracking an episode" }, false);
    fillTimeline(timeline, { label: "recovery status", runs }, span);
    details.appendChild(row);
  }

  if (rep.in_violation_at_end) {
    const warn = document.createElement("div");
    warn.className = "card-hint warn";
    warn.textContent = "Still in violation at episode end — the last episode never recovered.";
    details.appendChild(warn);
  }

  if (rep.episodes && rep.episodes.length) {
    const list = document.createElement("div");
    list.className = "repeated-episode-list";
    for (const ep of rep.episodes) {
      const row = document.createElement("div");
      row.className = "repeated-episode-row";
      const startBtn = document.createElement("button");
      startBtn.className = "chip";
      startBtn.textContent = `start f${ep.start_frame}`;
      startBtn.addEventListener("click", () => seekTo(ep.start_marker));
      row.appendChild(startBtn);
      if (ep.end_frame != null) {
        const endBtn = document.createElement("button");
        endBtn.className = "chip";
        endBtn.textContent = `end f${ep.end_frame}`;
        endBtn.addEventListener("click", () => seekTo(ep.end_marker));
        row.appendChild(endBtn);
      }
      const dur = document.createElement("span");
      dur.className = "repeated-episode-duration";
      dur.textContent = ep.duration_frames != null ? `${ep.duration_frames} frames` : "";
      row.appendChild(dur);
      const badge = document.createElement("span");
      badge.className = "badge " + (ep.recovered ? "badge-ok" : "badge-fail");
      badge.textContent = ep.recovered ? "recovered" : "unrecovered";
      row.appendChild(badge);
      list.appendChild(row);
    }
    details.appendChild(list);
  }
  return details;
}

// Renders the claude/human x gt_annotation/monitor_problem structured
// annotation schema (.claude/skills/ltl-ground-truth-annotation/SKILL.md) --
// separate from the existing free-text aiDraftBlock/verdictControls/noteBox
// (the reviewer's own running verdict), this shows the *structured* record:
// what actually happened (gt_annotation's per-occurrence trigger/resolve
// frames, reusing the same occurrence shape predicateBreakdown already
// renders) and whether the monitor's own reasoning was sound
// (monitor_problem), kept as two separate questions per that skill's design.
// Currently only "claude" is ever populated (via
// SafeManip/monitor/populate_claude_annotations.py); "human" renders
// identically whenever/if a human reviewer's own gt_annotation gets added
// through the same save_annotations "source" patch mechanism.
function groundTruthAnnotationSection(current, source, label) {
  const block = current && current[source];
  if (!block || (!block.gt_annotation && !block.monitor_problem)) return null;
  const details = document.createElement("details");
  details.className = "gt-annotation gt-annotation-" + source;
  const summary = document.createElement("summary");
  const problem = block.monitor_problem && block.monitor_problem.has_problem;
  summary.textContent = `${label} annotation` + (problem ? " — ⚠ monitor problem flagged" : "");
  details.appendChild(summary);

  const ann = block.gt_annotation;
  if (ann) {
    if (ann.source_note) {
      const note = document.createElement("div");
      note.className = "card-hint";
      note.textContent = ann.source_note;
      details.appendChild(note);
    }
    if (ann.confidence) {
      const conf = document.createElement("div");
      conf.className = "card-hint";
      conf.textContent = `confidence: ${ann.confidence}`;
      details.appendChild(conf);
    }
    if (ann.occurrences && ann.occurrences.length) {
      const list = document.createElement("div");
      list.className = "repeated-episode-list";
      for (const occ of ann.occurrences) {
        const row = document.createElement("div");
        row.className = "repeated-episode-row";
        if (occ.object) {
          const objChip = document.createElement("span");
          objChip.className = "chip";
          objChip.textContent = occ.object;
          row.appendChild(objChip);
        }
        if (occ.activation && occ.activation.frame != null) {
          const startBtn = document.createElement("button");
          startBtn.className = "chip";
          startBtn.textContent = `trigger f${occ.activation.frame}`;
          startBtn.addEventListener("click", () => seekTo(occ.activation.marker));
          row.appendChild(startBtn);
        }
        if (occ.end && occ.end.frame != null) {
          const endBtn = document.createElement("button");
          endBtn.className = "chip";
          endBtn.textContent = `end f${occ.end.frame}`;
          endBtn.addEventListener("click", () => seekTo(occ.end.marker));
          row.appendChild(endBtn);
        }
        const badge = document.createElement("span");
        const resolved = occ.end && occ.end.resolved;
        badge.className = "badge " + (resolved ? "badge-ok" : "badge-fail");
        badge.textContent = resolved ? "resolved" : "unresolved";
        row.appendChild(badge);
        list.appendChild(row);
      }
      details.appendChild(list);
    }
  }

  if (block.monitor_problem) {
    const warn = document.createElement("div");
    warn.className = "card-hint" + (problem ? " warn" : "");
    warn.textContent = problem
      ? `monitor_problem: ${block.monitor_problem.description || "(no description)"}`
      : "monitor_problem: none flagged";
    details.appendChild(warn);
  }
  return details;
}

function groundTruthAnnotationBlock(current) {
  const wrap = document.createElement("div");
  wrap.className = "gt-annotation-wrap";
  let any = false;
  for (const [source, label] of [["claude", "Claude"], ["human", "Human"]]) {
    const section = groundTruthAnnotationSection(current, source, label);
    if (section) {
      wrap.appendChild(section);
      any = true;
    }
  }
  return any ? wrap : null;
}

function ltlLine(ltl) {
  if (!ltl) return null;
  const line = document.createElement("div");
  line.className = "ltl-line";
  line.innerHTML = `<span class="ltl-tag">LTL</span><code>${ltl}</code>`;
  return line;
}

function renderViolation(v, ann, otherAnn) {
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

  const rep = repeatedViolationsBlock(v.repeated, v.predicate_breakdown && v.predicate_breakdown.window);
  if (rep) card.appendChild(rep);

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

  const gt = groundTruthAnnotationBlock(current);
  if (gt) card.appendChild(gt);

  card.appendChild(verdictControls("violations", v.index, current));
  card.appendChild(noteBox("violations", v.index, current));
  const others = otherAnnotatorsBlock("violations", v.index, otherAnn);
  if (others) card.appendChild(others);
  return card;
}

function renderSatisfied(s, ann, otherAnn) {
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

  const rep = repeatedViolationsBlock(s.repeated, s.predicate_breakdown && s.predicate_breakdown.window);
  if (rep) card.appendChild(rep);

  const draft = aiDraftBlock(current);
  if (draft) card.appendChild(draft);

  const gt = groundTruthAnnotationBlock(current);
  if (gt) card.appendChild(gt);

  card.appendChild(verdictControls("satisfied", s.index, current));
  card.appendChild(noteBox("satisfied", s.index, current));
  const others = otherAnnotatorsBlock("satisfied", s.index, otherAnn);
  if (others) card.appendChild(others);
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
  for (const v of detail.violations) vlist.appendChild(renderViolation(v, detail.annotations, detail.other_annotations));

  const slist = el("#satisfied-list");
  slist.innerHTML = "";
  for (const s of detail.satisfied) slist.appendChild(renderSatisfied(s, detail.annotations, detail.other_annotations));

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

  const others = otherAnnotatorsEpisodeBlock(detail.other_annotations);
  if (others) panel.appendChild(others);
}

init();
