"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, offline: false, busy: false, slow: false, notice: null, runs: 0, stopEdited: false, stopTask: "language_generation" };
const encoder = new TextEncoder();
const bytes = (text) => encoder.encode(text).length;
const plural = (count, noun) => `${Number(count).toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;
const labels = Object.fromEntries([...$("task-type").options].map((option) => [option.value, option.text]));
// Same limits as grounding.py.
const LIMITS = { source: 12000, question: 2000 };
const DEFAULTS = { "max-tokens": "64", temperature: "0.3", "task-type": "language_generation", "top-k": "40", "top-p": "0.9", "repetition-penalty": "1" };
// Temperature 0 decodes greedily, so Steady repeats itself; Balanced is the default sampling.
const PRESETS = {
  steady: { temperature: 0, "top-p": 0.9, "top-k": 40, "repetition-penalty": 1 },
  balanced: { temperature: 0.3, "top-p": 0.9, "top-k": 40, "repetition-penalty": 1 },
  varied: { temperature: 0.9, "top-p": 0.95, "top-k": 0, "repetition-penalty": 1.1 },
};
const INTRO = {
  model: ["What would you like to know?", "Ask a short question. Everything runs on your computer, and answers can be wrong."],
  sources: ["Search your own text", "Paste some text and ask about it. Studio quotes the passages that use the words in your question."],
};
const EXAMPLE = {
  source: "The Riverside Library is open from 9 am to 8 pm on weekdays and from 10 am to 4 pm on Saturdays. It is closed on Sundays and public holidays.\n\nMembers can borrow up to 12 books at a time for three weeks. Laptops can be borrowed for one day and must be returned to the front desk.\n\nPrinting costs 10 cents per page.",
  question: "How many books can members borrow?",
};
const THEMES = ["system", "light", "dark"];
// Touch keyboards have no Shift+Enter, so there Enter adds a line break and the arrow button sends.
const touch = matchMedia("(pointer: coarse)").matches;
const motion = () => (matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

// Storage can be switched off, for example in private windows, so reads and writes are allowed to fail.
function readStore(area, key) {
  try { return JSON.parse(window[area].getItem(key)); } catch { return null; }
}

function writeStore(area, key, value) {
  try { window[area].setItem(key, JSON.stringify(value)); } catch { /* Studio works without storage. */ }
}

function savedTheme() {
  const theme = readStore("localStorage", "studio-theme");
  return THEMES.includes(theme) ? theme : "system";
}

function applyTheme(theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $(`theme-${theme}`).checked = true;
  // The browser's own bar follows the page, including a theme picked here instead of the system one.
  const background = getComputedStyle(document.body).backgroundColor;
  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    meta.dataset.system ??= meta.content;
    meta.content = theme === "system" ? meta.dataset.system : background;
  }
}

// A newline stop keeps short answers to one line; multi-line tasks such as code start without one.
const defaultStops = (task) => (task === "language_generation" ? "\\n" : "");

function stopSequences() {
  // Commas and line breaks separate entries, so a typed \n or \t stands for that character.
  return $("stop-sequences").value.split(/[,\n]/).map((item) => item.trim().replace(/\\n/g, "\n").replace(/\\t/g, "\t")).filter(Boolean);
}

function settings() {
  // Number("") is 0 (unrestricted sampling), so a cleared field must stay invalid instead.
  const topK = $("top-k").value.trim();
  const stops = stopSequences();
  // The server rejects an empty list, so no entries means the field is omitted from the request.
  return { task_type: $("task-type").value, temperature: Number($("temperature").value),
    max_new_tokens: Number($("max-tokens").value), top_k: topK === "" ? NaN : Number(topK),
    top_p: Number($("top-p").value), repetition_penalty: Number($("repetition-penalty").value),
    stop_sequences: stops.length ? stops : undefined };
}

function promptTokens() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return 0;
  return 1 + encoder.encode(`<task_type>${settings().task_type}</task_type>\n<instruction>\n${prompt}\n</instruction>\n<answer>\n`).length;
}

function sourceMode() {
  return $("mode-sources").checked;
}

// The project's own checkpoints read task_type; hosted models such as SmolLM2 with LoRA ignore it.
function customModel() {
  // Unknown until the model loads, so the task field stays hidden rather than flashing up for LoRA models.
  const architecture = state.status?.model?.architecture;
  return Boolean(architecture) && !String(architecture).startsWith("llama");
}

function phase() {
  return state.offline ? "offline" : state.status?.state || "connecting";
}

function syncComposer() {
  if (!state.stopEdited && $("task-type").value !== state.stopTask) $("stop-sequences").value = defaultStops($("task-type").value);
  state.stopTask = $("task-type").value;
  const grounded = sourceMode();
  const config = settings();
  const count = promptTokens();
  const current = phase();
  const context = state.status?.model?.context_window;
  const sourceBytes = bytes($("source-text").value);
  const questionBytes = bytes($("prompt").value.trim());
  const sourceOverflow = grounded && (sourceBytes > LIMITS.source || questionBytes > LIMITS.question);
  const overflow = !grounded && Boolean(context) && count + config.max_new_tokens > context;
  const validK = Number.isInteger(config.top_k) && config.top_k >= 0 && config.top_k <= 259;
  const validStops = !config.stop_sequences || (config.stop_sequences.length <= 4 && config.stop_sequences.every((item) => encoder.encode(item).length <= 64));
  const busy = Boolean(state.status?.busy);
  const thread = state.runs > 0;

  document.body.classList.toggle("has-thread", thread);
  document.body.classList.toggle("searching", grounded);
  $("intro").hidden = thread;
  $("thread").hidden = !thread;
  [$("intro-title").textContent, $("intro-text").textContent] = INTRO[grounded ? "sources" : "model"];
  $("source-panel").hidden = !grounded;
  $("prompt").placeholder = grounded ? "Ask about your text" : "Ask a question";
  $("mode-model").disabled = current === "disabled";
  $("task-field").hidden = !customModel();
  $("starters").hidden = grounded || thread || ["error", "disabled", "offline"].includes(current);
  $("source-starters").hidden = !grounded || thread || sourceBytes > 0;
  $("new-session").hidden = !thread;
  $("new-session").disabled = state.busy;

  $("source-count").textContent = `${sourceBytes.toLocaleString()} / ${LIMITS.source.toLocaleString()} bytes`;
  $("source-count").classList.toggle("over", sourceBytes > LIMITS.source);
  // The count stays hidden until a limit is close, so short questions get a quiet composer.
  const [used, limit, unit] = grounded ? [questionBytes, LIMITS.question, "bytes"] : [count, context ? Math.max(context - config.max_new_tokens, 0) : 0, "tokens"];
  const near = limit > 0 && used > limit * 0.8;
  $("token-count").textContent = near ? `${used.toLocaleString()} / ${limit.toLocaleString()} ${unit}` : "";
  $("token-count").classList.toggle("over", near && used > limit);

  $("max-tokens-value").value = `${config.max_new_tokens} tokens`;
  $("temperature-value").value = config.temperature.toFixed(1);
  $("top-p-value").value = config.top_p.toFixed(2);
  $("repetition-penalty-value").value = config.repetition_penalty.toFixed(2);
  $("top-k").setAttribute("aria-invalid", String(!validK));
  // Moving any sampling control off a preset leaves no preset selected.
  const preset = validK && Object.keys(PRESETS).find((name) => Object.entries(PRESETS[name]).every(([id, value]) => Math.abs(Number($(id).value) - value) < 1e-9));
  for (const radio of document.querySelectorAll('input[name="preset"]')) radio.checked = radio.value === preset;
  $("stop-sequences").setAttribute("aria-invalid", String(!validStops));

  $("generate").disabled = state.busy || !count || current === "offline" || (grounded ? sourceOverflow || !sourceBytes : current !== "ready" || overflow || !validK || !validStops || busy);
  $("generate").classList.toggle("busy", state.busy);
  $("generate").setAttribute("aria-label", state.busy ? "Working on an answer" : grounded ? "Search" : "Send");

  const problem = grounded ? "" : overflow ? "Shorten your question, or lower Answer length in Settings."
    : !validK ? "Top K must be a whole number from 0 to 259. Change it in Settings, under Advanced."
    : !validStops ? "Stop sequences: use at most 4 entries in Settings, under Advanced, each at most 64 UTF-8 bytes."
    : "";
  let note = "", error = false;
  if (state.notice) ({ text: note, error } = state.notice);
  else if (current === "offline") [note, error] = ["Can't reach Studio. Start it again the way you started it before, and this page will reconnect on its own.", true];
  else if (sourceOverflow) [note, error] = [`Keep your text under ${LIMITS.source.toLocaleString()} bytes and your question under ${LIMITS.question.toLocaleString()}.`, true];
  else if (grounded && count && !sourceBytes) note = "Paste the text you want to search first.";
  else if (!grounded && ["error", "disabled"].includes(current)) [note, error] = ["The model couldn't load. Search my text still works, and the status button at the top shows why.", true];
  else if (problem) [note, error] = [problem, true];
  else if (!grounded && current === "loading") note = "Loading the model. The first start can take a minute.";
  else if (!grounded && busy && !state.busy) note = "The model is answering another request. Try again in a moment.";
  else if (state.busy && state.slow) note = "Still working. Longer answers take more time.";
  else if (thread) note = "Each question is answered on its own, without the earlier ones.";
  // Rewriting identical text would make some screen readers repeat it on every keystroke.
  if ($("feedback").textContent !== note) $("feedback").textContent = note;
  $("feedback").classList.toggle("error", error);
}

function notify(text, error = false) {
  state.notice = { text, error };
  syncComposer();
}

function announce(message) {
  $("announcer").textContent = "";
  // Clearing first makes screen readers announce a message even when it repeats.
  setTimeout(() => { $("announcer").textContent = message; }, 60);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function autosize() {
  const prompt = $("prompt");
  prompt.style.height = "auto";
  prompt.style.overflowY = "hidden";
  // An empty box keeps its one-row height; measuring a wrapped placeholder would stretch it.
  if (!prompt.value) return;
  prompt.style.height = `${Math.min(prompt.scrollHeight, 220)}px`;
  if (prompt.scrollHeight > 220) prompt.style.overflowY = "auto";
}

function reveal(turn) {
  // Long turns scroll to their start; short ones scroll the page to the bottom, above the composer.
  const room = window.innerHeight - document.querySelector(".dock").offsetHeight - 72;
  if (turn.offsetHeight > room) turn.scrollIntoView({ block: "start", behavior: motion() });
  else window.scrollTo({ top: document.documentElement.scrollHeight, behavior: motion() });
}

function addTurn(run) {
  run.turn = element("article", "turn");
  const question = element("p", "question", run.prompt);
  question.tabIndex = -1;
  run.turn.append(question);
  const tag = run.grounded ? "Searched your text" : run.custom && run.options.task_type !== "language_generation" ? labels[run.options.task_type] : "";
  if (tag) run.turn.append(element("p", "question-tag", tag));
  run.answer = element("div", "answer");
  const typing = element("div", "typing");
  typing.append(element("span", "visually-hidden", "Working on an answer"), element("i"), element("i"), element("i"));
  run.answer.append(typing);
  run.turn.append(run.answer);
  $("thread").append(run.turn);
}

function copyButton(text, label) {
  const button = element("button", "copy", "Copy");
  button.type = "button";
  button.setAttribute("aria-label", label);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      button.classList.add("done");
      announce("Copied");
      setTimeout(() => { button.textContent = "Copy"; button.classList.remove("done"); }, 1600);
    } catch {
      notify("Copying isn't available here. Select the text to copy it instead.", true);
    }
  });
  return button;
}

function showResult(run, response) {
  const meta = element("div", "answer-meta");
  if (run.grounded) {
    // Abstentions carry fallback text, so they show a note instead of copyable excerpts.
    const sources = response.abstained ? [] : response.sources || [];
    if (!sources.length) {
      run.answer.replaceChildren(element("p", "answer-note", "No passage in your text uses enough of the words in your question. Try fewer words, or words from your text."));
      announce("No matching passages in your text.");
      return;
    }
    const list = element("ol", "excerpts");
    for (const source of sources) {
      const item = element("li");
      item.append(element("span", "cite", source.id), element("blockquote", "", source.text));
      list.append(item);
    }
    meta.append(element("span", "", `${plural(sources.length, "passage")} quoted from your text`), copyButton(response.text, "Copy passages"));
    run.answer.replaceChildren(list, meta);
    announce(`Found ${plural(sources.length, "passage")}. ${sources.map((source) => source.text).join(" ")}`);
    return;
  }
  // Leading blank lines are noise; indentation on the first line is kept for code.
  const text = String(response.text || "").replace(/^\n+/, "").trimEnd();
  const code = run.custom && ["code_generation", "code_debugging"].includes(run.options.task_type);
  const details = [plural(response.generated_tokens, "token"), `${Number(response.elapsed_seconds).toFixed(1)}s`];
  if (response.finish_reason === "length") details.push("stopped at the length limit");
  meta.append(element("span", "", details.join(" · ")));
  if (text) {
    meta.append(copyButton(text, "Copy answer"));
    run.answer.replaceChildren(element("pre", code ? "answer-text code" : "answer-text", text), meta);
    announce(text);
    return;
  }
  const empty = response.finish_reason === "stop" ? "A stop sequence ended the response before any text. Clear Stop sequences in Settings, under Advanced, and try again."
    : "The model returned no text. Try rephrasing, or raise the temperature in Settings.";
  run.answer.replaceChildren(element("p", "answer-note", empty), meta);
  announce(empty);
}

async function post(path, payload) {
  let response;
  try {
    response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  } catch {
    throw new Error("Couldn't reach Studio. Check that it's still running, then try again.");
  }
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(result.error || `Studio answered with an error (${response.status}). Try again.`), { status: response.status });
  return result;
}

$("prompt-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("generate").disabled) return;
  const grounded = sourceMode();
  // The server rejects control characters other than tab and line breaks, which pasted text sometimes carries.
  const run = { prompt: $("prompt").value.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, " ").trim(), options: settings(), grounded, custom: customModel(), source_text: grounded ? $("source-text").value : "" };
  state.busy = true; state.slow = false; state.notice = null; state.runs += 1;
  addTurn(run);
  announce("Working on an answer");
  // Touch screens: the chip or button just used may have disappeared, so hand focus to the new question.
  if (touch) run.turn.firstChild.focus({ preventScroll: true });
  $("prompt").value = "";
  autosize(); syncComposer(); reveal(run.turn);
  const slow = setTimeout(() => { state.slow = true; syncComposer(); }, 8000);
  try {
    const payload = grounded ? { prompt: run.prompt, source_text: run.source_text } : { prompt: run.prompt, ...run.options };
    showResult(run, await post(grounded ? "/api/grounded" : "/api/generate", payload));
  } catch (error) {
    run.status = error.status;
    run.answer.replaceChildren(element("p", "answer-note error", error.message));
    announce(error.message);
    // Put the question back so it can be sent again, unless a new one is already being typed.
    if (!$("prompt").value) { $("prompt").value = run.prompt; autosize(); }
  } finally {
    clearTimeout(slow);
    state.busy = false; state.slow = false;
    // Searches never hold the model, and a 409 means another request still does.
    if (state.status && !grounded && run.status !== 409) state.status.busy = false;
    syncComposer(); reveal(run.turn);
    if (!touch && [document.body, $("generate")].includes(document.activeElement)) $("prompt").focus();
  }
});

function newSession() {
  if (state.busy) return;
  state.runs = 0; state.notice = null;
  $("thread").replaceChildren();
  $("prompt").value = "";
  autosize(); syncComposer();
  window.scrollTo({ top: 0 });
  $(touch ? "intro-title" : "prompt").focus();
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request failed with ${response.status}.`);
    state.status = await response.json();
    state.offline = false;
  } catch {
    state.offline = true;
  }
  // A sources-only server has no model, so searching your text is the only mode that works.
  if (phase() === "disabled" && !sourceMode()) $("mode-sources").checked = true;
  renderStatus(); syncComposer();
  setTimeout(pollStatus, phase() === "loading" ? 1200 : 5000);
}

function renderStatus() {
  const current = phase();
  // A search-only server never loads its model, so its details would describe weights that are not in use.
  const model = current === "disabled" ? {} : state.status?.model || {};
  const names = { ready: model.name || "Model ready", loading: "Loading model", disabled: "Search only", error: "Model unavailable", offline: "Offline", connecting: "Connecting" };
  $("status-dot").className = `dot ${current}`;
  $("model-name").textContent = names[current] || "Model unavailable";
  const size = model.parameters >= 1e9 ? `${(model.parameters / 1e9).toFixed(2)} billion` : model.parameters ? `${(model.parameters / 1e6).toFixed(1)} million` : null;
  const details = {
    Status: { ready: "Ready", loading: "Loading", disabled: "Search only, no model loaded", error: "Couldn't load", offline: "Can't reach the server", connecting: "Connecting" }[current],
    Model: model.name, Checkpoint: model.checkpoint, Parameters: size,
    Context: model.context_window ? `${model.context_window.toLocaleString()} ${customModel() ? "byte tokens" : "tokens"}` : null,
    Device: model.device?.toUpperCase(), Architecture: model.architecture, "Training steps": model.training_steps?.toLocaleString(),
    Error: current === "error" ? state.status?.error : null,
  };
  // Rebuilding on every poll would drop any text the user has selected in the About panel.
  const shown = JSON.stringify(details);
  if (shown === state.details) return;
  state.details = shown;
  $("model-details").replaceChildren(...Object.entries(details).filter(([, value]) => value).map(([key, value]) => {
    const row = element("div");
    row.append(element("dt", "", key), element("dd", "", String(value)));
    return row;
  }));
}

$("prompt").addEventListener("input", () => { state.notice = null; autosize(); });
window.addEventListener("resize", autosize);
for (const id of ["prompt", "source-text", "task-type", "temperature", "max-tokens", "top-k", "top-p", "repetition-penalty", "stop-sequences"]) $(id).addEventListener("input", syncComposer);
$("stop-sequences").addEventListener("input", () => { state.stopEdited = true; });
for (const radio of document.querySelectorAll('input[name="mode"]')) radio.addEventListener("change", () => { state.notice = null; syncComposer(); });
for (const radio of document.querySelectorAll('input[name="theme"]')) radio.addEventListener("change", () => {
  applyTheme(radio.value);
  writeStore("localStorage", "studio-theme", radio.value);
});
// Other Studio tabs pick up a theme change right away.
window.addEventListener("storage", (event) => {
  if (event.key === "studio-theme") applyTheme(savedTheme());
});
for (const radio of document.querySelectorAll('input[name="preset"]')) radio.addEventListener("change", () => {
  for (const [id, value] of Object.entries(PRESETS[radio.value])) $(id).value = String(value);
  syncComposer();
});

$("prompt").addEventListener("keydown", (event) => {
  // Enter sends. Shift+Enter, IME composition (Safari flags it only by keyCode 229) and touch keyboards add a line break.
  if (event.key !== "Enter" || event.shiftKey || event.metaKey || event.ctrlKey || event.isComposing || event.keyCode === 229 || touch) return;
  event.preventDefault();
  $("prompt-form").requestSubmit();
});
document.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]") || event.altKey) return;
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); $("prompt-form").requestSubmit(); return; }
  if (event.metaKey || event.ctrlKey || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (event.key === "/") { event.preventDefault(); $("prompt").focus(); }
});

$("new-session").addEventListener("click", newSession);
document.querySelector(".mark").addEventListener("click", (event) => {
  // Modified clicks keep their browser meaning, such as opening a new tab.
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  newSession();
});
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => {
  $("mode-model").checked = true;
  $("task-type").value = "language_generation";
  $("prompt").value = button.dataset.prompt;
  autosize(); syncComposer();
  if (!$("generate").disabled) $("prompt-form").requestSubmit();
  else $("prompt").focus();
}));
$("source-example").addEventListener("click", () => {
  $("source-text").value = EXAMPLE.source;
  $("prompt").value = EXAMPLE.question;
  autosize(); syncComposer();
  if (!$("generate").disabled) $("prompt-form").requestSubmit();
});
$("reset-settings").addEventListener("click", () => {
  for (const [id, value] of Object.entries(DEFAULTS)) $(id).value = value;
  state.stopEdited = false;
  $("stop-sequences").value = defaultStops($("task-type").value);
  state.stopTask = $("task-type").value;
  syncComposer();
});
for (const name of ["settings", "about"]) {
  const dialog = $(name);
  let pressed = false;
  $(`${name}-open`).addEventListener("click", () => dialog.showModal());
  dialog.querySelectorAll(".close-dialog").forEach((button) => button.addEventListener("click", () => dialog.close()));
  // Close on a backdrop click, but not when a drag that started inside, on a slider say, ends outside.
  dialog.addEventListener("pointerdown", (event) => { pressed = event.target === dialog; });
  dialog.addEventListener("click", (event) => { if (pressed && event.target === dialog) dialog.close(); });
}

applyTheme(savedTheme());
autosize(); renderStatus(); syncComposer(); pollStatus();
if (!touch) $("prompt").focus();
