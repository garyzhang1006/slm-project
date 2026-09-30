"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, offline: false, busy: false, slow: false, notice: null, stopEdited: false, stopTask: "language_generation", started: false, answered: 0 };
// Every question asked in this tab with each answer it got, so a turn can be drawn again from data.
const runs = [];
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
const FIELDS = { temperature: "temperature", "top-p": "top_p", "top-k": "top_k", "repetition-penalty": "repetition_penalty" };
const INTRO = {
  model: ["What would you like to know?", "Ask a short question. Everything runs on your computer, and answers can be wrong."],
  sources: ["Search your own text", "Paste some text or drop in a text file, then ask about it. Studio quotes the passages that use the words in your question."],
};
const EXAMPLE = {
  source: "The Riverside Library is open from 9 am to 8 pm on weekdays and from 10 am to 4 pm on Saturdays. It is closed on Sundays and public holidays.\n\nMembers can borrow up to 12 books at a time for three weeks. Laptops can be borrowed for one day and must be returned to the front desk.\n\nPrinting costs 10 cents per page.",
  question: "How many books can members borrow?",
};
const THEMES = ["system", "light", "dark"];
const CODE_TASKS = ["code_generation", "code_debugging"];
// Touch keyboards have no Shift+Enter, so there Enter adds a line break and the arrow button sends.
const touch = matchMedia("(pointer: coarse)").matches;
const motion = () => (matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

// Storage can be switched off, for example in private windows, so reads and writes are allowed to fail.
function readStore(area, key) {
  try { return JSON.parse(window[area].getItem(key)); } catch { return null; }
}

function writeStore(area, key, value) {
  try { window[area].setItem(key, JSON.stringify(value)); return true; } catch { return false; }
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

// Answer settings carry over to the next visit; Reset answer settings brings back the defaults.
// Single-key shortcuts can be turned off, for speech or switch software that types / or ? by accident.
function savedKeys() {
  return readStore("localStorage", "studio-shortcuts") === "off" ? "off" : "on";
}

function applyKeys(value) {
  $(`keys-${value}`).checked = true;
  for (const row of document.querySelectorAll(".single-key")) row.hidden = value === "off";
}

function saveSettings() {
  const values = Object.fromEntries([...Object.keys(DEFAULTS), "stop-sequences"].map((id) => [id, $(id).value]));
  writeStore("localStorage", "studio-settings", { ...values, stopEdited: state.stopEdited });
}

function restoreSettings() {
  const saved = readStore("localStorage", "studio-settings");
  if (!saved || typeof saved !== "object") return;
  for (const id of [...Object.keys(DEFAULTS), "stop-sequences"]) {
    // A task type from an older page may no longer exist, and a select would then show nothing.
    if (typeof saved[id] === "string" && (id !== "task-type" || saved[id] in labels)) $(id).value = saved[id];
  }
  state.stopEdited = saved.stopEdited === true;
  state.stopTask = $("task-type").value;
}

// A newline stop keeps short answers to one line; multi-line tasks such as code start without one.
const defaultStops = (task) => (task === "language_generation" ? "\\n" : "");

function stopSequences() {
  // Commas and line breaks separate entries, so a typed \n or \t stands for that character.
  return wellFormed($("stop-sequences").value).split(/[,\n]/).map((item) => item.trim().replace(/\\n/g, "\n").replace(/\\t/g, "\t")).filter(Boolean);
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

// The preset whose sampling values all match these request options, if any.
function presetFor(options) {
  return Object.keys(PRESETS).find((name) => Object.entries(PRESETS[name]).every(([id, value]) => Math.abs(options[FIELDS[id]] - value) < 1e-9));
}

function describeSettings(options, custom) {
  const stops = options.stop_sequences?.map((item) => item.replace(/\n/g, "\\n").replace(/\t/g, "\\t")).join(", ");
  return [...(custom ? [`Task type ${labels[options.task_type]}`] : []), `Temperature ${options.temperature.toFixed(1)}`,
    `Top P ${options.top_p.toFixed(2)}`, `Top K ${options.top_k}`, `Repetition penalty ${options.repetition_penalty.toFixed(2)}`,
    `Up to ${plural(options.max_new_tokens, "token")}`, stops ? `Stops at ${stops}` : "No stop sequences"].join(" · ");
}

// TextEncoder counts a lone surrogate as U+FFFD while the server rejects one, so both fields send U+FFFD.
const wellFormed = (text) => text.replace(/[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]/g, (match) => (match.length === 2 ? match : "\ufffd"));
// The server rejects control characters other than tab and line breaks, including DEL, C1 and bidi controls,
// which pasted text sometimes carries, so the question is counted exactly as it will be sent.
const cleanPrompt = (text) => wellFormed(text).replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F‪-‮⁦-⁩]/g, " ").trim();

function promptTokens(text) {
  const prompt = cleanPrompt(text);
  if (!prompt) return 0;
  return 1 + encoder.encode(`<task_type>${settings().task_type}</task_type>\n<instruction>\n${prompt}\n</instruction>\n<answer>\n`).length;
}

// promptTokens counts byte tokens. A BPE model such as SmolLM2 with LoRA needs far fewer, so for those the
// page would block questions that fit, and the server's own token check decides instead.
function overflows(prompt, config) {
  const context = state.status?.model?.context_window;
  return customModel() && Boolean(context) && promptTokens(prompt) + config.max_new_tokens > context;
}

const TOO_LONG = "Shorten your question, or lower Answer length in Settings.";

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
  const count = promptTokens($("prompt").value);
  const current = phase();
  const context = state.status?.model?.context_window;
  const sourceBytes = bytes($("source-text").value);
  const questionBytes = bytes(cleanPrompt($("prompt").value));
  const sourceOverflow = grounded && (sourceBytes > LIMITS.source || questionBytes > LIMITS.question);
  const overflow = !grounded && overflows($("prompt").value, config);
  const validK = Number.isInteger(config.top_k) && config.top_k >= 0 && config.top_k <= 259;
  const validStops = !config.stop_sequences || (config.stop_sequences.length <= 4 && config.stop_sequences.every((item) => encoder.encode(item).length <= 64));
  const busy = Boolean(state.status?.busy);
  const thread = runs.length > 0;
  // Switching modes after the page has loaded eases the search box and the welcome text in.
  const switched = state.started && grounded !== document.body.classList.contains("searching");

  document.body.classList.toggle("has-thread", thread);
  document.body.classList.toggle("searching", grounded);
  $("intro").hidden = thread;
  $("thread").hidden = !thread;
  [$("intro-title").textContent, $("intro-text").textContent] = INTRO[grounded ? "sources" : "model"];
  $("source-panel").hidden = !grounded;
  if (switched && grounded) arrive($("source-panel"), { opacity: 0, transform: "translateY(8px)" });
  if (switched && !thread) for (const id of ["intro", "starters", "source-starters"]) arrive($(id), { opacity: 0, transform: "none" });
  $("prompt").placeholder = grounded ? "Ask about your text" : "Ask a question";
  $("mode-model").disabled = current === "disabled";
  $("task-field").hidden = !customModel();
  $("starters").hidden = grounded || thread || ["error", "disabled", "offline"].includes(current);
  $("source-starters").hidden = !grounded || thread || sourceBytes > 0;
  $("new-session").hidden = !thread;
  $("download").hidden = !thread;
  $("new-session").disabled = state.busy;

  $("source-count").textContent = `${sourceBytes.toLocaleString()} / ${LIMITS.source.toLocaleString()} bytes`;
  $("source-count").classList.toggle("over", sourceBytes > LIMITS.source);
  $("source-clear").hidden = !sourceBytes;
  // The count stays hidden until a limit is close, so short questions get a quiet composer.
  const [used, limit, unit] = grounded ? [questionBytes, LIMITS.question, "bytes"] : [count, context && customModel() ? Math.max(context - config.max_new_tokens, 0) : 0, "tokens"];
  const near = limit > 0 && used > limit * 0.8;
  $("token-count").textContent = near ? `${used.toLocaleString()} / ${limit.toLocaleString()} ${unit}` : "";
  $("token-count").classList.toggle("over", near && used > limit);

  $("max-tokens-value").value = `${config.max_new_tokens} tokens`;
  $("temperature-value").value = config.temperature.toFixed(1);
  $("top-p-value").value = config.top_p.toFixed(2);
  $("repetition-penalty-value").value = config.repetition_penalty.toFixed(2);
  $("top-k").setAttribute("aria-invalid", String(!validK));
  // Moving any sampling control off a preset leaves no preset selected.
  const preset = presetFor(config);
  for (const radio of document.querySelectorAll('input[name="preset"]')) radio.checked = radio.value === preset;
  $("stop-sequences").setAttribute("aria-invalid", String(!validStops));

  $("generate").disabled = state.busy || !count || current === "offline" || (grounded ? sourceOverflow || !sourceBytes : current !== "ready" || overflow || !validK || !validStops || busy);
  $("generate").classList.toggle("busy", state.busy);
  // Try again asks the model with the current settings, so it follows the send button's rules.
  state.canRetry = !state.busy && current === "ready" && !busy && validK && validStops;
  for (const button of document.querySelectorAll('[data-action="retry"]')) button.disabled = !retryAllowed(button.dataset.grounded === "true");
  $("generate").setAttribute("aria-label", state.busy ? "Working on an answer" : grounded ? "Search" : "Send");

  const problem = grounded ? "" : overflow ? TOO_LONG
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
  // Starting or clearing a conversation pins or unpins the composer without always resizing it.
  syncScrollButton(); syncDock();
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
  // An earlier turn only needs to be in view; the latest one also brings the page to the bottom.
  if (turn !== $("thread").lastElementChild) {
    turn.scrollIntoView({ block: "nearest", behavior: motion() });
    return;
  }
  // Long turns scroll to their start; short ones scroll the page to the bottom, above the composer.
  const room = window.innerHeight - document.querySelector(".dock").offsetHeight - 72;
  if (turn.offsetHeight > room) turn.scrollIntoView({ block: "start", behavior: motion() });
  else window.scrollTo({ top: document.documentElement.scrollHeight, behavior: motion() });
}

// New content eases in from where it came from. The stylesheet's reduced-motion rule can't reach
// script animations, so that preference is checked here.
function arrive(node, from = { opacity: 0, transform: "translateY(4px)" }) {
  if (!node || motion() === "auto") return;
  node.animate([from, { opacity: 1, transform: "none" }], { duration: 240, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
}

// Tabbing to a button under the pinned composer scrolls it into view above the composer, however tall
// pasted text has made it. A composer that scrolls with the page, on short screens, covers nothing.
function syncDock() {
  const dock = document.querySelector(".dock");
  const pinned = getComputedStyle(dock).position === "sticky";
  document.documentElement.style.setProperty("--dock-height", `${pinned ? dock.offsetHeight : 0}px`);
}

// The jump button shows once the end of the conversation is out of view.
function syncScrollButton() {
  const below = document.documentElement.scrollHeight - (window.scrollY + window.innerHeight);
  $("scroll-latest").hidden = !runs.length || below < 120;
}

function addTurn(run, restored = false) {
  run.turn = element("article", restored ? "turn restored" : "turn");
  const row = element("div", "question-row");
  const edit = iconButton("edit", "Edit question");
  edit.addEventListener("click", () => reuse(run));
  run.question = element("p", "question", run.prompt);
  run.question.tabIndex = -1;
  row.append(edit, run.question);
  run.turn.append(row);
  if (run.tag) run.turn.append(element("p", "question-tag", run.tag));
  run.answer = element("div", "answer");
  run.answer.tabIndex = -1;
  run.turn.append(run.answer);
  $("thread").append(run.turn);
}

// Icons live in a template in index.html, so markup stays out of the script.
function icon(name) {
  return $("icons").content.querySelector(`[data-icon="${name}"]`).cloneNode(true);
}

function iconButton(name, label) {
  const button = element("button", "action");
  button.type = "button";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.append(icon(name));
  return button;
}

function copyButton(text, label) {
  const button = iconButton("copy", label);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.replaceChildren(icon("check"));
      button.classList.add("done");
      announce("Copied");
      setTimeout(() => { button.replaceChildren(icon("copy")); button.classList.remove("done"); }, 1600);
    } catch {
      notify("Copying isn't available here. Select the text to copy it instead.", true);
    }
  });
  return button;
}

// Searching needs only a running server; the model also has to be ready and free.
function retryAllowed(grounded) {
  return grounded ? !state.busy && phase() !== "offline" : Boolean(state.canRetry);
}

function retryButton(run, labelled = false) {
  const button = iconButton("retry", "Try again");
  // After a failure, Try again is the way forward, so it gets words as well as an icon.
  if (labelled) {
    button.classList.add("labelled");
    button.append(element("span", "", "Try again"));
  }
  button.dataset.action = "retry";
  button.dataset.grounded = String(run.grounded);
  button.disabled = !retryAllowed(run.grounded);
  // Answer length may have gone up since the question was sent, so the earlier question gets Send's check.
  button.addEventListener("click", () => (!run.grounded && overflows(run.prompt, settings()) ? notify(TOO_LONG, true) : ask(run, true)));
  return button;
}

// Each try is kept, and the arrows flip between them.
function pager(run) {
  const group = element("div", "pager");
  const previous = iconButton("previous", "Previous answer");
  const next = iconButton("next", "Next answer");
  // aria-disabled instead of disabled, so focus stays on an arrow that reaches the end.
  for (const [button, action, step, end] of [[previous, "previous", -1, 0], [next, "next", 1, run.answers.length - 1]]) {
    button.dataset.action = action;
    button.setAttribute("aria-disabled", String(run.shown === end));
    button.addEventListener("click", () => show(run, run.shown + step, step));
  }
  group.append(previous, element("span", "", `${run.shown + 1} / ${run.answers.length}`), next);
  return group;
}

// The next try slides in from the right and the previous one from the left, while the arrows stay put.
function show(run, index, step) {
  if (index < 0 || index >= run.answers.length || index === run.shown) return;
  run.shown = index;
  announce(`Answer ${index + 1} of ${run.answers.length}. ${redraw(run)}`);
  arrive(run.answer.firstElementChild, { opacity: 0.2, transform: `translateX(${step * 8}px)` });
  saveThread();
}

// Drawing replaces the answer's buttons, so focus moves to the same button in the new drawing.
function redraw(run) {
  const action = run.answer.contains(document.activeElement) ? document.activeElement.dataset.action : null;
  const message = drawAnswer(run);
  if (action) run.answer.querySelector(`[data-action="${action}"]`)?.focus();
  return message;
}

function actions(...buttons) {
  const group = element("div", "actions");
  group.append(...buttons);
  return group;
}

// The row under an answer: arrows between tries, a short summary, then the actions.
function answerMeta(run, summary, buttons, ...extra) {
  const meta = element("div", "answer-meta");
  if (run.answers.length > 1) meta.append(pager(run));
  if (summary) meta.append(element("span", "", summary));
  meta.append(...extra, actions(...buttons));
  return meta;
}

// Tries can use different settings, so each answer can show the ones it was made with.
function settingsToggle(run, answer) {
  const preset = presetFor(answer.options);
  const name = preset ? document.querySelector(`label[for="preset-${preset}"]`).textContent : "Custom";
  const button = element("button", "settings-used", name);
  button.type = "button";
  button.dataset.action = "settings";
  button.setAttribute("aria-label", `Settings used: ${name}`);
  button.setAttribute("aria-expanded", String(Boolean(answer.open)));
  button.append(icon("expand"));
  button.addEventListener("click", () => {
    answer.open = !answer.open;
    redraw(run);
    arrive(run.answer.querySelector(".answer-settings"), { opacity: 0, transform: "translateY(-4px)" });
  });
  return button;
}

// Marks the words that matched the question. Offsets count code points, as Python strings do.
function highlighted(text, spans) {
  const quote = element("blockquote");
  const characters = Array.from(text);
  let at = 0;
  for (const [start, end] of Array.isArray(spans) ? spans : []) {
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < at || end <= start || end > characters.length) continue;
    quote.append(characters.slice(at, start).join(""), element("mark", "", characters.slice(start, end).join("")));
    at = end;
  }
  quote.append(characters.slice(at).join(""));
  return quote;
}

// Draws the answer a turn is showing and returns what a screen reader should hear about it.
function drawAnswer(run) {
  const answer = run.answers[run.shown];
  if (answer.pending) {
    const typing = element("div", "typing");
    const elapsed = element("span", "elapsed");
    // The running count is for sighted users; screen readers already heard that an answer is on its way.
    elapsed.setAttribute("aria-hidden", "true");
    typing.append(element("span", "visually-hidden", "Working on an answer"), element("i"), element("i"), element("i"), elapsed);
    run.answer.replaceChildren(typing);
    return "Working on an answer";
  }
  if (answer.error) {
    run.answer.replaceChildren(element("p", answer.interrupted ? "answer-note" : "answer-note error", answer.error), answerMeta(run, "", [retryButton(run, true)]));
    return answer.error;
  }
  return showResult(run, answer);
}

function showResult(run, answer) {
  const response = answer.response;
  if (run.grounded) {
    // Abstentions carry fallback text, so they show a note instead of copyable excerpts.
    const sources = response.abstained ? [] : response.sources || [];
    if (!sources.length) {
      run.answer.replaceChildren(element("p", "answer-note", "No passage in your text uses enough of the words in your question. Try fewer words, or words from your text."));
      return "No matching passages in your text.";
    }
    const list = element("ol", "excerpts");
    for (const source of sources) {
      const item = element("li");
      item.append(element("span", "cite", source.id), highlighted(source.text, source.matches));
      list.append(item);
    }
    run.answer.replaceChildren(list, answerMeta(run, `${plural(sources.length, "passage")} quoted from your text`, [copyButton(response.text, "Copy passages")]));
    return `Found ${plural(sources.length, "passage")}. ${sources.map((source) => source.text).join(" ")}`;
  }
  // Leading blank lines are noise; indentation on the first line is kept for code.
  const text = String(response.text || "").replace(/^\n+/, "").trimEnd();
  const details = [plural(response.generated_tokens, "token"), `${Number(response.elapsed_seconds).toFixed(1)}s`];
  if (response.finish_reason === "length") details.push("stopped at the length limit");
  const meta = answerMeta(run, details.join(" · "), [...(text ? [copyButton(text, "Copy answer")] : []), retryButton(run)], settingsToggle(run, answer));
  const used = answer.open ? [element("p", "answer-settings", describeSettings(answer.options, answer.custom))] : [];
  if (text) {
    run.answer.replaceChildren(element("pre", answer.code ? "answer-text code" : "answer-text", text), meta, ...used);
    return text;
  }
  const empty = response.finish_reason === "stop" ? "A stop sequence ended the response before any text. Clear Stop sequences in Settings, under Advanced, and try again."
    : "The model returned no text. Try rephrasing, or raise the temperature in Settings.";
  run.answer.replaceChildren(element("p", "answer-note", empty), meta, ...used);
  return empty;
}

async function post(path, payload) {
  let response;
  try {
    response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  } catch {
    throw new Error("Couldn't reach Studio. Check that it's still running, then try again.");
  }
  const result = await response.json().catch(() => null);
  if (!response.ok) throw Object.assign(new Error(result?.error || `Studio answered with an error (${response.status}). Try again.`), { status: response.status });
  // A body cut off partway would otherwise be drawn, and saved, as an empty answer.
  if (!result) throw new Error("Studio's answer arrived incomplete. Try again.");
  return result;
}

// Asks for one more answer to a turn with the current settings, and shows it when it arrives.
async function ask(run, again = false) {
  if (state.busy) return;
  const grounded = run.grounded;
  const answer = { options: grounded ? null : settings(), pending: true };
  answer.custom = !grounded && customModel();
  answer.code = answer.custom && CODE_TASKS.includes(answer.options.task_type);
  // Failed tries give way to the new one instead of staying among the answers, wherever they sit.
  run.answers = run.answers.filter((earlier) => !earlier.error);
  run.answers.push(answer);
  run.shown = run.answers.length - 1;
  state.busy = true; state.slow = false; state.notice = null;
  if (!run.turn) addTurn(run);
  const waiting = drawAnswer(run);
  // The button just used disappears while the answer loads, so focus waits on the answer itself.
  // A screen reader reads the focused answer, which already says one is on its way, so it isn't announced twice.
  if (again) run.answer.focus({ preventScroll: true });
  // Touch screens: the chip or button just used may have disappeared, so hand focus to the new question.
  else if (touch) run.question.focus({ preventScroll: true });
  if (!again) announce(waiting);
  syncComposer(); reveal(run.turn);
  const slow = setTimeout(() => { state.slow = true; syncComposer(); }, 8000);
  const started = performance.now();
  const clock = setInterval(() => {
    const elapsed = run.answer.querySelector(".elapsed");
    if (elapsed) elapsed.textContent = `${Math.floor((performance.now() - started) / 1000)}s`;
  }, 1000);
  try {
    const payload = grounded ? { prompt: run.prompt, source_text: run.source_text } : { prompt: run.prompt, ...answer.options };
    answer.response = await post(grounded ? "/api/grounded" : "/api/generate", payload);
  } catch (error) {
    [answer.error, answer.status] = [error.message, error.status];
  } finally {
    delete answer.pending;
    clearTimeout(slow); clearInterval(clock);
    state.busy = false; state.slow = false;
    // Searches never hold the model, and a 409 means another request still does.
    if (!grounded && answer.status !== 409) {
      state.answered += 1;
      if (state.status) state.status.busy = false;
    }
    announce(drawAnswer(run));
    arrive(run.answer);
    // Leaving the page cancels the request; saving now would record that as a failure.
    if (!state.leaving) saveThread();
    syncComposer(); reveal(run.turn);
    if (again && [document.body, run.answer].includes(document.activeElement)) run.answer.querySelector('[data-action="retry"]')?.focus();
    else if (!touch && [document.body, $("generate")].includes(document.activeElement)) $("prompt").focus();
  }
}

$("prompt-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if ($("generate").disabled) return;
  const grounded = sourceMode();
  const task = $("task-type").value;
  runs.push({ prompt: cleanPrompt($("prompt").value), grounded, source_text: grounded ? wellFormed($("source-text").value) : "",
    tag: grounded ? "Searched your text" : customModel() && task !== "language_generation" ? labels[task] : "", answers: [], shown: 0 });
  $("prompt").value = "";
  autosize();
  ask(runs.at(-1));
});

// The conversation lasts as long as the tab: a refresh keeps it, closing the tab clears it.
function saveThread() {
  const keep = ({ options, custom, code, response, error, status, interrupted }) => ({ options, custom, code, response, error, status, interrupted });
  if (writeStore("sessionStorage", "studio-thread", {
    mode: sourceMode() ? "sources" : "model", source: $("source-text").value, draft: $("prompt").value,
    runs: runs.map(({ prompt, grounded, source_text, tag, answers, shown }) => ({ prompt, grounded, source_text, tag, shown, answers: answers.filter((answer) => !answer.pending).map(keep) })),
  })) {
    state.unsaved = false;
    return;
  }
  // A full store keeps the last copy that fit, and a refresh would bring back that older conversation unannounced.
  try { window.sessionStorage.removeItem("studio-thread"); } catch { /* Storage is off, so nothing old can come back. */ }
  // Storage that is switched off fails every save; say so only once there is a conversation to lose.
  if (!state.unsaved && runs.length) {
    notify("Studio couldn't save this conversation, so a refresh will clear it. Download it to keep a copy.", true);
    state.unsaved = true;
  }
}

function restoreThread() {
  const saved = readStore("sessionStorage", "studio-thread");
  if (!saved || !Array.isArray(saved.runs)) return;
  $(saved.mode === "sources" ? "mode-sources" : "mode-model").checked = true;
  if (typeof saved.source === "string") $("source-text").value = saved.source;
  if (typeof saved.draft === "string") $("prompt").value = saved.draft;
  try {
    for (const item of saved.runs) {
      const answers = (item.answers || []).filter((answer) => typeof answer.error === "string" || (answer.response && typeof answer.response === "object"));
      // A refresh while an answer was on its way loses that answer, and Try again asks for it again.
      if (!answers.length) answers.push({ error: "The page was refreshed before this answer arrived.", interrupted: true });
      const run = { prompt: String(item.prompt), grounded: item.grounded === true, source_text: String(item.source_text || ""), tag: String(item.tag || ""),
        answers, shown: Math.min(Math.max(Number(item.shown) || 0, 0), answers.length - 1) };
      runs.push(run);
      addTurn(run, true);
      drawAnswer(run);
    }
  } catch {
    // A saved conversation that no longer draws is dropped rather than half shown.
    runs.length = 0;
    $("thread").replaceChildren();
  }
}

function newSession() {
  if (state.busy) return;
  runs.length = 0; state.notice = null;
  $("thread").replaceChildren();
  $("prompt").value = "";
  saveThread(); autosize(); syncComposer();
  window.scrollTo({ top: 0 });
  $(touch ? "intro-title" : "prompt").focus();
}

// A Markdown copy of the conversation, with the settings behind each answer, for notes or a results log.
// Answers and quoted passages go in fenced blocks, which show their text exactly as the page does: no tag, entity,
// backslash or indent inside is read as markup, and a fence longer than any backtick run inside stays closed.
const fenced = (text) => { const fence = "`".repeat(Math.max(3, ...(text.match(/`+/g) || []).map((ticks) => ticks.length + 1))); return [fence, text, fence]; };
// A question, an error or a setting is plain text, so it escapes every character Markdown would read as markup.
const escapeMarkdown = (text) => text.replace(/[\\`*_[\]<&~#]/g, "\\$&");

function transcript() {
  const model = state.status?.model?.name;
  const lines = ["# slm studio", "", `${model ? `${escapeMarkdown(model)}, saved` : "Saved"} ${new Date().toLocaleString()}`];
  // Blank lines only separate blocks, so a block never adds a second one; text inside an answer is left as it is.
  const add = (...items) => { for (const item of items) if (item !== "" || lines.at(-1) !== "") lines.push(item); };
  for (const run of runs) {
    add("", `## ${escapeMarkdown(run.prompt.replace(/\s+/g, " "))}`, "");
    if (run.tag) add(`*${run.tag}*`, "");
    const answers = run.answers.filter((answer) => !answer.pending);
    answers.forEach((answer, index) => {
      if (answers.length > 1) add(`**Answer ${index + 1} of ${answers.length}**`, "");
      const response = answer.response;
      if (answer.error) add(`*${escapeMarkdown(answer.error)}*`, "");
      else if (run.grounded) {
        const sources = response.abstained ? [] : response.sources || [];
        if (!sources.length) add("*No passage in the text matched the question.*", "");
        for (const source of sources) add(`**${source.id}**`, "", ...fenced(source.text), "");
      } else {
        const text = String(response.text || "").replace(/^\n+/, "").trimEnd();
        add(...(text ? fenced(text) : ["*No text.*"]), "");
        add(`*${plural(response.generated_tokens, "token")} · ${Number(response.elapsed_seconds).toFixed(1)}s · ${escapeMarkdown(describeSettings(answer.options, answer.custom))}*`, "");
      }
    });
  }
  return `${lines.join("\n").trimEnd()}\n`;
}

function download() {
  const now = new Date();
  const pad = (number) => String(number).padStart(2, "0");
  const name = `slm-studio-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}.md`;
  const url = URL.createObjectURL(new Blob([transcript()], { type: "text/markdown" }));
  Object.assign(element("a"), { href: url, download: name }).click();
  // The click starts the download before the next task, so the address can be released then.
  setTimeout(() => URL.revokeObjectURL(url));
  announce(`Downloaded ${name}.`);
}

// Loads a text file into Search my text, from Open a file or a drop anywhere on the page.
async function openFile(file) {
  if (!file) return;
  const textual = file.type.startsWith("text/") || file.type === "application/json" || /\.(txt|md|markdown|csv|json|log)$/i.test(file.name);
  if (!textual) return notify("Studio can only search plain text files, such as .txt or .md.", true);
  const tooBig = `${file.name} is longer than ${LIMITS.source.toLocaleString()} bytes, the most Search my text takes. Paste the part you need instead.`;
  // Reading is skipped for anything far over the limit; line endings are counted once they are normalized.
  if (file.size > LIMITS.source * 4) return notify(tooBig, true);
  let text;
  try {
    text = (await file.text()).replace(/\r\n?/g, "\n");
  } catch {
    return notify(`Couldn't read ${file.name}. Try opening it again.`, true);
  }
  if (text.includes("\u0000")) return notify(`${file.name} doesn't look like plain text.`, true);
  if (bytes(text) > LIMITS.source) return notify(tooBig, true);
  $("mode-sources").checked = true;
  $("source-text").value = text;
  state.notice = null;
  syncComposer();
  $("prompt").focus();
  announce(`Opened ${file.name}.`);
}

// Puts an earlier question back in the box, in the mode it was asked in, ready to change and send.
function reuse(run) {
  if (run.grounded || !$("mode-model").disabled) $(run.grounded ? "mode-sources" : "mode-model").checked = true;
  // Pasted text is only filled in when the box is empty, so newer text is never replaced.
  if (run.grounded && !$("source-text").value) $("source-text").value = run.source_text;
  $("prompt").value = run.prompt;
  state.notice = null;
  autosize(); syncComposer();
  $("prompt").focus();
  $("prompt").setSelectionRange(run.prompt.length, run.prompt.length);
}

async function pollStatus() {
  const before = phase();
  const answered = state.answered;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request failed with ${response.status}.`);
    const status = await response.json();
    // A poll the server answered while this page's own request held the model would undo the busy = false
    // set when that answer arrived.
    if (state.answered !== answered && status?.busy) status.busy = false;
    state.status = status;
    state.offline = false;
  } catch {
    state.offline = true;
  }
  // A notice from before the server stopped or came back would hide the message about that change.
  if (phase() !== before) state.notice = null;
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
// A notice about a file or earlier text no longer applies once the text to search changes.
$("source-text").addEventListener("input", () => { state.notice = null; });
window.addEventListener("resize", () => { autosize(); syncScrollButton(); syncDock(); });
// Dragging the pasted-text box taller changes the composer without any other event.
new ResizeObserver(syncDock).observe(document.querySelector(".dock"), { box: "border-box" });
for (const id of ["prompt", "source-text", "task-type", "temperature", "max-tokens", "top-k", "top-p", "repetition-penalty", "stop-sequences"]) $(id).addEventListener("input", syncComposer);
$("stop-sequences").addEventListener("input", () => { state.stopEdited = true; });
for (const radio of document.querySelectorAll('input[name="mode"]')) radio.addEventListener("change", () => { state.notice = null; syncComposer(); });
for (const radio of document.querySelectorAll('input[name="theme"]')) radio.addEventListener("change", () => {
  applyTheme(radio.value);
  writeStore("localStorage", "studio-theme", radio.value);
});
for (const radio of document.querySelectorAll('input[name="keys"]')) radio.addEventListener("change", () => {
  applyKeys(radio.value);
  writeStore("localStorage", "studio-shortcuts", radio.value);
});
// Other Studio tabs pick up a theme or shortcut change right away.
window.addEventListener("storage", (event) => {
  if (event.key === "studio-theme") applyTheme(savedTheme());
  if (event.key === "studio-shortcuts") applyKeys(savedKeys());
});
for (const radio of document.querySelectorAll('input[name="preset"]')) radio.addEventListener("change", () => {
  for (const [id, value] of Object.entries(PRESETS[radio.value])) $(id).value = String(value);
  syncComposer();
});

$("prompt").addEventListener("keydown", (event) => {
  // Up in an empty box brings back the last question, as in a terminal.
  if (event.key === "ArrowUp" && !$("prompt").value && runs.length && !event.shiftKey && !event.altKey && !event.metaKey && !event.ctrlKey) {
    event.preventDefault();
    reuse(runs.at(-1));
    return;
  }
  // Enter sends. Shift+Enter, IME composition (Safari flags it only by keyCode 229) and touch keyboards add a line break.
  if (event.key !== "Enter" || event.shiftKey || event.metaKey || event.ctrlKey || event.isComposing || event.keyCode === 229 || touch) return;
  event.preventDefault();
  $("prompt-form").requestSubmit();
});
// A click on the composer's empty space lands in the question box, but never takes focus from a control or a text selection.
$("prompt-form").addEventListener("click", (event) => {
  if (!event.target.closest("button, input, textarea, label, select, .source") && !String(getSelection())) $("prompt").focus();
});
document.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]") || event.altKey) return;
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); $("prompt-form").requestSubmit(); return; }
  if (event.metaKey || event.ctrlKey || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (touch || !$("keys-on").checked) return;
  if (event.key === "/") { event.preventDefault(); $("prompt").focus(); }
  if (event.key === "?") {
    event.preventDefault();
    $("about").showModal();
    $("shortcuts").scrollIntoView({ block: "nearest" });
  }
});

$("new-session").addEventListener("click", newSession);
$("download").addEventListener("click", download);
$("source-open").addEventListener("click", () => $("source-file").click());
$("source-clear").addEventListener("click", () => {
  $("source-text").value = "";
  syncComposer();
  // The button hides once the text is gone, so focus moves to the empty box.
  $("source-text").focus();
});
$("source-file").addEventListener("change", () => {
  openFile($("source-file").files[0]);
  // Clearing lets the same file be picked again after it changes on disk.
  $("source-file").value = "";
});
// A file dropped anywhere is searched instead of replacing the page, which is what the browser would do.
const carriesFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
let dragDepth = 0;
document.addEventListener("dragenter", (event) => {
  if (!carriesFiles(event)) return;
  dragDepth += 1;
  document.body.classList.add("dragging");
});
document.addEventListener("dragleave", (event) => {
  if (!carriesFiles(event) || --dragDepth > 0) return;
  dragDepth = 0;
  document.body.classList.remove("dragging");
});
document.addEventListener("dragover", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
});
document.addEventListener("drop", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  openFile(event.dataTransfer.files[0]);
});
$("scroll-latest").addEventListener("click", () => {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: motion() });
  // The button hides once the page is at the bottom, so focus moves on to the latest answer.
  runs.at(-1)?.answer.focus({ preventScroll: true });
});
window.addEventListener("scroll", syncScrollButton, { passive: true });
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
  else $("prompt").focus();
});
$("reset-settings").addEventListener("click", () => {
  for (const [id, value] of Object.entries(DEFAULTS)) $(id).value = value;
  state.stopEdited = false;
  $("stop-sequences").value = defaultStops($("task-type").value);
  state.stopTask = $("task-type").value;
  syncComposer(); saveSettings();
});
// Change events arrive after a preset has filled in its values, and after a slider is let go.
for (const type of ["input", "change"]) $("settings").addEventListener(type, saveSettings);
for (const name of ["settings", "about"]) {
  const dialog = $(name);
  let pressed = false;
  $(`${name}-open`).addEventListener("click", () => dialog.showModal());
  dialog.querySelectorAll(".close-dialog").forEach((button) => button.addEventListener("click", () => dialog.close()));
  // Close on a backdrop click, but not when a drag that started inside, on a slider say, ends outside.
  dialog.addEventListener("pointerdown", (event) => { pressed = event.target === dialog; });
  dialog.addEventListener("click", (event) => { if (pressed && event.target === dialog) dialog.close(); });
}

// Typing and pasting are saved when the page goes away, rather than on every keystroke.
window.addEventListener("pagehide", () => { state.leaving = true; saveThread(); });
window.addEventListener("pageshow", () => { state.leaving = false; });
document.addEventListener("visibilitychange", () => { if (document.hidden) saveThread(); });

// Phones have no keyboard to take shortcuts from, and Macs name the modifier differently.
$("shortcuts").hidden = touch;
$("keys-field").hidden = touch;
if (/Mac|iPhone|iPad/.test(navigator.platform)) for (const key of document.querySelectorAll(".modifier")) key.textContent = "\u2318";
applyTheme(savedTheme()); applyKeys(savedKeys()); restoreSettings(); restoreThread();
autosize(); renderStatus(); syncComposer(); pollStatus();
state.started = true;
if (!touch) $("prompt").focus();
