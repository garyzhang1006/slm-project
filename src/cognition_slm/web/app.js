"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, busy: false, runs: [], selected: null };
const encoder = new TextEncoder();
const labels = Object.fromEntries([...$("task-type").options].map((option) => [option.value, option.text]));

function settings() {
  return { task_type: $("task-type").value, temperature: Number($("temperature").value),
    max_new_tokens: Number($("max-tokens").value), top_k: Number($("top-k").value) };
}

function promptTokens() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return 0;
  return 1 + encoder.encode(`<task_type>${settings().task_type}</task_type>\n<instruction>\n${prompt}\n</instruction>\n<answer>\n`).length;
}

function syncComposer() {
  const config = settings();
  const count = promptTokens();
  const context = state.status?.model?.context_window;
  const overflow = context && count + config.max_new_tokens > context;
  const validK = Number.isInteger(config.top_k) && config.top_k >= 0 && config.top_k <= 259;
  $("token-count").textContent = `${count.toLocaleString()}${context ? ` / ${context.toLocaleString()}` : ""} tokens`;
  $("token-count").classList.toggle("over-budget", Boolean(overflow));
  $("task-label").textContent = labels[config.task_type];
  $("temperature-value").value = config.temperature.toFixed(1);
  $("max-tokens-value").value = `${config.max_new_tokens} tokens`;
  $("generate").disabled = state.busy || state.status?.state !== "ready" || !count || overflow || !validK || Boolean(state.status?.busy);
  $("budget-note").textContent = overflow
    ? `Shorten your prompt: reserve ${config.max_new_tokens} tokens for the response.`
    : "Each prompt starts fresh. History is for your reference.";
  $("budget-note").classList.toggle("over-budget", Boolean(overflow));
  $("new-session").disabled = state.busy;
  $("mobile-new").disabled = state.busy;
}

function feedback(message, error = false) {
  $("feedback").textContent = message;
  $("feedback").classList.toggle("error", error);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderRun(run, loading = false) {
  $("intro").hidden = true;
  $("starters").hidden = true;
  const container = $("conversation");
  container.hidden = false;
  container.replaceChildren(element("div", "message-label", "YOUR PROMPT"), element("p", "user-prompt", run.prompt));
  const header = element("div", "response-header");
  header.append(element("div", "message-label", loading ? "GENERATING RESPONSE" : "MODEL RESPONSE"));
  container.append(header);
  if (loading) {
    const skeleton = element("div", "response-text");
    skeleton.setAttribute("aria-busy", "true");
    skeleton.setAttribute("aria-label", "Generating response");
    for (let i = 0; i < 3; i++) skeleton.append(element("div", "loading-line"));
    container.append(skeleton);
    return;
  }
  const response = run.result;
  const hasText = Boolean(response.text?.trim());
  container.append(element("pre", `response-text${hasText ? "" : " empty-response"}`, hasText ? response.text : "The model returned no visible text. Try another prompt or a higher temperature; this checkpoint is still at an early training stage."));
  if (hasText) {
    const copy = element("button", "copy-button", "Copy response");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(response.text); copy.textContent = "Copied"; }
      catch { feedback("Copy unavailable. Select the response text to copy it manually.", true); }
    });
    header.append(copy);
  }
  const elapsed = Number(response.elapsed_seconds).toFixed(1);
  container.append(element("p", "response-stats", `${response.generated_tokens} generated tokens · ${elapsed}s · ${labels[run.options.task_type]} · ${response.finish_reason === "eos" ? "End of response" : "Response limit reached"}`));
}

function renderHistory() {
  $("run-count").textContent = String(state.runs.length).padStart(2, "0");
  $("history-list").replaceChildren();
  if (!state.runs.length) $("history-list").append(element("p", "history-empty", "Your experiments will appear here."));
  [...state.runs].reverse().forEach((run) => {
    const button = element("button", `history-item${run.id === state.selected ? " active" : ""}`, run.prompt);
    button.title = run.prompt;
    button.disabled = state.busy;
    button.addEventListener("click", () => {
      state.selected = run.id;
      $("prompt").value = run.prompt;
      $("task-type").value = run.options.task_type;
      renderRun(run); renderHistory(); syncComposer(); feedback("Previous run. Edit the prompt to try again.");
    });
    $("history-list").append(button);
  });
}

function newSession() {
  if (state.busy) return;
  state.selected = null;
  $("prompt").value = "";
  $("intro").hidden = false;
  $("starters").hidden = false;
  $("conversation").hidden = true;
  feedback(""); renderHistory(); syncComposer(); $("prompt").focus();
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not connect to the local model.");
    state.status = await response.json();
    const { model, state: phase, error } = state.status;
    $("connection").textContent = phase === "ready" ? "Local model ready" : phase === "loading" ? "Loading model" : "Model unavailable";
    $("connection").className = `connection ${phase === "ready" ? "ready" : phase === "error" ? "error" : ""}`;
    if (model) {
      const size = model.parameters ? `${(model.parameters / 1e6).toFixed(1)}M` : "";
      const context = model.context_window ? `${model.context_window.toLocaleString()} context` : "";
      $("sidebar-detail").textContent = [size, model.device?.toUpperCase(), context].filter(Boolean).join(" · ") || "Loading checkpoint";
      $("composer-model").textContent = `Cognition ${size}`;
      const details = { Parameters: model.parameters?.toLocaleString(), Context: model.context_window ? `${model.context_window.toLocaleString()} byte tokens` : "Unknown", Device: model.device, Architecture: model.architecture, "Training iterations": model.training_steps ?? "Not recorded" };
      $("model-details").replaceChildren(...Object.entries(details).map(([key, value]) => {
        const row = element("div"); row.append(element("dt", "", key), element("dd", "", String(value ?? "Unknown"))); return row;
      }));
    }
    if (phase === "error") feedback(error || "Checkpoint could not load. Check the server terminal.", true);
    else if (phase === "loading" && !state.busy) feedback("Loading your checkpoint into memory. The first start can take a moment.");
    else if (["Loading your checkpoint", "Cannot reach the local server"].some((prefix) => $("feedback").textContent.startsWith(prefix))) feedback("");
  } catch {
    state.status = null;
    $("connection").textContent = "Disconnected";
    $("connection").className = "connection error";
    if (!state.busy) feedback("Cannot reach the local server. Start launch-studio.command, then refresh.", true);
  }
  syncComposer();
  setTimeout(pollStatus, state.status?.state === "loading" ? 1200 : 5000);
}

$("prompt-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("generate").disabled) return;
  const run = { id: Date.now(), prompt: $("prompt").value.trim(), options: settings() };
  state.busy = true; syncComposer(); renderHistory(); renderRun(run, true);
  $("generate").querySelector("span").textContent = "Generating…";
  feedback("Running on your machine. Longer responses take more time.");
  try {
    const response = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: run.prompt, ...run.options }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Generation failed. Try again.");
    run.result = result; state.runs.push(run); state.selected = run.id;
    renderRun(run); feedback("Run complete. Your prompt and response stay in this session.");
  } catch (error) {
    $("conversation").hidden = true; $("intro").hidden = false; $("starters").hidden = false;
    feedback(error.message || "Connection lost. Check the local server and try again.", true);
  } finally {
    state.busy = false;
    if (state.status) state.status.busy = false;
    $("generate").querySelector("span").textContent = "Run prompt";
    renderHistory(); syncComposer();
  }
});

$("prompt").addEventListener("input", syncComposer);
for (const id of ["task-type", "temperature", "max-tokens", "top-k"]) $(id).addEventListener("input", syncComposer);
$("new-session").addEventListener("click", newSession);
$("mobile-new").addEventListener("click", newSession);
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => {
  $("prompt").value = button.dataset.prompt; $("task-type").value = button.dataset.task;
  syncComposer(); $("prompt").focus();
}));
for (const name of ["settings", "about"]) {
  $(`${name}-open`).addEventListener("click", () => $(name).showModal());
  $(name).querySelectorAll(".close-dialog").forEach((button) => button.addEventListener("click", () => $(name).close()));
}
document.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]")) return;
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); $("prompt-form").requestSubmit(); }
  if (event.key.toLowerCase() === "n" && !event.metaKey && !event.ctrlKey && !event.altKey && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) && !document.querySelector("dialog[open]")) newSession();
});
syncComposer(); pollStatus();
