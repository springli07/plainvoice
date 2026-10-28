"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const source = byId("source");
  const analyzeButton = byId("analyze-button");
  const rewriteButton = byId("rewrite-button");
  const maxCharacters = 50000;
  let configuration = null;
  let busy = false;
  let pendingController = null;
  let revision = 0;
  let outputText = "";
  const example = "不难看出，这次更新让团队的工作方式发生了变化。我们在 2026 年 10 月开始试用，处理一份材料的时间从 30 分钟缩短到 18 分钟。\n\n综上所述，工具是否好用，需要看具体任务。小周说：\"它能帮我整理记录，但结论还得自己核对。\" 目前只有 12 位同事参加试用，还不能据此判断长期效果。";

  const characterCount = (text) => Array.from(text).length;
  const formatCount = (count) => new Intl.NumberFormat("zh-CN").format(count);
  const announce = (message) => { byId("activity-status").textContent = message; };
  const setError = (message = "") => {
    byId("error-message").textContent = message;
    byId("error-message").hidden = !message;
  };

  function updateButtons() {
    const length = characterCount(source.value);
    const hasText = source.value.trim().length > 0;
    const valid = hasText && length <= maxCharacters;
    analyzeButton.disabled = busy || !valid;
    rewriteButton.disabled = busy || !valid || !configuration?.configured;
    byId("clear-source").disabled = !source.value;
    byId("source-count").textContent = `${formatCount(length)} 字符 / 50,000`;
    byId("source-count").style.color = length > maxCharacters ? "#a34d35" : "";
    source.setAttribute("aria-invalid", String(length > maxCharacters));
  }

  function setBusy(value, kind = "") {
    busy = value;
    for (const [button, label, icon, action] of [
      [analyzeButton, "分析文风", "⌕", "analyze"],
      [rewriteButton, "保留原意改写", "↗", "rewrite"],
    ]) {
      const loading = value && kind === action;
      button.classList.toggle("is-loading", loading);
      button.querySelector(".button-label").textContent = loading ? (action === "analyze" ? "分析中…" : "改写中…") : label;
      button.querySelector(".button-icon").textContent = loading ? "" : icon;
    }
    document.querySelector(".analysis-panel").setAttribute("aria-busy", String(value && kind === "analyze"));
    document.querySelector(".rewrite-panel").setAttribute("aria-busy", String(value && kind === "rewrite"));
    updateButtons();
  }

  async function request(path, body, timeoutMilliseconds = 15000) {
    const controller = new AbortController();
    if (body !== undefined) pendingController = controller;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMilliseconds);
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? { Accept: "application/json" } : { "Content-Type": "application/json", Accept: "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        credentials: "same-origin",
      });
      let payload;
      try { payload = await response.json(); } catch { throw new Error("服务返回了无法识别的结果，请检查本地服务是否正常。"); }
      if (!response.ok) {
        const detail = typeof payload?.error === "string" ? payload.error : "请求未完成，请检查服务配置后重试。";
        throw new Error(detail);
      }
      return payload;
    } catch (error) {
      if (timedOut) throw new Error("请求超时，请检查模型服务或稍后重试。");
      if (error.name === "AbortError") throw error;
      if (error instanceof TypeError) throw new Error("无法连接本地服务，请确认 PlainVoice 仍在运行。");
      throw error;
    } finally {
      clearTimeout(timer);
      if (pendingController === controller) pendingController = null;
    }
  }

  function resetResults() {
    byId("analysis-empty").hidden = false;
    byId("analysis-result").hidden = true;
    byId("analysis-badge").textContent = "等待分析";
    byId("findings-list").replaceChildren();
    byId("rewrite-empty").hidden = false;
    byId("rewrite-result").hidden = true;
    byId("copy-output").hidden = true;
    outputText = "";
  }

  function selectFinding(text, start, end) {
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) return;
    const points = Array.from(text);
    // The API uses Unicode code-point offsets; textarea ranges use UTF-16 offsets.
    const from = points.slice(0, start).join("").length;
    const to = points.slice(0, end).join("").length;
    source.focus();
    source.setSelectionRange(from, to);
    announce(`已在原文中选中第 ${start + 1} 至 ${end} 个字符。`);
  }

  function renderAnalysis(result, input) {
    if (!result || !Array.isArray(result.findings) || !Number.isFinite(result.character_count)) {
      throw new Error("分析结果不完整，请检查服务版本。");
    }
    const findings = result.findings;
    byId("analysis-empty").hidden = true;
    byId("analysis-result").hidden = false;
    byId("analysis-badge").textContent = "分析完成";
    byId("metric-characters").textContent = formatCount(result.character_count);
    byId("metric-findings").textContent = formatCount(findings.length);
    const profile = typeof result.profile === "string" ? result.profile : result.profile?.name;
    byId("metric-profile").textContent = profile || "默认配置";
    byId("analysis-summary").textContent = findings.length
      ? `找到 ${findings.length} 处可留意的表达。它们可以合理使用；结合上下文决定是否调整，点击提示可定位原文。`
      : "当前配置没有找到需要提示的表达。这不代表文本没有其他可改进之处。";
    byId("analysis-disclaimer").textContent = typeof result.disclaimer === "string"
      ? result.disclaimer : "这些提示是可解释的文风观察，不代表文本质量或作者身份。";
    const list = byId("findings-list");
    list.replaceChildren();
    if (!findings.length) {
      const notice = document.createElement("p");
      notice.className = "no-findings";
      notice.textContent = "没有规则命中。清楚、准确、有自己的语气，比少用某几个词更重要。";
      list.append(notice);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const finding of findings.slice(0, 100)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "finding";
      const heading = document.createElement("span");
      heading.className = "finding-heading";
      const label = document.createElement("span");
      label.className = "finding-label";
      label.textContent = "表达提示";
      const location = document.createElement("span");
      location.className = "finding-location";
      location.textContent = Number.isInteger(finding.start) ? `第 ${finding.start + 1} 字符 ↗` : "查看原文 ↗";
      heading.append(label, location);
      const quote = document.createElement("span");
      quote.className = "finding-text";
      quote.textContent = typeof finding.text === "string" ? finding.text : "查看匹配表达";
      const message = document.createElement("span");
      message.className = "finding-message";
      message.textContent = typeof finding.message === "string" ? finding.message : "请结合语境判断是否需要调整。";
      button.append(heading, quote, message);
      button.addEventListener("click", () => selectFinding(input, finding.start, finding.end));
      fragment.append(button);
    }
    if (findings.length > 100) {
      const notice = document.createElement("p");
      notice.className = "analysis-disclaimer";
      notice.textContent = `为便于阅读，这里显示前 100 处提示，共 ${findings.length} 处。`;
      fragment.append(notice);
    }
    list.append(fragment);
  }

  const describeMode = (mode) => ({
    prompt: "提示词控制",
    "prompt-only": "提示词控制",
    prompt_only: "提示词控制",
    phrase: "短语上下文概率控制",
    "phrase-logits": "短语上下文概率控制",
    phrase_logits: "短语上下文概率控制",
    "static-token-bias": "静态 token 偏置",
    static_token_bias: "静态 token 偏置",
  }[mode] || (typeof mode === "string" ? mode : "提示词控制"));

  function renderRewrite(result, input) {
    if (!result || typeof result.text !== "string" || result.preservation?.passed !== true) {
      throw new Error("改写结果未通过保护检查，已停止展示。请检查模型输出后重试。");
    }
    outputText = result.text;
    byId("rewrite-empty").hidden = true;
    byId("rewrite-result").hidden = false;
    byId("copy-output").hidden = false;
    byId("copy-output").textContent = "复制改写";
    byId("comparison-source").textContent = input;
    byId("comparison-output").textContent = outputText;
    byId("rewrite-model").textContent = `${result.model || "已配置模型"} · ${describeMode(result.steering_mode)}`;
    const metrics = byId("comparison-metrics");
    metrics.replaceChildren();
    for (const text of [
      `字符：${formatCount(characterCount(input))} → ${formatCount(characterCount(outputText))}`,
      Array.isArray(result.analysis_before?.findings) && Array.isArray(result.analysis_after?.findings)
        ? `表达提示：${result.analysis_before.findings.length} → ${result.analysis_after.findings.length}（仅作观察）` : null,
    ]) {
      if (!text) continue;
      const item = document.createElement("span");
      item.textContent = text;
      metrics.append(item);
    }
    if (result.analysis_before) renderAnalysis(result.analysis_before, input);
  }

  async function run(kind) {
    const input = source.value;
    if (!input.trim() || characterCount(input) > maxCharacters || busy) return;
    if (kind === "rewrite" && !configuration?.configured) return;
    setError();
    const runRevision = revision;
    setBusy(true, kind);
    announce(kind === "analyze" ? "正在分析原文。" : "正在请求生成模型并检查事实保护内容。");
    try {
      const result = await request(`/api/${kind}`, { text: input }, kind === "rewrite" ? 180000 : 15000);
      if (revision !== runRevision) return;
      if (kind === "analyze") renderAnalysis(result, input);
      else renderRewrite(result, input);
      announce(kind === "analyze" ? "分析完成，可以点击表达提示定位原文。" : "改写完成，保护检查通过。请阅读前后对照。");
    } catch (error) {
      if (error.name !== "AbortError" && revision === runRevision) {
        setError(error.message || "请求未完成，请稍后重试。");
        announce("请求未完成。");
      }
    } finally {
      if (revision === runRevision) setBusy(false);
    }
  }

  source.addEventListener("input", () => {
    revision += 1;
    pendingController?.abort();
    setBusy(false);
    resetResults();
    setError();
    updateButtons();
  });
  source.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      if (!analyzeButton.disabled) void run("analyze");
    }
  });
  byId("load-example").addEventListener("click", () => {
    source.value = example;
    source.dispatchEvent(new Event("input"));
    source.focus();
    announce("已载入示例。这段文字用于演示分析，不是模型生成结果。");
  });
  byId("clear-source").addEventListener("click", () => {
    source.value = "";
    source.dispatchEvent(new Event("input"));
    source.focus();
    announce("已清空原文。");
  });
  analyzeButton.addEventListener("click", () => { void run("analyze"); });
  rewriteButton.addEventListener("click", () => { void run("rewrite"); });
  byId("copy-output").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(outputText);
      byId("copy-output").textContent = "已复制 ✓";
      announce("改写结果已复制。");
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(byId("comparison-output"));
      selection.removeAllRanges();
      selection.addRange(range);
      announce("浏览器无法自动复制，已选中改写结果，请手动复制。");
      byId("copy-output").textContent = "已选中，请手动复制";
    }
  });

  async function loadConfiguration() {
    try {
      configuration = await request("/api/config");
      const configured = configuration?.configured === true;
      configuration.configured = configured;
      byId("connection-status").classList.toggle("is-offline", !configured);
      byId("connection-status").lastChild.textContent = configured ? " 生成后端已接入" : " 本地分析可用";
      byId("backend-note").textContent = configured
        ? `生成后端：${configuration.model || configuration.backend || "已配置"} · ${describeMode(configuration.steering_mode)}。点击改写会将原文发送给该服务。`
        : "生成后端尚未配置，文风分析可直接使用。请按 README 在服务环境中设置模型与凭据后重启。";
    } catch {
      configuration = { configured: false };
      byId("connection-status").classList.add("is-offline");
      byId("connection-status").lastChild.textContent = " 配置读取失败";
      byId("backend-note").textContent = "未能读取生成配置。可以尝试分析，或检查本地服务后刷新页面。";
    }
    updateButtons();
  }

  updateButtons();
  void loadConfiguration();
})();
