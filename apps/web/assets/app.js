/*
 * Character Voice AI — the web app.
 *
 * Three independent pieces of state, each rendered on its own:
 *
 *   A. connection   — connecting | connected | degraded | disconnected   (top bar)
 *   B. microphone   — off | pending | on        (status chip; tracks really stopped)
 *   C. conversation — VoiceState below, the single source of truth for what the
 *                     app is doing (status bar, mic button, character presence)
 *
 * Server messages and local audio events are *mapped* onto VoiceState; nothing else
 * decides what the status bar says.
 */
(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // C. conversation state machine
  // ---------------------------------------------------------------------------

  const VoiceState = Object.freeze({
    IDLE: "idle",
    REQUESTING_MIC_PERMISSION: "requesting_mic_permission",
    LISTENING: "listening",
    SPEECH_DETECTED: "speech_detected",
    TRANSCRIBING: "transcribing",
    THINKING: "thinking",
    GENERATING_SPEECH: "generating_speech",
    SPEAKING: "speaking",
    ERROR: "error",
  });
  const S = VoiceState;

  // Where each state may go next. A move outside this table is logged (it means an
  // event arrived that this mapping did not expect) but still applied, so the UI can
  // never get stuck behind its own bookkeeping.
  const TRANSITIONS = {
    [S.IDLE]: [S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING, S.ERROR],
    [S.REQUESTING_MIC_PERMISSION]: [S.LISTENING, S.IDLE, S.ERROR],
    [S.LISTENING]: [S.SPEECH_DETECTED, S.TRANSCRIBING, S.IDLE, S.ERROR, S.THINKING],
    [S.SPEECH_DETECTED]: [S.TRANSCRIBING, S.IDLE, S.ERROR, S.THINKING],
    [S.TRANSCRIBING]: [S.THINKING, S.IDLE, S.ERROR],
    [S.THINKING]: [S.GENERATING_SPEECH, S.SPEAKING, S.IDLE, S.ERROR,
                   S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
    [S.GENERATING_SPEECH]: [S.SPEAKING, S.IDLE, S.ERROR,
                            S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
    [S.SPEAKING]: [S.GENERATING_SPEECH, S.IDLE, S.ERROR,
                   S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
    [S.ERROR]: [S.IDLE, S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
  };

  const app = {
    state: S.IDLE,
    conn: "connecting",          // A
    mic: "off",                  // B
    llmOk: true,
    session: null,               // { id, characterId, name, engine }
    model: "",
    socket: null,
    hasTalked: false,
    error: null,                 // { text, retry, raw }
    lastUserText: "",
    turn: null,                  // timing + bookkeeping for the turn in progress
    turns: [],                   // finished turn timings (developer info)
    staleTurnIds: new Set(),     // server turns we superseded or stopped
    acceptAudio: false,          // whether the binary frames that follow belong to us
    settings: loadSettings(),
  };

  // Exposed for tests and debugging: every transition is recorded here.
  window.__voice = { log: [], states: VoiceState, app };

  function setState(next, why) {
    const prev = app.state;
    if (prev === next) return;
    if (!(TRANSITIONS[prev] || []).includes(next)) {
      console.warn(`[voice] unexpected ${prev} → ${next} (${why || ""})`);
    }
    app.state = next;
    window.__voice.log.push({ from: prev, to: next, why: why || "", t: Math.round(performance.now()) });
    render();
  }

  // ---------------------------------------------------------------------------
  // DOM
  // ---------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  const root = $("app");
  const els = {
    connText: $("conn-text"), reconnect: $("reconnect"),
    name: $("character-name"), sub: $("character-sub"), avatarText: $("avatar-text"),
    presenceText: $("presence-text"),
    conversation: $("conversation"), messages: $("messages"), empty: $("empty"),
    statusIcon: $("status-icon"), statusLabel: $("status-label"), statusHint: $("status-hint"),
    statusAction: $("status-action"), meter: $("meter"), micChipText: $("mic-chip-text"),
    alert: $("alert"), alertText: $("alert-text"), alertRetry: $("alert-retry"),
    input: $("input"), send: $("send"), mic: $("mic"), micLabel: $("mic-label"),
    boot: $("boot"), bootText: $("boot-text"), bootRetry: $("boot-retry"), bootSpinner: $("boot-spinner"),
    settings: $("settings"), setCharacter: $("set-character"), setAutoplay: $("set-autoplay"),
    setVolume: $("set-volume"), setMic: $("set-mic"),
    devInfo: $("dev-info"), devTurns: $("dev-turns"), devLastError: $("dev-last-error"),
  };
  const meterBars = Array.from(els.meter.children);

  const ICONS = {
    dot: '<span class="idle-dot"></span>',
    rec: '<span class="rec-dot"></span>',
    spin: '<span class="spinner"></span>',
    think: '<svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8Z"/></svg>',
    speaker: '<svg viewBox="0 0 24 24"><path d="M11 5 6 9H3v6h3l5 4Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>',
    play: '<svg viewBox="0 0 24 24"><path d="M11 5 6 9H3v6h3l5 4Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/></svg>',
    mic: '<svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
  };

  function charName() { return (app.session && app.session.name) || "她"; }

  // What the status bar says for each state: [icon, label, hint, action].
  function statusView() {
    const name = charName();
    switch (app.state) {
      case S.REQUESTING_MIC_PERMISSION:
        return ["spin", "正在请求麦克风权限…", "请在浏览器提示中允许使用麦克风", null];
      case S.LISTENING:
        return ["rec", "正在聆听", "说完后点击「完成」", null];
      case S.SPEECH_DETECTED:
        return ["rec", "正在聆听", "听到你的声音了 · 说完点击「完成」", null];
      case S.TRANSCRIBING:
        return ["spin", "正在识别语音…", "麦克风已关闭", null];
      case S.THINKING:
        return ["think", "正在思考…", `${name}正在组织回复`, "stop"];
      case S.GENERATING_SPEECH:
        return ["spin", "正在生成语音…", "文字已就绪，正在合成声音", "stop"];
      case S.SPEAKING:
        return ["speaker", `${name}正在说话`, "点击麦克风可以打断她", "stop"];
      case S.ERROR:
        return ["alert", "出现问题", "", null];      // details + retry are in the alert below
      default:
        return ["dot", "准备就绪", app.hasTalked ? "点击麦克风继续对话" : "点击麦克风开始说话", null];
    }
  }

  const PRESENCE = {
    [S.IDLE]: "在线", [S.REQUESTING_MIC_PERMISSION]: "在线", [S.ERROR]: "在线",
    [S.LISTENING]: "正在听你说", [S.SPEECH_DETECTED]: "正在听你说",
    [S.TRANSCRIBING]: "正在听清你说的话", [S.THINKING]: "正在思考",
    [S.GENERATING_SPEECH]: "准备开口", [S.SPEAKING]: "正在说话",
  };

  function render() {
    root.dataset.state = app.state;
    root.dataset.conn = app.conn;
    root.dataset.mic = app.mic;

    // A. connection
    const connText = {
      connecting: "正在连接…", connected: "AI 已连接",
      degraded: "AI 未就绪", disconnected: "AI 未连接",
    }[app.conn];
    els.connText.textContent = connText;
    els.reconnect.hidden = app.conn !== "disconnected";

    // B. microphone
    els.micChipText.textContent =
      app.mic === "on" ? "麦克风开启" :
      app.mic === "pending" ? "等待授权" : "麦克风已关闭";

    // C. conversation
    const [icon, label, hint, action] = statusView();
    if (els.statusIcon.dataset.icon !== icon) {
      els.statusIcon.innerHTML = ICONS[icon];
      els.statusIcon.dataset.icon = icon;
    }
    els.statusLabel.textContent = label;
    els.statusHint.textContent = hint;
    els.statusAction.hidden = action !== "stop";
    if (action === "stop") {
      els.statusAction.textContent = "停止";
      els.statusAction.setAttribute("aria-label", "停止回复");
    }
    els.presenceText.textContent = PRESENCE[app.state] || "在线";

    // Errors live next to the composer with a way forward.
    if (app.error) {
      els.alert.hidden = false;
      els.alertText.textContent = app.error.text;
      els.alertRetry.hidden = !app.error.retry;
      els.alertRetry.textContent = app.error.retryLabel || "重试";
    } else {
      els.alert.hidden = true;
    }

    // Composer: send when there is text, microphone otherwise.
    const typed = els.input.value.trim().length > 0;
    const listening = app.state === S.LISTENING || app.state === S.SPEECH_DETECTED;
    const ready = app.conn === "connected" || app.conn === "degraded";
    els.send.hidden = !typed || listening;
    els.mic.hidden = typed && !listening;
    els.send.disabled = !ready;
    els.mic.disabled = !ready || app.state === S.REQUESTING_MIC_PERMISSION || app.state === S.TRANSCRIBING;
    els.mic.setAttribute("aria-pressed", listening ? "true" : "false");
    els.mic.setAttribute("aria-label",
      listening ? "完成说话，开始识别" :
      [S.THINKING, S.GENERATING_SPEECH, S.SPEAKING].includes(app.state) ? "打断并开始说话" :
      "开始说话");
    els.micLabel.textContent = "完成";

    if (!listening) resetMeter();
    renderDev();
  }

  // ---------------------------------------------------------------------------
  // messages
  // ---------------------------------------------------------------------------

  function scrollToEnd() {
    els.conversation.scrollTop = els.conversation.scrollHeight;
  }

  function addMessage(role, text, opts) {
    els.empty.hidden = true;
    const li = document.createElement("li");
    li.className = "msg " + (role === "user" ? "msg-user" : role === "note" ? "msg-note" : "msg-char");
    if (role === "note") {
      li.textContent = text;
      els.messages.appendChild(li);
      scrollToEnd();
      return { li };
    }
    const body = document.createElement("div");
    body.className = "msg-body";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    body.appendChild(bubble);
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    body.appendChild(meta);

    if (role === "char") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = charName().slice(0, 1);
      li.appendChild(avatar);
      const label = document.createElement("span");
      label.className = "visually-hidden";
      label.textContent = charName() + "：";
      bubble.prepend(label);
    } else {
      meta.style.justifyContent = "flex-end";
      const label = document.createElement("span");
      label.className = "visually-hidden";
      label.textContent = "你：";
      bubble.prepend(label);
      if (opts && opts.voice) {
        meta.innerHTML = '<span class="voice-tag">' + ICONS.mic + "语音输入</span>";
      }
    }
    li.appendChild(body);
    els.messages.appendChild(li);
    scrollToEnd();
    return { li, bubble, meta, text: "", clips: [] };
  }

  function setBubbleText(message, text) {
    message.text = text;
    const label = message.bubble.querySelector(".visually-hidden");
    message.bubble.textContent = text;
    if (label) message.bubble.prepend(label);
    scrollToEnd();
  }

  function showTyping(message) {
    message.bubble.innerHTML = '<span class="typing" aria-label="正在输入"><i></i><i></i><i></i></span>';
  }

  function addReplay(message) {
    if (message.replay || !message.clips.length) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "replay";
    button.innerHTML = ICONS.play + "<span>播放语音</span>";
    button.setAttribute("aria-label", "播放这条语音");
    button.addEventListener("click", () => {
      stopPlayback();
      message.clips.forEach((blob) => player.enqueue(blob, null));
    });
    message.meta.appendChild(button);
    message.replay = button;
  }

  // ---------------------------------------------------------------------------
  // audio playback
  // ---------------------------------------------------------------------------

  const player = {
    queue: [],
    current: null,
    enqueue(blob, turn) {
      const audio = new Audio(URL.createObjectURL(blob));
      audio.volume = app.settings.volume;
      audio.addEventListener("playing", () => {
        if (turn && !turn.playStart) turn.playStart = performance.now();
        if (app.state !== S.SPEAKING && !isListening()) setState(S.SPEAKING, "audio playing");
      });
      audio.addEventListener("ended", () => { this.current = null; this.next(); });
      audio.addEventListener("error", () => { this.current = null; this.next(); });
      this.queue.push(audio);
      if (!this.current) this.next();
    },
    next() {
      if (this.current) return;
      const audio = this.queue.shift();
      if (!audio) { onPlaybackDrained(); return; }
      this.current = audio;
      audio.play().catch(() => { this.current = null; this.next(); });
    },
    stop() {
      this.queue = [];
      if (this.current) { this.current.pause(); this.current = null; }
    },
    get busy() { return !!this.current || this.queue.length > 0; },
  };

  function stopPlayback() { player.stop(); }

  function onPlaybackDrained() {
    const turn = app.turn;
    if (isListening() || app.state === S.ERROR) return;
    if (turn && !turn.ended) {
      // Played what we have; the next chunk is still being synthesized.
      if (app.state === S.SPEAKING) setState(S.GENERATING_SPEECH, "waiting for next chunk");
      return;
    }
    if (app.state === S.SPEAKING) setState(S.IDLE, "playback finished");
  }

  function isListening() {
    return app.state === S.LISTENING || app.state === S.SPEECH_DETECTED ||
           app.state === S.REQUESTING_MIC_PERMISSION;
  }

  // ---------------------------------------------------------------------------
  // B. microphone
  // ---------------------------------------------------------------------------

  const MIC_RATE = 16000;
  const CAPTURE_WORKLET = `
    class Capture extends AudioWorkletProcessor {
      constructor() { super(); this.held = []; this.count = 0; }
      process(inputs) {
        const channel = inputs[0] && inputs[0][0];
        if (!channel) return true;
        this.held.push(new Float32Array(channel));
        this.count += channel.length;
        if (this.count >= 1024) {
          const out = new Float32Array(this.count);
          let at = 0;
          for (const block of this.held) { out.set(block, at); at += block.length; }
          this.held = []; this.count = 0;
          this.port.postMessage(out, [out.buffer]);
        }
        return true;
      }
    }
    registerProcessor("capture", Capture);
  `;
  const mic = { stream: null, context: null, node: null, analyser: null, raf: 0 };

  function resample(samples, from, to) {
    if (from === to) return samples;
    const ratio = from / to;
    const out = new Float32Array(Math.floor(samples.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const start = Math.floor(i * ratio);
      const stop = Math.min(Math.floor((i + 1) * ratio), samples.length);
      let total = 0;
      for (let n = start; n < stop; n++) total += samples[n];
      out[i] = stop > start ? total / (stop - start) : 0;
    }
    return out;
  }

  function sendPcm(samples, rate) {
    const socket = app.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const data = resample(samples, rate, MIC_RATE);
    const pcm = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) {
      pcm[i] = Math.round(Math.max(-1, Math.min(1, data[i])) * 32767);
    }
    socket.send(pcm.buffer);
  }

  async function startListening() {
    if (!app.socket || app.socket.readyState !== WebSocket.OPEN) return;
    clearError();
    // Speaking over her is allowed: stop the reply first, then listen.
    if ([S.THINKING, S.GENERATING_SPEECH, S.SPEAKING].includes(app.state)) interrupt();

    app.mic = "pending";
    setState(S.REQUESTING_MIC_PERMISSION, "mic button");
    try {
      const audio = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      if (app.settings.micId) audio.deviceId = { exact: app.settings.micId };
      mic.stream = await navigator.mediaDevices.getUserMedia({ audio });
      mic.context = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: MIC_RATE });
      await mic.context.audioWorklet.addModule(
        URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" })));
      const source = mic.context.createMediaStreamSource(mic.stream);
      mic.node = new AudioWorkletNode(mic.context, "capture");
      mic.node.port.onmessage = (event) => sendPcm(event.data, mic.context.sampleRate);
      source.connect(mic.node);
      const sink = mic.context.createGain();
      sink.gain.value = 0;
      mic.node.connect(sink).connect(mic.context.destination);
      mic.analyser = mic.context.createAnalyser();
      mic.analyser.fftSize = 512;
      source.connect(mic.analyser);
    } catch (error) {
      await releaseMic();
      const denied = error && (error.name === "NotAllowedError" || error.name === "SecurityError");
      const missing = error && (error.name === "NotFoundError" || error.name === "OverconstrainedError");
      showError(
        denied ? "无法访问麦克风，请检查浏览器权限。" :
        missing ? "没有找到可用的麦克风设备。" : "麦克风启动失败，请重试。",
        { retry: startListening, raw: error && (error.name + ": " + error.message) });
      return;
    }

    // "Turn" mode: the server closes the window after one utterance, and so do we.
    app.socket.send(JSON.stringify({ type: "user_audio_begin", sample_rate: MIC_RATE, mode: "turn" }));
    app.mic = "on";
    app.hasTalked = true;
    beginTurn("voice");
    app.turn.recStart = performance.now();
    setState(S.LISTENING, "microphone open");
    refreshMicDevices();
    drawMeter();
  }

  // Stops capture for real — tracks stopped, context closed — not just the icon.
  // Everything that ends capture happens synchronously, and the UI says so at once;
  // only closing the audio context (which captures nothing by then) is awaited.
  async function releaseMic() {
    cancelAnimationFrame(mic.raf);
    if (mic.node) { mic.node.port.onmessage = null; mic.node.disconnect(); mic.node = null; }
    if (mic.stream) { mic.stream.getTracks().forEach((track) => track.stop()); mic.stream = null; }
    mic.analyser = null;
    const ctx = mic.context;
    mic.context = null;
    if (app.mic !== "off") { app.mic = "off"; render(); }
    if (ctx) await ctx.close().catch(() => {});
  }

  // The user pressed 完成.
  async function finishListening() {
    if (!isListening()) return;
    if (app.turn) app.turn.recEnd = performance.now();
    if (app.socket && app.socket.readyState === WebSocket.OPEN) {
      app.socket.send(JSON.stringify({ type: "user_audio_end" }));
    }
    await releaseMic();
    // Immediately stop looking like we are listening; the server confirms with
    // "transcribing", or with "no_speech" if there was nothing to recognise.
    setState(S.TRANSCRIBING, "user pressed done");
  }

  function drawMeter() {
    if (!mic.analyser) return;
    const data = new Uint8Array(mic.analyser.fftSize);
    const smooth = new Array(meterBars.length).fill(0);
    const tick = () => {
      if (!mic.analyser) return;
      mic.analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) { const v = (data[i] - 128) / 128; sum += v * v; }
      // Decibels, not raw amplitude: speech sits around -40…-15 dBFS, which a linear
      // scale would draw as a barely moving line.
      const db = 20 * Math.log10(Math.sqrt(sum / data.length) + 1e-6);
      const level = Math.max(0, Math.min(1, (db + 55) / 40));
      meterBars.forEach((bar, i) => {
        const shape = 0.55 + 0.45 * Math.sin((i / (meterBars.length - 1)) * Math.PI);
        const target = level * shape * (0.75 + Math.random() * 0.25);
        smooth[i] = smooth[i] * 0.55 + target * 0.45;
        bar.style.height = Math.max(4, Math.round(4 + smooth[i] * 18)) + "px";
      });
      mic.raf = requestAnimationFrame(tick);
    };
    tick();
  }

  function resetMeter() {
    meterBars.forEach((bar) => { bar.style.height = ""; });
  }

  async function refreshMicDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
      const select = els.setMic;
      const current = app.settings.micId;
      select.innerHTML = '<option value="">系统默认</option>';
      devices.forEach((device, index) => {
        if (!device.deviceId || device.deviceId === "default") return;
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `麦克风 ${index + 1}`;
        select.appendChild(option);
      });
      select.value = current || "";
    } catch (e) { /* listing devices is optional */ }
  }

  // ---------------------------------------------------------------------------
  // turns
  // ---------------------------------------------------------------------------

  function beginTurn(kind) {
    finishTurnTiming();
    retireTurn();
    app.turn = { kind, start: performance.now(), message: null, ended: false, serverId: null };
  }

  // Late messages from a turn we have moved on from must not touch the new one.
  function retireTurn() {
    if (app.turn) {
      app.turn.ended = true;
      if (app.turn.serverId) app.staleTurnIds.add(app.turn.serverId);
    }
    app.acceptAudio = false;
  }

  // Does a message with this server turn id belong to the turn on screen?
  function acceptTurnId(id) {
    if (!id || id === "pending") return true;
    if (app.staleTurnIds.has(id)) return false;
    const turn = app.turn;
    if (!turn) return false;
    if (!turn.serverId) { turn.serverId = id; return true; }
    return turn.serverId === id;
  }

  function finishTurnTiming() {
    const t = app.turn;
    if (!t || t.recorded) return;
    t.recorded = true;
    const s = (a, b) => (a && b ? ((b - a) / 1000).toFixed(2) : "—");
    const llmFrom = t.transcribedAt || t.sentAt;
    app.turns.unshift({
      kind: t.kind === "voice" ? "语音" : "文字",
      rec: s(t.recStart, t.recEnd),
      stt: s(t.recEnd, t.transcribedAt),
      text: s(llmFrom, t.firstTextAt),
      audio: s(t.firstTextAt, t.firstAudioAt),
      total: s(t.recStart || t.sentAt, t.playStart || t.firstTextAt),
    });
    app.turns = app.turns.slice(0, 8);
  }

  function sendText(text) {
    if (!text || !app.socket || app.socket.readyState !== WebSocket.OPEN) return;
    clearError();
    if (isListening()) { releaseMic(); }
    stopPlayback();
    beginTurn("text");
    app.turn.sentAt = performance.now();
    app.hasTalked = true;
    app.lastUserText = text;
    addMessage("user", text);
    app.socket.send(JSON.stringify({ type: "user_text", text }));
    setState(S.THINKING, "text sent");
  }

  function interrupt() {
    stopPlayback();
    if (app.socket && app.socket.readyState === WebSocket.OPEN) {
      app.socket.send(JSON.stringify({ type: "interrupt" }));
    }
    const turn = app.turn;
    if (turn) {
      if (turn.message) {
        if (!turn.message.text) turn.message.li.remove(); else addReplay(turn.message);
      }
      retireTurn();
    }
    if ([S.THINKING, S.GENERATING_SPEECH, S.SPEAKING].includes(app.state)) {
      setState(S.IDLE, "stopped by user");
    }
  }

  // ---------------------------------------------------------------------------
  // errors
  // ---------------------------------------------------------------------------

  function showError(text, opts) {
    opts = opts || {};
    app.error = { text, retry: opts.retry || null, retryLabel: opts.retryLabel, raw: opts.raw || "" };
    if (opts.raw) els.devLastError.textContent = "最近错误：" + opts.raw;
    stopPlayback();
    if (app.turn) app.turn.ended = true;
    setState(S.ERROR, text);
    render();
  }

  function clearError() {
    if (!app.error) return;
    app.error = null;
    if (app.state === S.ERROR) setState(S.IDLE, "error dismissed");
    render();
  }

  // Server error → words a user can act on. The raw text goes to developer info.
  function friendlyError(message) {
    const raw = message.message || "";
    const code = message.code || "";
    const lower = raw.toLowerCase();
    const resend = app.lastUserText ? () => sendText(app.lastUserText) : null;
    if (code === "no_speech") {
      return [raw || "没有听清，请再说一次", startListening, "再说一次"];
    }
    if (code === "no_stt" || app.state === S.TRANSCRIBING) {
      return ["语音识别失败，请重试", startListening, "重新说"];
    }
    if (lower.includes("ollama") || lower.includes("llm") || lower.includes("qwen") ||
        app.state === S.THINKING) {
      return ["本地 Qwen 服务未响应", resend, "重试"];
    }
    if (lower.includes("sovits") || lower.includes("sidecar") || lower.includes("synthes") ||
        lower.includes("tts") || app.state === S.GENERATING_SPEECH) {
      return ["语音生成失败", resend, "重试"];
    }
    return ["出了点问题，请重试", resend, "重试"];
  }

  // ---------------------------------------------------------------------------
  // server messages → state
  // ---------------------------------------------------------------------------

  function currentCharMessage(turnId) {
    const turn = app.turn;
    if (!turn) return null;
    if (!turn.message) {
      turn.message = addMessage("char", "");
      showTyping(turn.message);
    }
    if (turnId) turn.message.turnId = turnId;
    return turn.message;
  }

  function onServerMessage(event) {
    if (event.data instanceof Blob) {
      const turn = app.turn;
      if (!app.acceptAudio || !turn || turn.ended) return;   // a reply the user stopped
      if (!turn.firstAudioAt) turn.firstAudioAt = performance.now();
      const message = currentCharMessage();
      message.clips.push(event.data);
      if (app.settings.autoplay && !isListening()) player.enqueue(event.data, turn);
      return;
    }
    let message;
    try { message = JSON.parse(event.data); } catch (e) { return; }
    // A transcript belongs to the listening window, not to a reply turn: its id is
    // whatever turn the server last ran, so it is never filtered.
    if (message.type !== "transcript" && message.turn_id !== undefined &&
        !acceptTurnId(message.turn_id)) {
      if (message.type === "audio_begin") app.acceptAudio = false;
      return;
    }
    const turn = app.turn;

    switch (message.type) {
      case "audio_begin":
        app.acceptAudio = !!(turn && !turn.ended);
        break;

      case "vad":
        if (message.event === "speech_start" && app.state === S.LISTENING) {
          setState(S.SPEECH_DETECTED, "server heard speech");
        }
        break;

      case "state":
        onServerState(message.state);
        break;

      case "transcript":
        if (turn) turn.transcribedAt = performance.now();
        app.lastUserText = message.text;
        addMessage("user", message.text, { voice: true });
        if (app.state === S.TRANSCRIBING) setState(S.THINKING, "transcript received");
        currentCharMessage();
        break;

      case "character_text": {
        if (!turn || turn.ended) break;
        if (!turn.firstTextAt) turn.firstTextAt = performance.now();
        const msg = currentCharMessage(message.turn_id);
        setBubbleText(msg, msg.text + message.text);
        break;
      }

      case "turn_end":
        if (!turn) break;
        turn.ended = true;
        if (turn.message) {
          if (!turn.message.text) turn.message.li.remove(); else addReplay(turn.message);
        }
        if (message.was_interrupted && turn.message && turn.message.text) {
          addMessage("note", "（已打断）");
        }
        finishTurnTiming();
        if (!player.busy && !isListening() && app.state !== S.ERROR) setState(S.IDLE, "turn ended");
        break;

      case "error": {
        const [text, retry, retryLabel] = friendlyError(message);
        if (turn && turn.message && !turn.message.text) turn.message.li.remove();
        releaseMic();
        showError(text, { retry, retryLabel, raw: `${message.code || ""} ${message.message || ""}` });
        break;
      }
    }
  }

  function onServerState(state) {
    switch (state) {
      case "transcribing":
        // The server has the whole utterance (silence ended it, or we pressed 完成).
        // Stop capturing now, so the microphone is never on when we are not listening.
        if (isListening()) {
          if (app.turn && !app.turn.recEnd) app.turn.recEnd = performance.now();
          releaseMic();
        }
        if (app.state !== S.TRANSCRIBING) setState(S.TRANSCRIBING, "server transcribing");
        break;
      case "thinking":
      case "planning":
        if (!app.turn || app.turn.ended) break;
        if (!isListening() && app.state !== S.THINKING && app.state !== S.ERROR) {
          setState(S.THINKING, "server " + state);
        }
        currentCharMessage();
        break;
      case "synthesizing":
        if (!app.turn || app.turn.ended) break;
        if (!isListening() && app.state !== S.SPEAKING && app.state !== S.ERROR) {
          setState(S.GENERATING_SPEECH, "server synthesizing");
        }
        break;
      case "interrupted":
        stopPlayback();
        break;
      // "speaking" means audio is on its way; SPEAKING starts when playback does.
      // "idle" means the server is done; we go idle when playback is done.
    }
  }

  // ---------------------------------------------------------------------------
  // A. connection / session
  // ---------------------------------------------------------------------------

  async function api(path, options) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = response.statusText;
      try { const body = await response.json(); detail = body.detail || detail; } catch (e) {}
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return response.json();
  }

  function prettyModel(model) {
    // "qwen3:4b" → "Qwen3 4B"
    if (!model) return "";
    const [family, size] = model.split(":");
    const name = family.charAt(0).toUpperCase() + family.slice(1);
    return size ? `${name} ${size.toUpperCase()}` : name;
  }

  async function checkLlm() {
    try {
      const health = await api("/api/llm/health");
      app.llmOk = !!health.ok;
      app.model = health.model || app.model;
      app.llmError = health.ok ? "" : health.error || "";
    } catch (e) {
      app.llmOk = false;
    }
    if (app.socket && app.socket.readyState === WebSocket.OPEN) {
      app.conn = app.llmOk ? "connected" : "degraded";
    }
    updateCharacterHeader();
    render();
  }

  function updateCharacterHeader() {
    const name = (app.session && app.session.name) || "…";
    els.name.textContent = name;
    els.avatarText.textContent = name.slice(0, 1);
    document.querySelectorAll(".js-name").forEach((node) => { node.textContent = name; });
    const model = prettyModel(app.model);
    els.sub.textContent = app.conn === "degraded"
      ? "本地 AI 暂未就绪"
      : "本地角色语音" + (model ? " · " + model : "");
    document.title = name + " · Character Voice AI";
  }

  async function boot() {
    els.bootSpinner.hidden = false;
    els.bootRetry.hidden = true;
    els.bootText.textContent = "正在连接本地 AI…";
    try {
      const [health, characters] = await Promise.all([api("/health"), api("/characters")]);
      fillCharacterPicker(characters, health.default_character);
      await openSession(app.settings.characterId || health.default_character);
      els.boot.classList.add("done");
      setTimeout(() => { els.boot.hidden = true; }, 300);
    } catch (error) {
      els.bootSpinner.hidden = true;
      els.bootText.textContent = "无法连接本地 AI。请确认后端服务已启动。";
      els.bootRetry.hidden = false;
      els.devLastError.textContent = "最近错误：" + error.message;
    }
  }

  function fillCharacterPicker(characters, fallback) {
    const details = characters.details && characters.details.length
      ? characters.details
      : (characters.characters || []).map((id) => ({ id, name: id }));
    els.setCharacter.innerHTML = "";
    details.forEach((c) => {
      const option = document.createElement("option");
      option.value = c.id;
      option.textContent = c.name === c.id ? c.id : `${c.name}`;
      els.setCharacter.appendChild(option);
    });
    const known = details.some((c) => c.id === app.settings.characterId);
    els.setCharacter.value = known ? app.settings.characterId : fallback;
    if (!known) app.settings.characterId = fallback;
  }

  async function openSession(characterId) {
    app.conn = "connecting";
    render();
    const data = await api("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ character_id: characterId || null }),
    });
    app.session = { id: data.session_id, characterId: data.character_id, name: data.character_name, engine: data.engine };
    updateCharacterHeader();
    await connectSocket();
    await checkLlm();
  }

  function connectSocket() {
    return new Promise((resolve, reject) => {
      const url = location.origin.replace(/^http/, "ws") + "/sessions/" + app.session.id + "/ws";
      const socket = new WebSocket(url);
      socket.binaryType = "blob";
      let opened = false;
      socket.onopen = () => {
        opened = true;
        app.socket = socket;
        app.conn = app.llmOk ? "connected" : "degraded";
        render();
        resolve();
      };
      socket.onmessage = onServerMessage;
      socket.onclose = () => {
        if (app.socket !== socket) return;      // a socket we replaced on purpose
        app.socket = null;
        app.conn = "disconnected";
        releaseMic();
        stopPlayback();
        if (opened) showError("连接已断开", { retry: reconnect, retryLabel: "重新连接" });
        render();
        if (!opened) reject(new Error("WebSocket failed"));
      };
    });
  }

  async function closeSession() {
    await releaseMic();
    stopPlayback();
    const socket = app.socket;
    app.socket = null;
    if (socket) { try { socket.send(JSON.stringify({ type: "bye" })); } catch (e) {} socket.close(); }
    if (app.session) { fetch("/sessions/" + app.session.id, { method: "DELETE" }).catch(() => {}); }
  }

  async function reconnect() {
    clearError();
    try {
      await closeSession();
      await openSession(app.settings.characterId);
    } catch (error) {
      app.conn = "disconnected";
      showError("无法连接本地 AI", { retry: reconnect, retryLabel: "重新连接", raw: error.message });
    }
  }

  async function newConversation() {
    els.messages.innerHTML = "";
    els.empty.hidden = false;
    app.turn = null;
    app.hasTalked = false;
    app.lastUserText = "";
    clearError();
    setState(S.IDLE, "new conversation");
    await reconnect();
  }

  // ---------------------------------------------------------------------------
  // settings + developer info
  // ---------------------------------------------------------------------------

  function loadSettings() {
    const defaults = { autoplay: true, volume: 1, micId: "", characterId: "" };
    try { return Object.assign(defaults, JSON.parse(localStorage.getItem("cvai.settings") || "{}")); }
    catch (e) { return defaults; }
  }

  function saveSettings() {
    try { localStorage.setItem("cvai.settings", JSON.stringify(app.settings)); } catch (e) {}
  }

  function renderDev() {
    if (!els.settings.open) return;
    const rows = [
      ["会话", app.session ? app.session.id : "—"],
      ["语音引擎", app.session ? app.session.engine : "—"],
      ["对话模型", app.model || "—"],
      ["连接", app.conn],
      ["麦克风", app.mic],
      ["对话状态", app.state],
    ];
    els.devInfo.innerHTML = "";
    rows.forEach(([k, v]) => {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      els.devInfo.append(dt, dd);
    });
    els.devTurns.innerHTML = "";
    app.turns.forEach((t) => {
      const tr = document.createElement("tr");
      [t.kind, t.rec, t.stt, t.text, t.audio, t.total].forEach((v) => {
        const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
      });
      els.devTurns.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------------------
  // wiring
  // ---------------------------------------------------------------------------

  els.mic.addEventListener("click", () => {
    if (app.state === S.LISTENING || app.state === S.SPEECH_DETECTED) finishListening();
    else startListening();
  });

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = els.input.value.trim();
    if (!text) return;
    els.input.value = "";
    autosize();
    sendText(text);
  });

  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
    render();
  }
  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || els.settings.open) return;
    if (app.state === S.LISTENING || app.state === S.SPEECH_DETECTED) finishListening();
    else if ([S.THINKING, S.GENERATING_SPEECH, S.SPEAKING].includes(app.state)) interrupt();
  });

  els.statusAction.addEventListener("click", interrupt);
  els.alertRetry.addEventListener("click", () => {
    const retry = app.error && app.error.retry;
    clearError();
    if (retry) retry();
  });
  $("alert-close").addEventListener("click", clearError);
  els.reconnect.addEventListener("click", reconnect);
  els.bootRetry.addEventListener("click", boot);
  $("new-chat").addEventListener("click", newConversation);

  $("open-settings").addEventListener("click", () => {
    els.setAutoplay.checked = app.settings.autoplay;
    els.setVolume.value = app.settings.volume;
    els.settings.showModal();
    renderDev();
  });
  els.setAutoplay.addEventListener("change", () => { app.settings.autoplay = els.setAutoplay.checked; saveSettings(); });
  els.setVolume.addEventListener("input", () => {
    app.settings.volume = Number(els.setVolume.value);
    if (player.current) player.current.volume = app.settings.volume;
    saveSettings();
  });
  els.setMic.addEventListener("change", () => { app.settings.micId = els.setMic.value; saveSettings(); });
  els.setCharacter.addEventListener("change", () => {
    app.settings.characterId = els.setCharacter.value;
    saveSettings();
    els.settings.close();
    newConversation();
  });

  // Keep the connection pill honest: the LLM can go away while the page is open.
  setInterval(() => { if (app.socket) checkLlm(); }, 20000);

  render();
  boot();
})();
