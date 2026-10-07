// The voice conversation engine: socket, microphone, playback and ONE state machine.
// Framework-free; React reads snapshots through useSyncExternalStore.
//
// Three separate pieces of state:
//   A. conn  — the backend session (connecting | connected | disconnected)
//   B. mic   — whether audio is really being captured (off | pending | on)
//   C. state — what the conversation is doing (VoiceState)

import { api, ApiError, wsUrl } from "@/lib/api";

export const VoiceState = {
  IDLE: "idle",
  REQUESTING_MIC_PERMISSION: "requesting_mic_permission",
  LISTENING: "listening",
  SPEECH_DETECTED: "speech_detected",
  TRANSCRIBING: "transcribing",
  THINKING: "thinking",
  GENERATING_SPEECH: "generating_speech",
  SPEAKING: "speaking",
  ERROR: "error",
} as const;
export type VoiceStateValue = (typeof VoiceState)[keyof typeof VoiceState];
const S = VoiceState;

const TRANSITIONS: Record<VoiceStateValue, VoiceStateValue[]> = {
  idle: [S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING, S.ERROR],
  requesting_mic_permission: [S.LISTENING, S.IDLE, S.ERROR],
  listening: [S.SPEECH_DETECTED, S.TRANSCRIBING, S.IDLE, S.ERROR, S.THINKING],
  speech_detected: [S.TRANSCRIBING, S.IDLE, S.ERROR, S.THINKING],
  transcribing: [S.THINKING, S.IDLE, S.ERROR],
  thinking: [S.GENERATING_SPEECH, S.SPEAKING, S.IDLE, S.ERROR, S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
  generating_speech: [S.SPEAKING, S.IDLE, S.ERROR, S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
  speaking: [S.GENERATING_SPEECH, S.IDLE, S.ERROR, S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
  error: [S.IDLE, S.REQUESTING_MIC_PERMISSION, S.LISTENING, S.THINKING],
};

export function canTransition(from: VoiceStateValue, to: VoiceStateValue): boolean {
  return TRANSITIONS[from].includes(to);
}

export type ChatMessage = {
  id: string;
  role: "user" | "char" | "note";
  text: string;
  voice?: boolean;
  typing?: boolean;
  hasAudio?: boolean;
};

export type ChatError = { text: string; retryLabel?: string; canRetry: boolean };

export type Snapshot = {
  conn: "connecting" | "connected" | "disconnected";
  mic: "off" | "pending" | "on";
  state: VoiceStateValue;
  error: ChatError | null;
  messages: ChatMessage[];
  characterName: string;
  modelLabel: string;
  voiceLabel: string;
  hasTalked: boolean;
};

type Turn = {
  kind: "voice" | "text";
  serverId: string | null;
  ended: boolean;
  message: ChatMessage | null;
  clips: Blob[];
  t: Record<string, number>;
};

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

let nextId = 0;
const newId = () => `m${++nextId}`;

export class VoiceEngine {
  private snap: Snapshot;
  private listeners = new Set<() => void>();
  private socket: WebSocket | null = null;
  private sessionId: string | null = null;
  private turn: Turn | null = null;
  private staleTurnIds = new Set<string>();
  private acceptAudio = false;
  private lastUserText = "";
  private retryFn: (() => void) | null = null;
  private clipsByMessage = new Map<string, Blob[]>();
  private queue: HTMLAudioElement[] = [];
  private current: HTMLAudioElement | null = null;
  private mic: { stream: MediaStream | null; context: AudioContext | null; node: AudioWorkletNode | null; analyser: AnalyserNode | null } =
    { stream: null, context: null, node: null, analyser: null };
  readonly log: { from: string; to: string; why: string; t: number }[] = [];
  settings = { autoplay: true, volume: 1, micId: "" };

  configure(settings: { autoplay: boolean; volume: number; micId: string }) { this.settings = { ...settings }; }

  constructor(private characterId: string) {
    this.snap = { conn: "connecting", mic: "off", state: S.IDLE, error: null, messages: [],
                  characterName: "", modelLabel: "", voiceLabel: "", hasTalked: false };
  }

  // -- store ------------------------------------------------------------------------
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => this.listeners.delete(listener); };
  getSnapshot = () => this.snap;
  private patch(p: Partial<Snapshot>) { this.snap = { ...this.snap, ...p }; this.listeners.forEach((l) => l()); }

  private setState(next: VoiceStateValue, why: string) {
    const prev = this.snap.state;
    if (prev === next) return;
    if (!canTransition(prev, next)) console.warn(`[voice] unexpected ${prev} → ${next} (${why})`);
    this.log.push({ from: prev, to: next, why, t: Math.round(performance.now()) });
    this.patch({ state: next });
  }

  private isListening() {
    const s = this.snap.state;
    return s === S.LISTENING || s === S.SPEECH_DETECTED || s === S.REQUESTING_MIC_PERMISSION;
  }

  // -- messages ---------------------------------------------------------------------
  private addMessage(m: Omit<ChatMessage, "id">): ChatMessage {
    const msg = { ...m, id: newId() };
    this.patch({ messages: [...this.snap.messages, msg] });
    return msg;
  }
  private updateMessage(id: string, p: Partial<ChatMessage>) {
    this.patch({ messages: this.snap.messages.map((m) => (m.id === id ? { ...m, ...p } : m)) });
    if (this.turn?.message?.id === id) this.turn.message = { ...this.turn.message, ...p };
  }
  private removeMessage(id: string) { this.patch({ messages: this.snap.messages.filter((m) => m.id !== id) }); }

  private charMessage(): ChatMessage | null {
    const turn = this.turn;
    if (!turn) return null;
    if (!turn.message) turn.message = this.addMessage({ role: "char", text: "", typing: true });
    return turn.message;
  }

  // -- connection -------------------------------------------------------------------
  async connect() {
    // Tests and the developer console read state transitions from window.__voice.log.
    if (typeof window !== "undefined") (window as unknown as { __voice: VoiceEngine }).__voice = this;
    this.patch({ conn: "connecting" });
    try {
      const data = await api.post<{ session_id: string; character_name: string; model_label?: string; voice_label?: string }>(
        "/sessions", { studio_character_id: this.characterId });
      this.sessionId = data.session_id;
      this.patch({ characterName: data.character_name, modelLabel: data.model_label ?? "", voiceLabel: data.voice_label ?? "" });
      await this.openSocket();
    } catch (e) {
      this.patch({ conn: "disconnected" });
      const message = e instanceof ApiError ? e.message : "无法连接本地 AI";
      this.showError(message, () => this.reconnect(), "重新连接");
    }
  }

  private openSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsUrl(`/sessions/${this.sessionId}/ws`));
      socket.binaryType = "blob";
      let opened = false;
      socket.onopen = () => { opened = true; this.socket = socket; this.patch({ conn: "connected" }); resolve(); };
      socket.onmessage = (event) => this.onMessage(event);
      socket.onclose = () => {
        if (this.socket !== socket && opened) return;
        this.socket = null;
        this.patch({ conn: "disconnected" });
        void this.releaseMic();
        this.stopPlayback();
        if (opened) this.showError("连接已断开", () => this.reconnect(), "重新连接");
        else reject(new Error("WebSocket failed"));
      };
    });
  }

  async reconnect() { this.clearError(); await this.dispose(false); await this.connect(); }

  async dispose(deleteSession = true) {
    await this.releaseMic();
    this.stopPlayback();
    const socket = this.socket;
    this.socket = null;
    if (socket) { try { socket.send(JSON.stringify({ type: "bye" })); } catch { /* closing */ } socket.close(); }
    if (deleteSession && this.sessionId) void fetch(`/sessions/${this.sessionId}`, { method: "DELETE" }).catch(() => {});
  }

  greet(text: string) { if (text && !this.snap.messages.length) this.addMessage({ role: "char", text }); }

  // -- turns ------------------------------------------------------------------------
  private retireTurn() {
    if (this.turn) { this.turn.ended = true; if (this.turn.serverId) this.staleTurnIds.add(this.turn.serverId); }
    this.acceptAudio = false;
  }
  private beginTurn(kind: Turn["kind"]) {
    this.retireTurn();
    this.turn = { kind, serverId: null, ended: false, message: null, clips: [], t: { start: performance.now() } };
  }
  private acceptTurnId(id: string | null | undefined): boolean {
    if (!id || id === "pending") return true;
    if (this.staleTurnIds.has(id)) return false;
    if (!this.turn) return false;
    if (!this.turn.serverId) { this.turn.serverId = id; return true; }
    return this.turn.serverId === id;
  }

  /** False when nothing was sent (empty, or not connected yet), so the caller keeps the text. */
  sendText(text: string): boolean {
    text = text.trim();
    if (!text || this.socket?.readyState !== WebSocket.OPEN) return false;
    this.clearError();
    if (this.isListening()) void this.releaseMic();
    this.stopPlayback();
    this.beginTurn("text");
    this.lastUserText = text;
    this.addMessage({ role: "user", text });
    this.patch({ hasTalked: true });
    this.socket.send(JSON.stringify({ type: "user_text", text }));
    this.setState(S.THINKING, "text sent");
    return true;
  }

  interrupt() {
    this.stopPlayback();
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: "interrupt" }));
    const turn = this.turn;
    if (turn?.message) { if (!turn.message.text) this.removeMessage(turn.message.id); else this.finishMessage(turn); }
    this.retireTurn();
    if ([S.THINKING, S.GENERATING_SPEECH, S.SPEAKING].includes(this.snap.state as never)) this.setState(S.IDLE, "stopped by user");
  }

  private finishMessage(turn: Turn) {
    if (!turn.message) return;
    this.clipsByMessage.set(turn.message.id, turn.clips);
    this.updateMessage(turn.message.id, { typing: false, hasAudio: turn.clips.length > 0 });
  }

  replay(messageId: string) {
    const clips = this.clipsByMessage.get(messageId);
    if (!clips?.length) return;
    this.stopPlayback();
    clips.forEach((blob) => this.enqueue(blob, false));
  }

  // -- playback ---------------------------------------------------------------------
  private enqueue(blob: Blob, live: boolean) {
    const audio = new Audio(URL.createObjectURL(blob));
    audio.volume = this.settings.volume;
    audio.addEventListener("playing", () => {
      if (live && this.turn && !this.turn.t.playStart) this.turn.t.playStart = performance.now();
      if (live && this.snap.state !== S.SPEAKING && !this.isListening()) this.setState(S.SPEAKING, "audio playing");
    });
    audio.addEventListener("ended", () => { this.current = null; this.next(live); });
    audio.addEventListener("error", () => { this.current = null; this.next(live); });
    this.queue.push(audio);
    if (!this.current) this.next(live);
  }
  private next(live: boolean) {
    if (this.current) return;
    const audio = this.queue.shift();
    if (!audio) { if (live) this.onDrained(); return; }
    this.current = audio;
    audio.play().catch(() => { this.current = null; this.next(live); });
  }
  private stopPlayback() { this.queue = []; if (this.current) { this.current.pause(); this.current = null; } }
  private onDrained() {
    if (this.isListening() || this.snap.state === S.ERROR) return;
    if (this.turn && !this.turn.ended) {
      if (this.snap.state === S.SPEAKING) this.setState(S.GENERATING_SPEECH, "waiting for next chunk");
      return;
    }
    if (this.snap.state === S.SPEAKING) this.setState(S.IDLE, "playback finished");
  }
  get playing() { return !!this.current || this.queue.length > 0; }

  // -- microphone -------------------------------------------------------------------
  async startListening() {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.clearError();
    if ([S.THINKING, S.GENERATING_SPEECH, S.SPEAKING].includes(this.snap.state as never)) this.interrupt();
    this.patch({ mic: "pending" });
    this.setState(S.REQUESTING_MIC_PERMISSION, "mic button");
    try {
      const audio: MediaTrackConstraints = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      if (this.settings.micId) audio.deviceId = { exact: this.settings.micId };
      const stream = await navigator.mediaDevices.getUserMedia({ audio });
      const context = new AudioContext({ sampleRate: MIC_RATE });
      await context.audioWorklet.addModule(URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" })));
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, "capture");
      node.port.onmessage = (event) => this.sendPcm(event.data as Float32Array, context.sampleRate);
      source.connect(node);
      const sink = context.createGain();
      sink.gain.value = 0;
      node.connect(sink).connect(context.destination);
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      this.mic = { stream, context, node, analyser };
    } catch (error) {
      await this.releaseMic();
      const name = (error as Error)?.name;
      const text = name === "NotAllowedError" || name === "SecurityError" ? "无法访问麦克风，请检查浏览器权限。"
        : name === "NotFoundError" || name === "OverconstrainedError" ? "没有找到可用的麦克风设备。" : "麦克风启动失败，请重试。";
      this.showError(text, () => void this.startListening(), "重试");
      return;
    }
    this.socket!.send(JSON.stringify({ type: "user_audio_begin", sample_rate: MIC_RATE, mode: "turn" }));
    this.patch({ mic: "on", hasTalked: true });
    this.beginTurn("voice");
    this.turn!.t.recStart = performance.now();
    this.setState(S.LISTENING, "microphone open");
  }

  private sendPcm(samples: Float32Array, rate: number) {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    let data = samples;
    if (rate !== MIC_RATE) {
      const ratio = rate / MIC_RATE;
      data = new Float32Array(Math.floor(samples.length / ratio));
      for (let i = 0; i < data.length; i++) data[i] = samples[Math.floor(i * ratio)];
    }
    const pcm = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) pcm[i] = Math.round(Math.max(-1, Math.min(1, data[i])) * 32767);
    this.socket.send(pcm.buffer);
  }

  /** Stops capture for real (tracks stopped) and says so at once. */
  async releaseMic() {
    const { stream, context, node } = this.mic;
    if (node) { node.port.onmessage = null; node.disconnect(); }
    stream?.getTracks().forEach((t) => t.stop());
    this.mic = { stream: null, context: null, node: null, analyser: null };
    if (this.snap.mic !== "off") this.patch({ mic: "off" });
    if (context) await context.close().catch(() => {});
  }

  async finishListening() {
    if (!this.isListening()) return;
    if (this.turn) this.turn.t.recEnd = performance.now();
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: "user_audio_end" }));
    await this.releaseMic();
    this.setState(S.TRANSCRIBING, "user pressed done");
  }

  /** 0..1 microphone level, decibel-scaled. */
  level(): number {
    const analyser = this.mic.analyser;
    if (!analyser) return 0;
    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) { const v = (data[i] - 128) / 128; sum += v * v; }
    const db = 20 * Math.log10(Math.sqrt(sum / data.length) + 1e-6);
    return Math.max(0, Math.min(1, (db + 55) / 40));
  }

  // -- errors -----------------------------------------------------------------------
  private showError(text: string, retry: (() => void) | null, retryLabel?: string) {
    this.retryFn = retry;
    this.stopPlayback();
    if (this.turn) this.turn.ended = true;
    this.patch({ error: { text, retryLabel, canRetry: !!retry } });
    this.setState(S.ERROR, text);
  }
  clearError() {
    if (!this.snap.error) return;
    this.retryFn = null;
    this.patch({ error: null });
    if (this.snap.state === S.ERROR) this.setState(S.IDLE, "error dismissed");
  }
  retry() { const fn = this.retryFn; this.clearError(); fn?.(); }

  private friendlyError(message: { code?: string; message?: string }): [string, (() => void) | null, string] {
    const raw = message.message ?? "";
    const lower = raw.toLowerCase();
    const resend = this.lastUserText ? () => this.sendText(this.lastUserText) : null;
    const listenAgain = () => void this.startListening();
    if (message.code === "no_speech") return [raw || "没有听清，请再说一次", listenAgain, "再说一次"];
    if (message.code === "no_stt" || this.snap.state === S.TRANSCRIBING) return ["语音识别失败，请重试", listenAgain, "重新说"];
    // Hosted APIs fail in ways the user can act on; say which.
    if (/error code: 429|quota|rate limit|too many requests/.test(lower)) return ["模型服务商额度不足或请求过于频繁，请稍后再试", resend, "重试"];
    if (/error code: 401|invalid api key|incorrect api key|unauthorized/.test(lower)) return ["模型的 API Key 无效，请在「模型」页重新配置", null, "重试"];
    if (/error code: 5\d\d|high demand|overloaded|service unavailable/.test(lower)) return ["模型服务商暂时繁忙，请稍后重试", resend, "重试"];
    if (/ollama|llm|qwen|openai|api key|model/.test(lower) || this.snap.state === S.THINKING) return ["对话模型没有响应", resend, "重试"];
    if (/sovits|sidecar|synthes|tts/.test(lower) || this.snap.state === S.GENERATING_SPEECH) return ["语音生成失败", resend, "重试"];
    return ["出了点问题，请重试", resend, "重试"];
  }

  // -- server → state ---------------------------------------------------------------
  private onMessage(event: MessageEvent) {
    if (event.data instanceof Blob) {
      const turn = this.turn;
      if (!this.acceptAudio || !turn || turn.ended) return;
      if (!turn.t.firstAudio) turn.t.firstAudio = performance.now();
      this.charMessage();
      turn.clips.push(event.data);
      if (this.settings.autoplay && !this.isListening()) this.enqueue(event.data, true);
      return;
    }
    let msg: Record<string, unknown>;
    try { msg = JSON.parse(event.data as string); } catch { return; }
    const type = msg.type as string;
    if (type !== "transcript" && "turn_id" in msg && !this.acceptTurnId(msg.turn_id as string | null)) {
      if (type === "audio_begin") this.acceptAudio = false;
      return;
    }
    const turn = this.turn;
    switch (type) {
      case "audio_begin":
        this.acceptAudio = !!(turn && !turn.ended);
        break;
      case "vad":
        if (msg.event === "speech_start" && this.snap.state === S.LISTENING) this.setState(S.SPEECH_DETECTED, "server heard speech");
        break;
      case "state":
        this.onServerState(msg.state as string);
        break;
      case "transcript":
        if (turn) turn.t.transcribed = performance.now();
        this.lastUserText = msg.text as string;
        this.addMessage({ role: "user", text: msg.text as string, voice: true });
        if (this.snap.state === S.TRANSCRIBING) this.setState(S.THINKING, "transcript received");
        this.charMessage();
        break;
      case "character_text": {
        if (!turn || turn.ended) break;
        if (!turn.t.firstText) turn.t.firstText = performance.now();
        const m = this.charMessage()!;
        this.updateMessage(m.id, { text: m.text + (msg.text as string), typing: false });
        break;
      }
      case "turn_end":
        if (!turn) break;
        turn.ended = true;
        if (turn.message) { if (!turn.message.text) this.removeMessage(turn.message.id); else this.finishMessage(turn); }
        if (msg.was_interrupted && turn.message?.text) this.addMessage({ role: "note", text: "（已打断）" });
        if (!this.playing && !this.isListening() && this.snap.state !== S.ERROR) this.setState(S.IDLE, "turn ended");
        break;
      case "error": {
        const [text, retry, label] = this.friendlyError(msg as { code?: string; message?: string });
        if (turn?.message && !turn.message.text) this.removeMessage(turn.message.id);
        void this.releaseMic();
        this.showError(text, retry, label);
        break;
      }
    }
  }

  private onServerState(state: string) {
    switch (state) {
      case "transcribing":
        if (this.isListening()) { if (this.turn && !this.turn.t.recEnd) this.turn.t.recEnd = performance.now(); void this.releaseMic(); }
        if (this.snap.state !== S.TRANSCRIBING) this.setState(S.TRANSCRIBING, "server transcribing");
        break;
      case "thinking":
      case "planning":
        if (!this.turn || this.turn.ended) break;
        if (!this.isListening() && this.snap.state !== S.THINKING && this.snap.state !== S.ERROR) this.setState(S.THINKING, `server ${state}`);
        this.charMessage();
        break;
      case "synthesizing":
        if (!this.turn || this.turn.ended) break;
        if (!this.isListening() && this.snap.state !== S.SPEAKING && this.snap.state !== S.ERROR) this.setState(S.GENERATING_SPEECH, "server synthesizing");
        break;
      case "interrupted":
        this.stopPlayback();
        break;
    }
  }
}
