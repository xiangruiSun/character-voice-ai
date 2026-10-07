import { describe, expect, it, vi } from "vitest";

import { canTransition, VoiceEngine, VoiceState } from "./voice-engine";

/** An engine with a fake open socket; server messages are fed straight in. */
function engine() {
  const e = new VoiceEngine("ch_test");
  const sent: unknown[] = [];
  const socket = { readyState: WebSocket.OPEN, send: vi.fn((m: unknown) => sent.push(m)), close: vi.fn() };
  Object.assign(e as unknown as { socket: unknown }, { socket });
  e.settings.autoplay = false;
  const server = (msg: Record<string, unknown>) =>
    (e as unknown as { onMessage: (ev: MessageEvent) => void }).onMessage({ data: JSON.stringify(msg) } as MessageEvent);
  return { e, sent, server, states: () => e.log.map((l) => l.to) };
}

describe("transition table", () => {
  it("allows the documented happy path", () => {
    const path = ["idle", "requesting_mic_permission", "listening", "speech_detected", "transcribing",
                  "thinking", "generating_speech", "speaking", "idle"] as const;
    for (let i = 0; i < path.length - 1; i++) expect(canTransition(path[i], path[i + 1])).toBe(true);
  });
  it("refuses skipping transcription", () => {
    expect(canTransition(VoiceState.LISTENING, VoiceState.SPEAKING)).toBe(false);
    expect(canTransition(VoiceState.TRANSCRIBING, VoiceState.SPEAKING)).toBe(false);
  });
});

describe("text turn", () => {
  it("reports a message typed before the session connects as not sent", () => {
    const e = new VoiceEngine("ch_test");                 // no socket yet: still connecting
    expect(e.sendText("你好")).toBe(false);
    expect(e.getSnapshot().messages).toEqual([]);
    expect(e.getSnapshot().state).toBe(VoiceState.IDLE);
  });

  it("walks thinking → generating speech → idle from server events", () => {
    const { e, server, states } = engine();
    e.sendText("你好");
    expect(e.getSnapshot().state).toBe(VoiceState.THINKING);
    server({ type: "state", state: "thinking", turn_id: "t1" });
    server({ type: "state", state: "planning", turn_id: "t1" });
    const typing = e.getSnapshot().messages.at(-1)!;
    expect(typing.role).toBe("char");
    expect(typing.typing).toBe(true);
    server({ type: "character_text", text: "你好呀。", turn_id: "t1" });
    server({ type: "state", state: "synthesizing", turn_id: "t1" });
    expect(e.getSnapshot().state).toBe(VoiceState.GENERATING_SPEECH);
    server({ type: "turn_end", turn_id: "t1", was_interrupted: false });
    expect(states()).toEqual(["thinking", "generating_speech", "idle"]);
    expect(e.getSnapshot().messages.map((m) => [m.role, m.text])).toEqual([["user", "你好"], ["char", "你好呀。"]]);
  });

  it("ignores late messages from a superseded turn", () => {
    const { e, server } = engine();
    e.sendText("第一句");
    server({ type: "state", state: "thinking", turn_id: "t1" });
    e.sendText("第二句");                       // supersedes t1
    server({ type: "character_text", text: "旧回答", turn_id: "t1" });
    server({ type: "turn_end", turn_id: "t1", was_interrupted: true });
    expect(e.getSnapshot().state).toBe(VoiceState.THINKING);
    server({ type: "state", state: "thinking", turn_id: "t2" });
    server({ type: "character_text", text: "新回答", turn_id: "t2" });
    const texts = e.getSnapshot().messages.map((m) => m.text);
    expect(texts).not.toContain("旧回答");
    expect(texts).toContain("新回答");
  });
});

describe("voice turn", () => {
  it("shows the transcript as the user's message, then thinks", () => {
    const { e, server } = engine();
    (e as unknown as { beginTurn: (k: string) => void }).beginTurn("voice");
    (e as unknown as { patch: (p: object) => void }).patch({ state: VoiceState.TRANSCRIBING });
    server({ type: "transcript", text: "今天在做什么？", turn_id: "pending" });
    const snap = e.getSnapshot();
    expect(snap.state).toBe(VoiceState.THINKING);
    expect(snap.messages[0]).toMatchObject({ role: "user", text: "今天在做什么？", voice: true });
  });

  it("offers to listen again when nothing was heard", () => {
    const { e, server } = engine();
    (e as unknown as { patch: (p: object) => void }).patch({ state: VoiceState.TRANSCRIBING });
    server({ type: "error", code: "no_speech", message: "没有听到声音，请再试一次" });
    const snap = e.getSnapshot();
    expect(snap.state).toBe(VoiceState.ERROR);
    expect(snap.error).toMatchObject({ text: "没有听到声音，请再试一次", retryLabel: "再说一次", canRetry: true });
  });

  it("names the failing stage", () => {
    const { e, server } = engine();
    e.sendText("你好");
    server({ type: "error", code: "turn_failed", message: "Ollama is not running at http://127.0.0.1:11434" });
    expect(e.getSnapshot().error?.text).toBe("对话模型没有响应");
  });

  it.each([
    ["Error code: 429 - You exceeded your current quota", "模型服务商额度不足或请求过于频繁，请稍后再试"],
    ["Error code: 503 - This model is currently experiencing high demand", "模型服务商暂时繁忙，请稍后重试"],
    ["Error code: 401 - invalid api key", "模型的 API Key 无效，请在「模型」页重新配置"],
  ])("explains a hosted API failure: %s", (message, expected) => {
    const { e, server } = engine();
    e.sendText("你好");
    server({ type: "error", code: "turn_failed", message });
    expect(e.getSnapshot().error?.text).toBe(expected);
  });
});
