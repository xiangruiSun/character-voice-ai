// Per-browser chat preferences (not secrets, not server state): auto-play, volume, mic.
// Exposed as an external store so React reads it without effects or hydration mismatch.

export type ChatSettings = { autoplay: boolean; volume: number; micId: string };
const KEY = "cvai.chat-settings";
export const DEFAULT_CHAT_SETTINGS: ChatSettings = { autoplay: true, volume: 1, micId: "" };
const listeners = new Set<() => void>();
let cached: ChatSettings | null = null;

export function loadChatSettings(): ChatSettings {
  if (cached) return cached;
  try { cached = { ...DEFAULT_CHAT_SETTINGS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") }; }
  catch { cached = { ...DEFAULT_CHAT_SETTINGS }; }
  return cached!;
}

export function saveChatSettings(settings: ChatSettings) {
  cached = settings;
  try { localStorage.setItem(KEY, JSON.stringify(settings)); } catch { /* private mode */ }
  listeners.forEach((l) => l());
}

export function subscribeChatSettings(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export const serverChatSettings = () => DEFAULT_CHAT_SETTINGS;
