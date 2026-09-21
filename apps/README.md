# Applications

- `api/` — FastAPI backend (Milestones 9-13): conversation orchestrator, WebSocket
  endpoint speaking the `cvai_audio_protocol` messages, and the TTS router.
- `web/` — Next.js + TypeScript frontend (Milestones 9-13): text chat first, then
  microphone capture via AudioWorklet, then streamed playback and barge-in.

Deliberately empty until Milestone 9. Spec §29: do not start by polishing the website.
