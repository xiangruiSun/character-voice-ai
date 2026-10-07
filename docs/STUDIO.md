# Character AI Studio

Connect a brain → create a voice → create a character → talk.

The Studio is the user-facing product built on the conversation pipeline: a Next.js
website served by the FastAPI backend, a SQLite database, a Celery training worker, and
the same STT → LLM → TTS conversation engine (with its explicit voice states) as before.

```
Browser ── http://127.0.0.1:8000 ─► FastAPI ──┬─► SQLite (data/studio.db)        metadata
  (Next.js static export,                     ├─► data/  (LocalFilesystemStorage) audio, checkpoints
   served by FastAPI)                         ├─► LLMProvider ─► Ollama / OpenAI-compatible API
                                              ├─► GPT-SoVITS api_v2 (:9880)       voice synthesis
                                              └─► Redis/Memurai (:6379) ─► Celery worker ─► GPU training
```

## Start everything

Prerequisites (one-time): Python env (`.venv`), Node.js, Ollama with `qwen3:4b`, the
GPT-SoVITS environment (`models/gpt_sovits/src`, see `docs/VOICE_TRAINING.md`),
and a Redis on :6379 — either Memurai (Redis-compatible, needs admin):
`winget install Memurai.MemuraiDeveloper`, or, without admin rights, a portable Redis in
the git-ignored `tools/redis/`: `powershell -ExecutionPolicy Bypass -File scripts\install_redis_portable.ps1`.

```powershell
cd C:\Users\gaming\character-voice-ai
powershell -ExecutionPolicy Bypass -File scripts\start_studio.ps1
```

It checks Ollama, starts Memurai or else the portable Redis, rebuilds the website if its sources changed, starts the
voice engine, the backend and the training worker (one window each), and opens
<http://127.0.0.1:8000/>. Close a window to stop that service.

Individually:

| Service | Command |
|---|---|
| Backend + website | `.venv\Scripts\python.exe -m cvai_api.app --port 8000` |
| Training worker | `.venv\Scripts\python.exe -m cvai_studio.workers.celery_app` |
| Voice engine | `powershell -ExecutionPolicy Bypass -File scripts\start_voice_server.ps1` |
| Website (dev, hot reload) | `cd apps\studio; npm run dev` → <http://localhost:3000> (proxies the API to :8000) |
| Website (build) | `cd apps\studio; npm run build` → `apps/studio/out`, served by the backend |

On first start the backend migrates the database and imports the pre-Studio setup once:
the local Qwen connection, 卡提希娅's trained voice, and the character 卡提希娅.

## Pages

| Route | Purpose |
|---|---|
| `/chat` | Talk to a character (text or voice). Guides first-time users through the three setup steps |
| `/characters` | Character cards; the builder (Identity → Brain → Voice → Preview) |
| `/voice-studio` | Voice packs, trained voices, training jobs; `/voice-studio/pack?id=…` is the 6-step workflow |
| `/models` | Model connections: add (local / public API / custom endpoint), test, set default |
| `/settings` | Chat preferences (auto-play, volume, microphone, theme) and service health |

The pre-Studio pages remain at `/classic` (voice chat), `/classic/chat` (text) and `/dev`.

## Configuration (`.env` in the repository root, git-ignored)

| Variable | Default | Meaning |
|---|---|---|
| `STUDIO_DATA_DIR` | `data/` | Uploads, datasets, checkpoints, the SQLite file, `secret.key` |
| `STUDIO_DATABASE_URL` | `sqlite:///data/studio.db` | Any SQLAlchemy URL; PostgreSQL works for hosted use |
| `STUDIO_REDIS_URL` | `redis://127.0.0.1:6379/0` | Celery broker |
| `STUDIO_SECRET_KEY` | generated into `data/secret.key` | Encrypts stored API keys (Fernet) |
| `SELF_HOSTED_MODE`, `ALLOW_PRIVATE_MODEL_ENDPOINTS` | `true` | Set `false` in a hosted deployment to refuse localhost/LAN model endpoints |
| `GPT_SOVITS_DIR`, `GPT_SOVITS_URL` | `models/gpt_sovits/src`, `http://127.0.0.1:9880` | Training engine and its API |
| `STUDIO_MAX_UPLOAD_MB` | `2048` | Per-file upload limit |

API keys are entered in the website, sent once to the backend, stored encrypted, and
never returned (the UI shows `sk-••••••••1234`). Back up `data/secret.key` with the
database: without it stored keys cannot be decrypted and must be re-entered.

## Architecture

```
services/studio/cvai_studio/
  api/          routes.py (thin), schemas.py (what may leave the backend)
  services/     model_connections, voice_packs, training, voice_models, characters, seed
  workers/      celery_app.py (worker, dispatch, health), tasks.py
  providers/    storage.py (StorageProvider), training/ (VoiceTrainingProvider, GPT-SoVITS)
  db/           models.py (SQLAlchemy), session.py, migrations/ (Alembic)
  domain/       enums.py — statuses and their transition tables
  core/         config, security (SecretStore), logging (redaction)
apps/studio/    Next.js (TypeScript, Tailwind, shadcn/ui on Base UI, TanStack Query)
  app/          one folder per route
  features/     chat (VoiceEngine state machine), characters, model-connections, voice-studio
  lib/          api client, types (backend contracts), queries (TanStack), formatting
```

Boundaries that keep parts replaceable:

- **LLMs** — `LLMProvider` (`complete`, `stream`, `complete_structured`, `test_connection`,
  `list_models`, `check`). `OllamaLLMProvider` and the OpenAI-compatible provider are the
  two implementations; only `ModelConnectionService` turns a stored connection into one.
- **Training** — `VoiceTrainingProvider` (`presets`, `advanced_options`, `validate_dataset`,
  `prepare_dataset`, `train` with structured `TrainingEvent`s, `synthesize`,
  `evaluate_checkpoint`). GPT-SoVITS v2ProPlus is the first engine; another registers in
  `services/training.engines()`.
- **Storage** — every file is a key (`voicepacks/vp_…/raw/….wav`); `LocalFilesystemStorage`
  maps keys under `data/`. An S3-compatible provider implements the same five methods.
- **Secrets** — `SecretStore`; `FernetSecretStore` locally, a vault/KMS later.
- **Characters** reference a `ModelConnection` and a `VoiceModel` by id (or follow the
  default connection), and are turned into a chat session by `CharacterService.session_parts`.

State machines (explicit tables in `domain/enums.py`, transitions checked):

```
Model connection   DRAFT → TESTING → CONNECTED | ERROR          (+ DISABLED)
Voice pack         EMPTY → UPLOADING → PROCESSING → NEEDS_REVIEW → READY   (+ ERROR)
Training job       QUEUED → VALIDATING → PREPARING → TRAINING → EVALUATING → COMPLETED
                   any running stage → FAILED | CANCEL_REQUESTED → CANCELLED
Voice conversation IDLE → REQUESTING_MIC_PERMISSION → LISTENING → SPEECH_DETECTED
                   → TRANSCRIBING → THINKING → GENERATING_SPEECH → SPEAKING → IDLE  (+ ERROR)
```

Training never runs in a request: `POST /api/training-jobs` creates the job, commits, and
queues it; the worker runs it and commits each step; the page follows
`GET /api/training-jobs/{id}/events` (Server-Sent Events). Closing the browser does not
stop training. A worker that dies mid-job marks that job failed when it restarts.

## API (all under `/api`)

```
GET/POST        /model-connections            PATCH/DELETE /model-connections/{id}
POST            /model-connections/test       (unsaved draft)   /model-connections/models (discover)
POST            /model-connections/{id}/test  /{id}/default     GET /model-connections/{id}/models
GET/POST        /voice-packs                  GET/PATCH/DELETE /voice-packs/{id}
POST            /voice-packs/{id}/upload      /{id}/prepare  /{id}/rights  /{id}/approve
GET             /voice-packs/{id}/samples     PATCH /audio-samples/{id}
GET/POST        /training-jobs                GET /training-jobs/options
GET             /training-jobs/{id}           /{id}/events (SSE)    POST /{id}/cancel  /{id}/retry
GET             /voice-models                 PATCH/DELETE /voice-models/{id}   POST /{id}/synthesize
GET/POST        /characters                   PATCH/DELETE /characters/{id}   POST /{id}/avatar, /preview
GET             /system/health                /system/overview      /files/{key}
POST /sessions {"studio_character_id": …}  + WS /sessions/{id}/ws   (chat)
```

Errors are `{"detail": {"message": "…", "hint": "…"}}` with 400/404/409/503.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\test_studio.py -q      # services, workers (fake engine), API
cd apps\studio; npm test                                         # voice engine, workflow, wizard
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Settings shows 任务队列 / 训练进程 down | Start Redis (`Start-Service Memurai`, or re-run `start_studio.ps1`, which starts the portable copy) and the worker window |
| Training stays 排队中 | The worker is not running; start it — queued jobs then begin |
| 训练时显存不足 | Close other GPU programs or lower 批大小 in the training step's advanced settings |
| 语音引擎未启动 | `scripts\start_voice_server.ps1` |
| A stored API key "could not be decrypted" | `data/secret.key` changed; re-enter the key on the Models page |
| Page shows 无法连接到本地后端 | The backend window was closed; restart it |
