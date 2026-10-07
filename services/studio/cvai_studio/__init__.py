"""Character AI Studio: model connections, voice packs, training, characters.

Layers, outermost first (each depends only on the ones below it):

    api/        FastAPI routers — HTTP in, schemas out, no business logic
    services/   use cases; the only place that talks to repositories and providers
    workers/    Celery tasks; call services, never routes
    providers/  adapters: training engines, storage, secrets (LLM adapters live in
                cvai_llm_providers and are reused, not duplicated)
    db/         SQLAlchemy models, session, Alembic migrations
    domain/     enums and state machines — no I/O at all
    core/       settings, security, logging
"""
