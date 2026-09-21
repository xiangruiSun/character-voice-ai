# Infrastructure

Compose files and container definitions for the engine sidecars (Milestones 3-5).

Nothing here yet, and nothing planned beyond local containers: spec §25 rules out
autoscaling, multi-GPU orchestration and distributed training for V1. Storage for the
MVP is the local filesystem — no PostgreSQL, Redis, S3, auth or billing until the
character voice pipeline works.
