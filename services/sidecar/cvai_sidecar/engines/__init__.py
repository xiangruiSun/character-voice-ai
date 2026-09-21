"""Per-engine sidecar adapters.

Each module exposes ``ALLOWED_CALLS`` and ``build(**options)``. Modules are imported by
name at start-up (``CVAI_SIDECAR_ENGINE``) and never all at once — each one imports a
library that only exists in its own container.

GPT-SoVITS and Fish Speech have no module here: they ship their own HTTP servers, and
the client adapters speak those directly rather than wrapping them again.
"""
