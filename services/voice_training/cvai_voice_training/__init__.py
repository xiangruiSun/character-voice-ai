"""Dataset export and training support (Milestones 3-5).

One approved dataset, six engines, six incompatible training formats. This package holds
the translation layer, so the Voice Pack stays the single source of truth and an engine's
format is a *view* of it rather than a second copy that drifts.

Three properties hold for every exporter, and each is enforced rather than documented:

* **Held-out lines are never exported.** They are the benchmark's ground truth and the
  listening test's anchor. Training on them would make every subsequent number
  meaningless, and the mistake is invisible once made.
* **Exports are reproducible and self-describing.** Each writes an ``export.json``
  recording the voice pack version, the split counts, and the exporter version.
* **Audio is copied, never moved or modified.** ``clean/`` remains the canonical audio.
"""

from __future__ import annotations

from .exporters import (
    EXPORTERS,
    CosyVoiceExporter,
    DatasetExporter,
    ExportResult,
    FishSpeechExporter,
    GPTSoVITSExporter,
    QwenTTSExporter,
    VoxCPMExporter,
    export_dataset,
    get_exporter,
)

__all__ = [
    "EXPORTERS",
    "CosyVoiceExporter",
    "DatasetExporter",
    "ExportResult",
    "FishSpeechExporter",
    "GPTSoVITSExporter",
    "QwenTTSExporter",
    "VoxCPMExporter",
    "export_dataset",
    "get_exporter",
]
