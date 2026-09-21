"""Human review of a voice pack (spec §8: "the system should support manual correction").

Automatic annotation gets a pack most of the way; the last stretch is a person listening.
This produces a single self-contained page — no server, no build step — where a reviewer
can play each segment, fix the transcript, set the real style, and approve or reject.
What comes back is a JSON patch applied to the pipeline state, and every touched segment
is marked ``human_edited`` so no later automatic pass overwrites it.

The review order is deliberate: **lowest quality score first**. The instinct is to show
the good clips first because approving them feels productive, but the clips that decide
whether a dataset is any good are the marginal ones, and they are the ones that get
rubber-stamped at the end of a long session.
"""

from __future__ import annotations

import json
from pathlib import Path

from cvai_types import CVAIModel, RejectionReason, ReviewStatus, VoicePackManifest
from pydantic import Field

from .backends import STUB_TRANSCRIPT_SOURCE
from .state import PipelineState, SegmentRecord

REVIEW_FILENAME = "review.html"
PATCH_FILENAME = "review-patch.json"


class ReviewPatchEntry(CVAIModel):
    segment_id: str
    transcript: str | None = None
    style: str | None = None
    review_status: ReviewStatus | None = None
    rejection_reason: RejectionReason | None = None
    notes: str | None = None
    #: Mark this segment as a confirmed example of the character, for the speaker
    #: centroid. A handful of these turns the speaker filter from a guess into a check.
    speaker_anchor: bool | None = None


class ReviewPatch(CVAIModel):
    voicepack_id: str
    reviewer: str = ""
    entries: list[ReviewPatchEntry] = Field(default_factory=list)


def apply_review_patch(state: PipelineState, patch: ReviewPatch) -> int:
    """Apply a reviewer's decisions. Returns the number of segments changed."""
    if patch.voicepack_id != state.voicepack_id:
        raise ValueError(
            f"patch is for voice pack {patch.voicepack_id!r}, state is "
            f"{state.voicepack_id!r}"
        )

    changed = 0
    anchors = set(state.speaker_anchor_segment_ids)
    for entry in patch.entries:
        segment = state.segment(entry.segment_id)
        if segment is None:
            continue
        touched = False

        if entry.transcript is not None and entry.transcript != segment.transcript:
            segment.transcript = entry.transcript
            segment.transcript_source = "human"
            segment.transcript_confidence = 1.0
            touched = True
        if entry.style is not None and entry.style != segment.style:
            segment.style = entry.style
            touched = True
        if entry.notes is not None and entry.notes != segment.reviewer_notes:
            segment.reviewer_notes = entry.notes
            touched = True
        if entry.review_status is not None:
            status = entry.review_status
            # A reviewer who edited the text and approved it has *corrected* it; the
            # distinction is worth keeping, since corrected samples are evidence about
            # how good the ASR was.
            if status is ReviewStatus.APPROVED and entry.transcript is not None:
                status = ReviewStatus.CORRECTED
            if status != segment.review_status:
                segment.review_status = status
                touched = True
            segment.rejection_reason = (
                entry.rejection_reason
                if status is ReviewStatus.REJECTED
                else None
            )
            if status is ReviewStatus.REJECTED and segment.rejection_reason is None:
                segment.rejection_reason = RejectionReason.OTHER
        if entry.speaker_anchor is not None:
            if entry.speaker_anchor:
                anchors.add(entry.segment_id)
            else:
                anchors.discard(entry.segment_id)
            touched = True

        if touched:
            segment.human_edited = True
            changed += 1

    state.speaker_anchor_segment_ids = sorted(anchors)
    return changed


def load_review_patch(path: Path) -> ReviewPatch:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReviewPatch.model_validate(data)


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------


def write_review_page(
    state: PipelineState,
    manifest: VoicePackManifest,
    processed_dir: Path,
    *,
    limit: int | None = None,
    status_filter: ReviewStatus | None = None,
) -> Path:
    path = Path(processed_dir) / REVIEW_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_review_page(state, manifest, limit=limit, status_filter=status_filter),
        encoding="utf-8",
    )
    return path


def _segments_for_review(
    state: PipelineState,
    limit: int | None,
    status_filter: ReviewStatus | None,
) -> list[SegmentRecord]:
    pool = list(state.segments)
    if status_filter is not None:
        pool = [s for s in pool if s.review_status is status_filter]
    # Worst first: the marginal clips are the ones that decide dataset quality.
    pool.sort(key=lambda s: (s.quality_score if s.quality_score is not None else 1.0))
    return pool[:limit] if limit else pool


def render_review_page(
    state: PipelineState,
    manifest: VoicePackManifest,
    *,
    limit: int | None = None,
    status_filter: ReviewStatus | None = None,
) -> str:
    segments = _segments_for_review(state, limit, status_filter)
    styles = [s.name for s in manifest.styles]
    payload = {
        "voicepack_id": state.voicepack_id,
        "styles": styles,
        "rejection_reasons": [r.value for r in RejectionReason],
        "segments": [
            {
                "segment_id": s.segment_id,
                # Relative to processed/, where the page lives.
                "audio": _relative_audio(s.audio_path),
                "transcript": s.transcript or "",
                "transcript_source": s.transcript_source or "",
                "is_stub": s.transcript_source == STUB_TRANSCRIPT_SOURCE,
                "style": s.style or "",
                "emotion_auto": s.emotion_auto or "",
                "status": s.review_status.value,
                "notes": s.reviewer_notes,
                "human_edited": s.human_edited,
                "anchor": s.segment_id in state.speaker_anchor_segment_ids,
                "metrics": {
                    "duration_s": round(s.duration_s, 2),
                    "quality": s.quality_score,
                    "speaker": s.speaker_similarity,
                    "snr_db": s.snr_db,
                    "lufs": s.lufs,
                    "clipping": s.clipping_ratio,
                    "f0": s.f0_mean_hz,
                    "cps": s.chars_per_second,
                    "pauses": s.pause_count,
                },
                "warnings": s.quality_notes,
            }
            for s in segments
        ],
    }
    summary = state.summary()
    return _TEMPLATE.format(
        voicepack_id=state.voicepack_id,
        count=len(segments),
        total=len(state.segments),
        approved=summary["approved"],
        pending=summary["pending_review"],
        minutes=summary["approved_minutes"],
        payload=json.dumps(payload, ensure_ascii=False),
    )


def _relative_audio(pack_relative: str) -> str:
    prefix = "processed/"
    return pack_relative[len(prefix) :] if pack_relative.startswith(prefix) else f"../{pack_relative}"


_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>语音包校对 · {voicepack_id}</title>
<style>
  :root {{
    --bg:#fff; --fg:#16181d; --muted:#5b6270; --line:#e3e6ec; --card:#f7f8fa;
    --accent:#2f6df6; --ok:#1f9254; --bad:#c23b3b; --warn:#b3730a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#14161a; --fg:#e9ecf1; --muted:#98a1b0; --line:#2a2f38; --card:#1b1e24;
      --accent:#6f9bff; --ok:#4ec281; --bad:#ef7676; --warn:#e0a84a;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}}
  .wrap{{max-width:900px;margin:0 auto;padding:20px 16px 110px}}
  h1{{font-size:20px;margin:0 0 4px}}
  .sub{{color:var(--muted);font-size:13px;margin-bottom:16px}}
  .toolbar{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:16px}}
  .toolbar input,.toolbar select{{padding:7px 9px;border:1px solid var(--line);
    border-radius:8px;background:var(--bg);color:var(--fg)}}
  .seg{{border:1px solid var(--line);border-left-width:4px;border-radius:10px;
    padding:14px;margin-bottom:12px;background:var(--card)}}
  .seg[data-status="approved"],.seg[data-status="corrected"]{{border-left-color:var(--ok)}}
  .seg[data-status="rejected"]{{border-left-color:var(--bad);opacity:.6}}
  .seg[data-status="pending"]{{border-left-color:var(--warn)}}
  .head{{display:flex;justify-content:space-between;gap:12px;align-items:baseline}}
  .sid{{font-size:12px;color:var(--muted);font-family:ui-monospace,monospace}}
  .metrics{{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted);
    margin:6px 0 10px}}
  .metrics b{{font-weight:600;color:var(--fg)}}
  .flag{{color:var(--bad);font-weight:600}}
  audio{{width:100%;margin-bottom:10px}}
  textarea{{width:100%;min-height:52px;padding:8px;border-radius:8px;
    border:1px solid var(--line);background:var(--bg);color:var(--fg);
    font:15px/1.5 inherit}}
  textarea.stub{{border-color:var(--warn);background:rgba(224,168,74,.08)}}
  .row{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px}}
  button{{padding:7px 12px;border-radius:8px;border:1px solid var(--line);
    background:var(--bg);color:var(--fg);font:inherit;cursor:pointer}}
  button.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
  .bar{{position:fixed;left:0;right:0;bottom:0;background:var(--bg);
    border-top:1px solid var(--line);padding:10px 16px;display:flex;
    justify-content:space-between;align-items:center;gap:12px}}
  .bar button{{border-color:var(--accent);background:var(--accent);color:#fff}}
  .hint{{font-size:12px;color:var(--muted)}}
</style>
</head>
<body>
<div class="wrap">
  <h1>语音包校对 · {voicepack_id}</h1>
  <div class="sub">
    显示 {count} / {total} 条 · 已通过 {approved} 条（{minutes} 分钟）· 待校对 {pending} 条 ·
    按质量分升序排列，问题最多的在最前面
  </div>

  <div class="toolbar">
    <input id="reviewer" placeholder="校对人" autocomplete="off">
    <select id="filter">
      <option value="">全部</option>
      <option value="pending">仅待校对</option>
      <option value="approved">仅已通过</option>
      <option value="rejected">仅已拒绝</option>
    </select>
    <span class="hint">快捷键：A 通过 · R 拒绝 · 空格 播放</span>
  </div>

  <div id="list"></div>
</div>

<div class="bar">
  <span id="progress"></span>
  <span>
    <button id="download">导出校对结果 JSON</button>
  </span>
</div>

<script id="payload" type="application/json">{payload}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById("payload").textContent);
  const edits = {{}};
  const storeKey = "cvai-review-" + data.voicepack_id;
  try {{
    const saved = localStorage.getItem(storeKey);
    if (saved) Object.assign(edits, JSON.parse(saved));
  }} catch (e) {{}}

  function persist() {{
    try {{ localStorage.setItem(storeKey, JSON.stringify(edits)); }} catch (e) {{}}
  }}
  function edit(id) {{
    edits[id] = edits[id] || {{ segment_id: id }};
    return edits[id];
  }}

  const list = document.getElementById("list");

  data.segments.forEach(function (seg) {{
    const saved = edits[seg.segment_id] || {{}};
    const card = document.createElement("div");
    card.className = "seg";
    card.dataset.status = saved.review_status || seg.status;
    card.dataset.id = seg.segment_id;

    const head = document.createElement("div");
    head.className = "head";
    const sid = document.createElement("span");
    sid.className = "sid";
    sid.textContent = seg.segment_id;
    head.appendChild(sid);
    card.appendChild(head);

    const m = seg.metrics;
    const metrics = document.createElement("div");
    metrics.className = "metrics";
    function metric(label, value, bad) {{
      if (value === null || value === undefined) return;
      const span = document.createElement("span");
      span.innerHTML = label + " <b>" + value + "</b>";
      if (bad) span.classList.add("flag");
      metrics.appendChild(span);
    }}
    metric("时长", m.duration_s + "s", m.duration_s < 1 || m.duration_s > 20);
    metric("质量", m.quality, m.quality !== null && m.quality < 0.6);
    metric("说话人", m.speaker, m.speaker !== null && m.speaker < 0.55);
    metric("SNR", m.snr_db !== null ? m.snr_db + "dB" : null, m.snr_db !== null && m.snr_db < 8);
    metric("削波", m.clipping, m.clipping > 0.01);
    metric("F0", m.f0 !== null ? Math.round(m.f0) + "Hz" : null, false);
    metric("字/秒", m.cps, false);
    metric("停顿", m.pauses, false);
    if (seg.emotion_auto) metric("自动情绪", seg.emotion_auto, false);
    (seg.warnings || []).forEach(function (w) {{ metric("⚠", w, true); }});
    card.appendChild(metrics);

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = seg.audio;
    card.appendChild(audio);

    const text = document.createElement("textarea");
    text.value = saved.transcript !== undefined ? saved.transcript : seg.transcript;
    if (seg.is_stub) {{
      text.classList.add("stub");
      text.placeholder = "ASR 未运行，请人工输入文本";
    }}
    text.addEventListener("input", function () {{
      edit(seg.segment_id).transcript = text.value;
      persist();
    }});
    card.appendChild(text);

    const row = document.createElement("div");
    row.className = "row";

    const style = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = ""; blank.textContent = "（未指定风格）";
    style.appendChild(blank);
    data.styles.forEach(function (name) {{
      const option = document.createElement("option");
      option.value = name; option.textContent = name;
      style.appendChild(option);
    }});
    style.value = saved.style !== undefined ? saved.style : (seg.style || "");
    style.addEventListener("change", function () {{
      edit(seg.segment_id).style = style.value;
      persist();
    }});
    row.appendChild(style);

    const reason = document.createElement("select");
    data.rejection_reasons.forEach(function (name) {{
      const option = document.createElement("option");
      option.value = name; option.textContent = name;
      reason.appendChild(option);
    }});
    reason.value = saved.rejection_reason || "low_quality";
    reason.style.display = card.dataset.status === "rejected" ? "" : "none";
    reason.addEventListener("change", function () {{
      edit(seg.segment_id).rejection_reason = reason.value;
      persist();
    }});

    const approve = document.createElement("button");
    approve.textContent = "通过 (A)";
    const reject = document.createElement("button");
    reject.textContent = "拒绝 (R)";
    const anchor = document.createElement("button");
    anchor.textContent = "标记为本角色基准";
    anchor.title = "用于构建说话人中心向量；几条可靠样本就够了";

    function paint() {{
      const status = card.dataset.status;
      approve.classList.toggle("on", status === "approved" || status === "corrected");
      reject.classList.toggle("on", status === "rejected");
      reason.style.display = status === "rejected" ? "" : "none";
      anchor.classList.toggle("on", Boolean(edits[seg.segment_id] &&
        edits[seg.segment_id].speaker_anchor !== undefined
          ? edits[seg.segment_id].speaker_anchor : seg.anchor));
      updateProgress();
    }}

    approve.addEventListener("click", function () {{
      card.dataset.status = "approved";
      edit(seg.segment_id).review_status = "approved";
      persist(); paint();
    }});
    reject.addEventListener("click", function () {{
      card.dataset.status = "rejected";
      const e = edit(seg.segment_id);
      e.review_status = "rejected";
      e.rejection_reason = reason.value;
      persist(); paint();
    }});
    anchor.addEventListener("click", function () {{
      const e = edit(seg.segment_id);
      e.speaker_anchor = !(e.speaker_anchor !== undefined ? e.speaker_anchor : seg.anchor);
      persist(); paint();
    }});

    row.appendChild(approve);
    row.appendChild(reject);
    row.appendChild(reason);
    row.appendChild(anchor);
    card.appendChild(row);

    card.addEventListener("keydown", function (event) {{
      if (event.target.tagName === "TEXTAREA") return;
      if (event.key === "a" || event.key === "A") approve.click();
      if (event.key === "r" || event.key === "R") reject.click();
      if (event.key === " ") {{ event.preventDefault(); audio.paused ? audio.play() : audio.pause(); }}
    }});
    card.tabIndex = 0;

    list.appendChild(card);
    paint();
  }});

  document.getElementById("filter").addEventListener("change", function (event) {{
    const want = event.target.value;
    Array.from(list.children).forEach(function (card) {{
      const status = card.dataset.status;
      const match = !want || status === want ||
        (want === "approved" && status === "corrected");
      card.style.display = match ? "" : "none";
    }});
  }});

  function updateProgress() {{
    let decided = 0;
    Array.from(list.children).forEach(function (card) {{
      if (card.dataset.status !== "pending") decided += 1;
    }});
    document.getElementById("progress").textContent =
      decided + " / " + data.segments.length + " 已决定";
  }}

  document.getElementById("download").addEventListener("click", function () {{
    const reviewer = (document.getElementById("reviewer").value || "").trim();
    const entries = Object.values(edits);
    if (!entries.length) {{ alert("还没有任何修改。"); return; }}
    const patch = {{
      voicepack_id: data.voicepack_id,
      reviewer: reviewer,
      entries: entries
    }};
    const blob = new Blob([JSON.stringify(patch, null, 2)], {{ type: "application/json" }});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "review-patch.json";
    link.click();
  }});

  updateProgress();
}})();
</script>
</body>
</html>
"""
