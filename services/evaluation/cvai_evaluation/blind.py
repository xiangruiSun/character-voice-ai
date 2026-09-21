"""Blind listening-test assembly (spec §18).

The rule is simple and the implementation must be boring: the rater must not be able to
tell which system produced a clip, from the filename, the order, or anything else on the
page. So:

* items get opaque ids (``item-000``…), assigned after shuffling;
* the mapping back to candidates lives in a **separate key file** that the rating page
  never loads;
* real character recordings are mixed in as hidden anchors, indistinguishable from the
  candidates in the interface;
* the shuffle seed is recorded, so the same test can be rebuilt exactly.

The generated page is deliberately a single self-contained HTML file with no build step
and no network access: it has to work from a USB stick, in a meeting room, on someone
else's laptop.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from cvai_core.runlog import RunPaths
from cvai_types import (
    AXIS_PROMPTS,
    BenchmarkRun,
    BlindItem,
    BlindKey,
    BlindKeyEntry,
    BlindTestSet,
    GroundTruthItem,
    RatingAxis,
)

GROUND_TRUTH_CANDIDATE_ID = "ground_truth"

DEFAULT_INSTRUCTIONS = """\
请先听几条该角色的真实录音，建立印象，然后为下面每一条音频打分。

- 说话人相似度：这听起来像是目标角色的声音吗？
- 自然度：这听起来像真人录音吗？
- 角色相似度：这听起来像这个角色平时说话的方式吗？
- AI 痕迹：能多明显地听出这是 AI 生成的？（越低越好）

其中部分音频是角色的真实录音，请不要猜测哪些是生成的，按听感打分即可。
"""


def load_ground_truth(run_paths: RunPaths) -> list[GroundTruthItem]:
    path = run_paths.root / "ground_truth.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruthItem.model_validate(entry) for entry in data]


def build_blind_test(
    run: BenchmarkRun,
    run_paths: RunPaths,
    *,
    ground_truth: list[GroundTruthItem] | None = None,
    shuffle_seed: int = 0,
    instructions: str = DEFAULT_INSTRUCTIONS,
    max_items: int | None = None,
) -> tuple[BlindTestSet, BlindKey]:
    """Assemble the blind item list and its separate answer key."""
    entries: list[tuple[str, str, str, bool]] = []  # audio_path, candidate, sentence, gt

    for record in run.succeeded_records():
        if not record.audio_path:
            continue
        entries.append(
            (record.audio_path, record.candidate_id, record.sentence_id, False)
        )

    anchors = ground_truth if ground_truth is not None else load_ground_truth(run_paths)
    for anchor in anchors:
        entries.append(
            (anchor.audio_path, GROUND_TRUTH_CANDIDATE_ID, anchor.item_id, True)
        )

    rng = random.Random(shuffle_seed)
    rng.shuffle(entries)
    if max_items is not None:
        entries = entries[:max_items]

    texts = {r.sentence_id: r.text for r in run.records}
    for anchor in anchors:
        texts.setdefault(anchor.item_id, anchor.transcript)

    items: list[BlindItem] = []
    key_entries: list[BlindKeyEntry] = []
    for position, (audio_path, candidate_id, sentence_id, is_ground_truth) in enumerate(
        entries
    ):
        item_id = f"item-{position:03d}"
        suffix = Path(audio_path).suffix or ".wav"
        items.append(
            BlindItem(
                item_id=item_id,
                # Flat and opaque: the source path encodes the candidate, so it must
                # not reach the rater's browser.
                audio_path=f"audio/{item_id}{suffix}",
                position=position,
                text=texts.get(sentence_id, ""),
            )
        )
        key_entries.append(
            BlindKeyEntry(
                item_id=item_id,
                candidate_id=candidate_id,
                sentence_id=sentence_id,
                is_ground_truth=is_ground_truth,
                source_audio_path=audio_path,
            )
        )

    test_id = f"{run.run_id}-blind"
    return (
        BlindTestSet(
            test_id=test_id,
            run_id=run.run_id,
            items=items,
            shuffle_seed=shuffle_seed,
            instructions=instructions,
        ),
        BlindKey(test_id=test_id, run_id=run.run_id, entries=key_entries),
    )


def write_blind_test(
    run_paths: RunPaths, test: BlindTestSet, key: BlindKey
) -> dict[str, Path]:
    """Copy the audio under opaque names, then write items, key and the rating page."""
    run_paths.ensure()

    audio_dir = run_paths.blind / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sources = {e.item_id: e.source_audio_path for e in key.entries}
    for item in test.items:
        source = sources.get(item.item_id)
        if not source:
            continue
        origin = run_paths.root / source
        if origin.is_file():
            shutil.copy2(origin, run_paths.blind / item.audio_path)

    items_file = run_paths.blind / "items.json"
    key_file = run_paths.blind / "key.json"
    page_file = run_paths.blind / "rate.html"

    items_file.write_text(
        json.dumps(test.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    key_file.write_text(
        json.dumps(key.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page_file.write_text(render_rating_page(test), encoding="utf-8")
    return {"items": items_file, "key": key_file, "page": page_file}


# --------------------------------------------------------------------------------------
# Rating page
# --------------------------------------------------------------------------------------

_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>盲测评分 · {test_id}</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #16181d; --muted: #5b6270; --line: #e3e6ec;
    --accent: #2f6df6; --card: #f7f8fa;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14161a; --fg: #e9ecf1; --muted: #98a1b0; --line: #2a2f38;
      --accent: #6f9bff; --card: #1b1e24;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.6 -apple-system, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  }}
  .wrap {{ max-width: 760px; margin: 0 auto; padding: 24px 16px 96px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
  .intro {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 16px; white-space: pre-wrap; font-size: 14px; margin-bottom: 20px;
  }}
  .rater {{ display: flex; gap: 8px; align-items: center; margin-bottom: 20px; }}
  .rater input {{
    flex: 1; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px;
    background: var(--bg); color: var(--fg);
  }}
  .item {{
    border: 1px solid var(--line); border-radius: 12px; padding: 16px;
    margin-bottom: 14px; background: var(--card);
  }}
  .item.done {{ border-color: var(--accent); }}
  .idx {{ color: var(--muted); font-size: 12px; letter-spacing: .04em; }}
  .text {{ font-size: 16px; margin: 6px 0 10px; }}
  audio {{ width: 100%; margin-bottom: 12px; }}
  .axis {{ display: grid; grid-template-columns: 1fr; gap: 4px; margin-bottom: 10px; }}
  .axis label {{ font-size: 13px; color: var(--muted); }}
  .scale {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .scale button {{
    flex: 1 1 52px; min-width: 52px; padding: 7px 0; cursor: pointer;
    border: 1px solid var(--line); border-radius: 8px;
    background: var(--bg); color: var(--fg); font: inherit;
  }}
  .scale button[aria-pressed="true"] {{
    background: var(--accent); border-color: var(--accent); color: #fff;
  }}
  .row {{ display: flex; gap: 12px; align-items: center; margin-top: 8px; }}
  textarea {{
    width: 100%; min-height: 48px; padding: 8px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg); font: inherit;
  }}
  .bar {{
    position: fixed; left: 0; right: 0; bottom: 0; background: var(--bg);
    border-top: 1px solid var(--line); padding: 10px 16px;
    display: flex; gap: 12px; align-items: center; justify-content: space-between;
  }}
  .bar button {{
    padding: 9px 16px; border-radius: 8px; border: 1px solid var(--accent);
    background: var(--accent); color: #fff; font: inherit; cursor: pointer;
  }}
  .bar .ghost {{ background: transparent; color: var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>角色声音盲测评分</h1>
  <div class="sub">{test_id} · 共 {count} 条 · 顺序已随机，评分与系统无关</div>
  <div class="intro">{instructions}</div>
  <div class="rater">
    <label for="rater">评分人</label>
    <input id="rater" placeholder="你的名字或代号" autocomplete="off">
  </div>
  <div id="items"></div>
</div>
<div class="bar">
  <span id="progress">0 / {count}</span>
  <span>
    <button class="ghost" id="save">保存草稿</button>
    <button id="download">导出评分 JSON</button>
  </span>
</div>
<script id="payload" type="application/json">{payload}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById("payload").textContent);
  const axes = {axes};
  const answers = {{}};
  const storageKey = "cvai-ratings-" + data.test_id;

  try {{
    const saved = localStorage.getItem(storageKey);
    if (saved) Object.assign(answers, JSON.parse(saved));
  }} catch (e) {{ /* private mode: ratings simply are not restored */ }}

  const container = document.getElementById("items");
  data.items.forEach(function (item, index) {{
    const card = document.createElement("div");
    card.className = "item";
    card.id = "card-" + item.item_id;

    const idx = document.createElement("div");
    idx.className = "idx";
    idx.textContent = "#" + (index + 1) + " · " + item.item_id;
    card.appendChild(idx);

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = item.text || "(无文本)";
    card.appendChild(text);

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = item.audio_path;
    card.appendChild(audio);

    axes.forEach(function (axis) {{
      const box = document.createElement("div");
      box.className = "axis";
      const label = document.createElement("label");
      label.textContent = axis.prompt;
      box.appendChild(label);
      const scale = document.createElement("div");
      scale.className = "scale";
      for (let score = 1; score <= 5; score++) {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = score;
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", function () {{
          setScore(item.item_id, axis.key, score);
          Array.from(scale.children).forEach(function (sibling, i) {{
            sibling.setAttribute("aria-pressed", String(i + 1 === score));
          }});
          render();
        }});
        scale.appendChild(button);
      }}
      box.appendChild(scale);
      card.appendChild(box);
    }});

    const row = document.createElement("div");
    row.className = "row";
    const realLabel = document.createElement("label");
    const real = document.createElement("input");
    real.type = "checkbox";
    real.addEventListener("change", function () {{
      setScore(item.item_id, "believed_real", real.checked);
    }});
    realLabel.appendChild(real);
    realLabel.appendChild(document.createTextNode(" 我认为这是真人录音"));
    row.appendChild(realLabel);
    card.appendChild(row);

    const comment = document.createElement("textarea");
    comment.placeholder = "备注（可选）";
    comment.addEventListener("input", function () {{
      setScore(item.item_id, "comment", comment.value);
    }});
    card.appendChild(comment);

    container.appendChild(card);

    const existing = answers[item.item_id];
    if (existing) {{
      axes.forEach(function (axis) {{
        const value = existing[axis.key];
        if (!value) return;
        const scale = card.querySelectorAll(".scale")[axes.indexOf(axis)];
        Array.from(scale.children).forEach(function (sibling, i) {{
          sibling.setAttribute("aria-pressed", String(i + 1 === value));
        }});
      }});
      real.checked = Boolean(existing.believed_real);
      comment.value = existing.comment || "";
    }}
  }});

  function setScore(itemId, key, value) {{
    answers[itemId] = answers[itemId] || {{}};
    answers[itemId][key] = value;
    persist();
  }}

  function persist() {{
    try {{ localStorage.setItem(storageKey, JSON.stringify(answers)); }} catch (e) {{}}
  }}

  function complete(itemId) {{
    const entry = answers[itemId];
    if (!entry) return false;
    return axes.every(function (axis) {{ return entry[axis.key]; }});
  }}

  function render() {{
    let done = 0;
    data.items.forEach(function (item) {{
      const card = document.getElementById("card-" + item.item_id);
      const isDone = complete(item.item_id);
      if (isDone) done += 1;
      card.classList.toggle("done", isDone);
    }});
    document.getElementById("progress").textContent = done + " / " + data.items.length;
  }}

  document.getElementById("save").addEventListener("click", function () {{
    persist();
    alert("已保存到本机浏览器。");
  }});

  document.getElementById("download").addEventListener("click", function () {{
    const rater = (document.getElementById("rater").value || "").trim();
    if (!rater) {{ alert("请先填写评分人。"); return; }}
    const rows = [];
    data.items.forEach(function (item) {{
      const entry = answers[item.item_id];
      if (!entry || !complete(item.item_id)) return;
      rows.push({{
        rater_id: rater,
        item_id: item.item_id,
        speaker_similarity: entry.speaker_similarity,
        naturalness: entry.naturalness,
        character_similarity: entry.character_similarity,
        ai_artifact_level: entry.ai_artifact_level,
        believed_real: Boolean(entry.believed_real),
        comment: entry.comment || ""
      }});
    }});
    if (!rows.length) {{ alert("还没有完成任何一条评分。"); return; }}
    const blob = new Blob([JSON.stringify(rows, null, 2)], {{ type: "application/json" }});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "ratings-" + rater + ".json";
    link.click();
  }});

  render();
}})();
</script>
</body>
</html>
"""


def render_rating_page(test: BlindTestSet) -> str:
    axes = [
        {"key": axis.value, "prompt": AXIS_PROMPTS[axis]}
        for axis in (
            RatingAxis.SPEAKER_SIMILARITY,
            RatingAxis.NATURALNESS,
            RatingAxis.CHARACTER_SIMILARITY,
            RatingAxis.AI_ARTIFACT_LEVEL,
        )
    ]
    payload = json.dumps(
        {
            "test_id": test.test_id,
            "items": [
                {
                    "item_id": item.item_id,
                    "audio_path": item.audio_path,
                    "text": item.text,
                }
                for item in test.items
            ],
        },
        ensure_ascii=False,
    )
    return _PAGE_TEMPLATE.format(
        test_id=test.test_id,
        count=len(test.items),
        instructions=test.instructions,
        payload=payload,
        axes=json.dumps(axes, ensure_ascii=False),
    )


# --------------------------------------------------------------------------------------
# webMUSHRA export
# --------------------------------------------------------------------------------------


def export_webmushra_config(test: BlindTestSet, run_paths: RunPaths) -> Path:
    """Emit a webMUSHRA-compatible YAML config for the same blind set.

    Offered because webMUSHRA is the established listening-test framework and some
    evaluators will already have it set up. The local page above stays the default: it
    needs no server, and it asks the four questions spec §18 specifies rather than a
    generic MUSHRA scale.
    """
    import yaml

    pages: list[dict[str, object]] = [
        {
            "type": "generic",
            "id": "intro",
            "name": "说明",
            "content": test.instructions.replace("\n", "<br>"),
        }
    ]
    for item in test.items:
        pages.append(
            {
                "type": "mushra",
                "id": item.item_id,
                "name": item.text or item.item_id,
                "content": "请为这条音频打分",
                "showWaveform": False,
                "enableLooping": True,
                "stimuli": {item.item_id: item.audio_path},
            }
        )

    config = {
        "testname": f"Character voice blind test · {test.test_id}",
        "testId": test.test_id,
        "bufferSize": 2048,
        "stopOnErrors": True,
        "showButtonPreviousPage": True,
        "remoteService": "service/write.php",
        "pages": pages,
    }
    path = run_paths.blind / "webmushra.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path
