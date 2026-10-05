"""Does the voice say every word? Synthesize LLM-style replies, transcribe, compare.

Text goes through the app's own path — ``normalize()`` then ``estimate_chunks()`` — so
the voice engine receives exactly what it would in a conversation. Each chunk is
synthesized by the GPT-SoVITS server (whatever weights it has loaded, or the ones given
with --gpt/--sovits), transcribed locally with FunASR, and scored two ways:

* CER  — character error rate against the text sent to TTS (punctuation ignored).
* PER  — the same on toneless pinyin. ASR often writes a correctly *spoken* word with a
         same-sounding character (摘桂 → 斋桂); PER does not count that as a miss, so it
         is the better measure of "was every word spoken".

    python scripts/eval_voice_coverage.py --label zero-shot
    python scripts/eval_voice_coverage.py --label finetuned \
        --gpt GPT_weights_v2ProPlus/cartethyia-e15.ckpt \
        --sovits SoVITS_weights_v2ProPlus/cartethyia_e8_s352.pth
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

import httpx
from cvai_core.paths import repo_root
from cvai_text_normalizer import normalize
from cvai_text_normalizer.chunking import estimate_chunks
from cvai_stt_providers.funasr_stt import FunASRSTTProvider
from pypinyin import lazy_pinyin

sys.stdout.reconfigure(encoding="utf-8")

#: Replies of the kind the LLM writes for her: short and long, questions, ellipses,
#: numbers, names from her world, a list. None of these are lines from the dataset.
SENTENCES = [
    "嗯……我在这里，漂泊者。",
    "今天的风有点大，你出门的时候记得多穿一件外套。",
    "谢谢你一直陪着我，我会把这份心意好好记住的。",
    "你问我最喜欢什么？大概是埃格拉小镇的青枝月桂沙拉吧。",
    "如果你累了，就先休息一会儿，我会在旁边守着你。",
    "我们明天早上八点出发，大概需要走三个小时才能到达拉古那。",
    "这件事情说来话长……不过既然你想听，我就慢慢告诉你。",
    "小心！前面好像有残象在徘徊，我们绕过去吧。",
    "虽然我已经尝不出味道了，但是看到大家吃得开心，我也会很高兴。",
    "贯彻本心是我的信条，所以无论发生什么，我都不会轻易放弃。",
    "你今天看起来心情不错，是遇到了什么好事吗？",
    "坎特蕾拉说过，真正的勇气不是不害怕，而是害怕的时候依然向前走。",
    "第一，先确认周围安全；第二，收集回音火种；第三，回到船上和大家会合。",
    "这里一共有十二座雕像，我们已经净化了其中的七座，还剩下五座。",
    "我曾经以为自己只是一段被困住的频率，直到你出现，才让我重新找回了自己的名字。",
    "嗯，约定好了，下次换我来保护你。",
    "唔……这个问题有点难，让我想一想再回答你，好吗？",
    "夜深了，早点休息吧，明天也要拜托你了。",
    "漂泊者，你相信命运吗？我曾经不相信，但现在我觉得，能遇见你也许就是命运吧。",
    "如果有一天你也陷入了困境，我一定会成为那把为你破开黑暗的剑。",
]

_KEEP = re.compile(r"[一-鿿A-Za-z0-9]")


def bare(text: str) -> str:
    return "".join(_KEEP.findall(text))


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, list[str]]:
    """Levenshtein distance plus the reference tokens that were deleted."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    missing, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            missing.append(ref[i - 1]); i -= 1
        else:
            j -= 1
    return d[n][m], missing[::-1]


def load_reference() -> tuple[str, str]:
    bank = json.loads((repo_root() / "voicepacks/cartethyia_cn/metadata/references.json")
                      .read_text(encoding="utf-8"))
    best = max((s for s in bank["samples"] if s["style"] == "neutral"),
               key=lambda s: s["quality_score"])
    return str(repo_root() / "voicepacks/cartethyia_cn" / best["audio_path"]), best["transcript"]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:9880")
    parser.add_argument("--gpt", help="GPT weights path, relative to the GPT-SoVITS folder")
    parser.add_argument("--sovits", help="SoVITS weights path, relative to the GPT-SoVITS folder")
    parser.add_argument("--out", type=Path, default=repo_root() / "runs" / "voice_eval")
    args = parser.parse_args()

    out = args.out / args.label
    out.mkdir(parents=True, exist_ok=True)
    ref_audio, ref_text = load_reference()
    stt = FunASRSTTProvider()

    async with httpx.AsyncClient(base_url=args.server, timeout=300) as tts:
        if args.gpt:
            (await tts.get("/set_gpt_weights", params={"weights_path": args.gpt})).raise_for_status()
        if args.sovits:
            (await tts.get("/set_sovits_weights", params={"weights_path": args.sovits})).raise_for_status()

        totals = {"chars": 0, "char_err": 0, "syll": 0, "syll_err": 0, "missing": 0}
        report = []
        for index, sentence in enumerate(SENTENCES, start=1):
            spoken_text, heard = "", ""
            for c, chunk in enumerate(estimate_chunks(normalize(sentence))):
                response = await tts.post("/tts", json={
                    "text": chunk, "text_lang": "zh", "ref_audio_path": ref_audio,
                    "prompt_text": ref_text, "prompt_lang": "zh", "text_split_method": "cut5",
                    "top_k": 15, "top_p": 1.0, "temperature": 1.0,
                    "repetition_penalty": 1.35, "seed": 1234 + index, "media_type": "wav",
                })
                response.raise_for_status()
                wav = out / f"{index:02d}_{c}.wav"
                wav.write_bytes(response.content)
                heard += (await stt.transcribe(wav)).text
                spoken_text += chunk

            ref_c, hyp_c = list(bare(spoken_text)), list(bare(heard))
            ref_p, hyp_p = lazy_pinyin(ref_c), lazy_pinyin(hyp_c)
            char_err, _ = edit_ops(ref_c, hyp_c)
            syll_err, missing = edit_ops(ref_p, hyp_p)
            totals["chars"] += len(ref_c); totals["char_err"] += char_err
            totals["syll"] += len(ref_p); totals["syll_err"] += syll_err
            totals["missing"] += len(missing)
            per = syll_err / max(1, len(ref_p))
            report.append({"sent": spoken_text, "heard": heard, "per": per, "missing": missing})
            flag = "OK " if per <= 0.05 else "BAD"
            print(f"{flag} {index:2d} PER {per:5.1%}  {spoken_text}")
            if per > 0.05:
                print(f"         heard: {heard}")
                if missing:
                    print(f"         missing syllables: {' '.join(missing)}")

    cer = totals["char_err"] / totals["chars"]
    per = totals["syll_err"] / totals["syll"]
    print(f"\n[{args.label}] CER {cer:.1%} · PER {per:.1%} · "
          f"{totals['missing']} of {totals['syll']} syllables missing "
          f"({totals['missing'] / totals['syll']:.1%}) · "
          f"{sum(r['per'] <= 0.05 for r in report)}/{len(report)} sentences fully spoken")
    (out / "report.json").write_text(json.dumps(
        {"label": args.label, "cer": cer, "per": per, **totals, "sentences": report},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
