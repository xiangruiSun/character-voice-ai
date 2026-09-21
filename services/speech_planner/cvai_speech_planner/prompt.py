"""Building the character's system prompt (spec §11).

Spec §11 is explicit that the LLM should not get a generic role-playing prompt, and
that V1 uses "system prompt + original dialogue examples + character lore + conversation
history" rather than fine-tuning. The load-bearing part of that list is the **original
dialogue examples**: personality adjectives tell a model what to aim for, real lines
show it. Everything here is arranged so those lines get the most room.

The prompt is written in Chinese. A Chinese-speaking character described in English
drifts toward translationese — the model keeps the register of the instructions, not the
register of the examples.
"""

from __future__ import annotations

import json
from typing import Sequence

from cvai_types import CharacterProfile, DialogueExample, LLMMessage, Role

#: Rules that apply to every character, regardless of profile. Kept separate from the
#: profile's own ``forbidden_behavior`` so a character author cannot accidentally delete
#: them by rewriting their list.
UNIVERSAL_RULES: tuple[str, ...] = (
    "只输出角色会说的话，不要有旁白、动作描写或解释。",
    "不要提及人工智能、模型、提示词或任何系统设定。",
    "不要使用表情符号或颜文字。",
)

_SECTION = "【{title}】\n{body}"


def render_system_prompt(
    profile: CharacterProfile,
    *,
    examples: Sequence[DialogueExample] = (),
    memory_facts: Sequence[str] = (),
    style_names: Sequence[str] | None = None,
    include_format_instructions: bool = True,
) -> str:
    """Assemble the system prompt for one turn."""
    parts: list[str] = [f"你是{profile.character_name}。请始终以她的身份说话。"]

    if profile.personality:
        parts.append(_SECTION.format(title="性格", body=_bullets(profile.personality)))
    if profile.background.strip():
        parts.append(_SECTION.format(title="背景", body=profile.background.strip()))
    if profile.world_knowledge:
        parts.append(
            _SECTION.format(
                title="你了解的世界",
                body=_bullets(profile.world_knowledge)
                # The list doubles as a boundary. Without this line the model happily
                # invents lore, which is the fastest way to break a character for
                # someone who knows the source.
                + "\n（以上之外的设定，你并不确定，不要以肯定语气断言。）",
            )
        )
    if profile.relationship_style.strip():
        parts.append(
            _SECTION.format(title="你和对方的关系", body=profile.relationship_style.strip())
        )

    habits = _speaking_habits(profile)
    if habits:
        parts.append(_SECTION.format(title="说话习惯", body=habits))

    forbidden = list(UNIVERSAL_RULES) + list(profile.forbidden_behavior)
    parts.append(_SECTION.format(title="绝对不要", body=_bullets(forbidden)))

    if memory_facts:
        parts.append(_SECTION.format(title="你记得的事", body=_bullets(memory_facts)))

    if examples:
        parts.append(
            _SECTION.format(title="原作台词（模仿这种语气，不要照抄）", body=_examples(examples))
        )

    if include_format_instructions:
        parts.append(_format_instructions(profile, style_names))

    return "\n\n".join(parts)


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item.strip()}" for item in items if item and item.strip())


def _speaking_habits(profile: CharacterProfile) -> str:
    habits = profile.speaking_habits
    lines: list[str] = []
    length = {
        "short": "句子偏短，很少长篇大论。",
        "medium": "句子长度适中。",
        "long": "句子偏长。",
    }.get(habits.typical_sentence_length, habits.typical_sentence_length)
    if length:
        lines.append(f"- {length}")
    if habits.frequent_expressions:
        lines.append("- 常用说法：" + "、".join(habits.frequent_expressions))
    if habits.verbal_tics:
        lines.append("- 口头习惯：" + "、".join(habits.verbal_tics))
    for habit in habits.punctuation_habits:
        lines.append(f"- {habit}")
    return "\n".join(lines)


def _examples(examples: Sequence[DialogueExample]) -> str:
    lines: list[str] = []
    for example in examples:
        if example.user.strip():
            lines.append(f"对方：{example.user.strip()}")
        label = f"{'（' + example.style + '）' if example.style != 'neutral' else ''}"
        lines.append(f"她{label}：{example.character.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_instructions(
    profile: CharacterProfile, style_names: Sequence[str] | None
) -> str:
    """Tell the model it is also directing the performance (spec §12).

    The user only ever hears ``text``. Everything else routes to the voice stack, and
    saying so in the prompt matters: a model that thinks the metadata will be read aloud
    writes stage directions into it.
    """
    styles = list(style_names or profile.available_styles)
    schema_hint = {
        "text": "她要说的话（只有这一项会被说出来）",
        "emotion": f"从这些里选一个：{'、'.join(styles)}",
        "emotion_intensity": "0 到 1 之间的小数",
        "speaking_rate": "very_slow / slow / slightly_slow / normal / slightly_fast / fast / very_fast",
        "volume_style": "whisper / soft / normal / loud / shout",
        "pause_style": "clipped / natural / drawn_out / hesitant",
        "ending_style": "falling / flat / rising / trailing_off / cut_off",
        "direction_note": "一句话说明你为什么这样演（不会被说出来）",
    }
    return _SECTION.format(
        title="输出格式",
        body=(
            "你不只是写台词，你同时是这句话的演出指导。\n"
            "只输出一个 JSON 对象，不要有其他文字、不要用代码块包裹：\n"
            + json.dumps(schema_hint, ensure_ascii=False, indent=2)
            + f"\n\ntext 控制在 {profile.llm.max_chars_per_reply} 字以内。"
            "\n只有 text 会被念出来，其余字段是给语音系统的演出指示。"
        ),
    )


def build_messages(
    profile: CharacterProfile,
    user_text: str,
    *,
    history: Sequence[LLMMessage] = (),
    examples: Sequence[DialogueExample] = (),
    memory_facts: Sequence[str] = (),
    style_names: Sequence[str] | None = None,
) -> list[LLMMessage]:
    """System prompt + trimmed history + the new user turn."""
    messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content=render_system_prompt(
                profile,
                examples=examples,
                memory_facts=memory_facts,
                style_names=style_names,
            ),
        )
    ]
    messages.extend(history)
    messages.append(LLMMessage(role=Role.USER, content=user_text))
    return messages


def trim_history(
    history: Sequence[LLMMessage], max_turns: int
) -> list[LLMMessage]:
    """Keep the most recent ``max_turns`` exchanges.

    Trimming from the front, not the back: the recent turns carry the conversational
    state, and the character's identity lives in the system prompt rather than in old
    messages. Milestone 8's summarization replaces what is dropped here.
    """
    if max_turns <= 0:
        return []
    keep = max_turns * 2  # a user message and an assistant message per turn
    return list(history[-keep:])
