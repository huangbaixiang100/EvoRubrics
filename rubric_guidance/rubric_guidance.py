"""EvoRubrics training module."""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

from rubric_guidance.prompts import (
    FOLLOWBENCH_RUBRICS_GENERATOR_SYSTEM_PROMPT,
    MTBENCH_RUBRICS_GENERATOR_SYSTEM_PROMPT,
    UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT,
)


def build_rubrics_user_prompt(question: str) -> str:
    q = (question or "").strip()
    return (
        "Generate 9 to 11 rubrics that are SPECIFIC to and DERIVED FROM the following healthcare question "
        "(OUTPUT RUBRICS ONLY):\n\n"
        f"{q}"
    )


def build_rubric_guided_query(question: str, generated_rubrics: str) -> str:
    """English docstring."""
    rubrics_block = (generated_rubrics or "").strip()
    return (
        f"{(question or '').strip()}\n\n"
        "Below are evaluation-style rubrics for this question. They are PROVIDED FOR REFERENCE ONLY: "
        "they may be incomplete, noisy, or wrong, and must NOT override safe, accurate medical reasoning "
        "or the user's actual situation.\n\n"
        "How to use them: draft the answer you would normally give, then skim the rubrics as an optional "
        "self-check—see if anything suggests a useful clarification or omission. "
        "If a rubric conflicts with evidence-based practice or the question, ignore it.\n\n"
        "[Rubrics (reference only)]\n"
        f"{rubrics_block}\n\n"
        "WHEN USING RUBRICS TO POLISH (OPTIONAL):\n"
        "- Higher absolute point values suggest stronger emphasis if you choose to align with the checklist.\n"
        "- Negative-point lines describe common gaps or pitfalls; treat them as hints, not mandatory accusations.\n"
        "- Positive-point lines describe strengths you might reinforce if they fit your answer.\n"
        "- If rubrics disagree with each other or with your judgment, prefer accuracy and appropriate uncertainty.\n\n"
        "Now provide your final response to the user query directly (your own clinical reasoning comes first)."
    )


def build_open_rubrics_chat_messages(
    prompt_messages: List[Dict[str, Any]] | None,
    instruction: str = "",
    fallback_question: str = "",
) -> List[Dict[str, str]]:
    """English docstring."""
    base: List[Dict[str, str]] = [
        {"role": "system", "content": UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT}
    ]
    inst = (instruction or "").strip()
    if inst:
        base.append({"role": "system", "content": inst})
    raw = prompt_messages or []
    if raw:
        for msg in raw:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role not in ("user", "assistant", "system") or not isinstance(content, str):
                continue
            base.append({"role": role, "content": content})
    elif (fallback_question or "").strip():
        base.append({"role": "user", "content": fallback_question.strip()})
    return base


_MTBENCH_FOLLOWUP_COMPUTE_CATEGORIES = frozenset({"reasoning", "math", "coding", "stem"})


def _is_followup_compute_turn(category: str, turn_index: int) -> bool:
    return turn_index >= 1 and (category or "").strip().lower() in _MTBENCH_FOLLOWUP_COMPUTE_CATEGORIES


_OPEN_ENDED_CATEGORIES = frozenset({"writing", "humanities", "roleplay"})
_POINTS_RE = re.compile(r"^points:\s*(-?\d+)\s*$", re.IGNORECASE)
_FIELD_POINTS_RE = re.compile(r"^points:\s*([+-]?\d+)\s*$", re.IGNORECASE)
_FIELD_CRITERION_RE = re.compile(r"^criterion:\s*(.+?)\s*$", re.IGNORECASE)
_FIELD_WHY_RE = re.compile(r"^why:\s*(.+?)\s*$", re.IGNORECASE)
_RUBRIC_LINE_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")
_DIRTY_GUIDANCE_RE = re.compile(
    r"\b("
    r"rubric|criterion|criteria|points?|score|scoring|judge|evaluation|"
    r"output only|numbered rubrics|two lines|positive and negative|"
    r"mandatory rules?|required output format"
    r")\b",
    re.IGNORECASE,
)


def _parse_mtbench_rubrics(generated_rubrics: str) -> List[Tuple[str, int]]:
    """Parse the MT-Bench two-line rubric format into (criterion, points)."""
    lines = [
        ln.strip()
        for ln in (generated_rubrics or "").splitlines()
        if ln.strip()
    ]
    parsed: List[Tuple[str, int]] = []
    i = 0
    while i < len(lines):
        crit_m = _RUBRIC_LINE_RE.match(lines[i])
        if not crit_m:
            i += 1
            continue
        criterion = crit_m.group(1).strip()
        points = 1
        if i + 1 < len(lines):
            points_m = _POINTS_RE.match(lines[i + 1])
            if points_m:
                points = int(points_m.group(1))
                i += 2
            else:
                i += 1
        else:
            i += 1
        if criterion:
            parsed.append((criterion, points))
    return parsed


def _parse_mtbench_structured_rubrics(
    generated_rubrics: str,
) -> List[Tuple[str, int, str]]:
    """Parse criterion/points/why rubrics from the MT-Bench rubric writer."""
    rows: List[Tuple[str, int, str]] = []
    criterion = ""
    points: int | None = None
    why = ""

    def flush() -> None:
        nonlocal criterion, points, why
        if criterion and points is not None:
            rows.append((criterion.strip(), int(points), why.strip()))
        criterion = ""
        points = None
        why = ""

    for raw in (generated_rubrics or "").splitlines() + [""]:
        line = raw.strip()
        if not line:
            flush()
            continue
        cm = _FIELD_CRITERION_RE.match(line)
        if cm:
            if criterion or points is not None or why:
                flush()
            criterion = cm.group(1).strip()
            continue
        pm = _FIELD_POINTS_RE.match(line)
        if pm:
            try:
                points = int(pm.group(1))
            except ValueError:
                points = None
            continue
        wm = _FIELD_WHY_RE.match(line)
        if wm:
            why = wm.group(1).strip()
            continue

    return rows


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _clean_guidance_item(item: str) -> str:
    s = re.sub(r"\s+", " ", (item or "").strip())
    s = re.sub(r"^(?:criterion|criteria)\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*points?\s*:\s*-?\d+\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip(" -;")
    if not s or _DIRTY_GUIDANCE_RE.search(s):
        return ""
    if len(s.split()) < 3:
        return ""
    return s


def _rubric_note_from_fields(criterion: str, points: int, why: str) -> str:
    crit = _clean_guidance_item(criterion)
    if not crit:
        return ""
    reason = _clean_guidance_item(why)
    if points < 0:
        lowered = crit.lower()
        if not any(
            token in lowered
            for token in (
                "fail",
                "omit",
                "incorrect",
                "wrong",
                "ignore",
                "violate",
                "extra",
                "unnecessary",
                "unsupported",
                "hallucinat",
            )
        ):
            return ""
        note = f"Avoid this pitfall: {crit}"
    else:
        note = crit
    if reason:
        note = f"{note} ({reason})"
    return note


def _extract_guidance_items(generated_rubrics: str) -> List[str]:
    """Read the newer bullet guidance format, tolerating numbered bullets."""
    items: List[str] = []
    for raw in (generated_rubrics or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _POINTS_RE.match(line):
            continue
        m = _BULLET_LINE_RE.match(line)
        candidate = m.group(1) if m else line
        cleaned = _clean_guidance_item(candidate)
        if cleaned:
            items.append(cleaned)
    return _dedupe_keep_order(items)


def _fallback_mtbench_guidance(category: str, turn_index: int) -> List[str]:
    cat = (category or "").strip().lower()
    base: List[str] = []
    if turn_index >= 1:
        base.append(
            "Follow the latest user instruction first; use prior turns only "
            "as context."
        )
    if cat in {"math", "reasoning"}:
        base.append(
            "Solve the latest question directly, check the reasoning, and "
            "state the final answer clearly."
        )
    elif cat == "coding":
        base.append(
            "Match the latest implementation constraint and provide correct, "
            "runnable code."
        )
    elif cat == "extraction":
        base.append(
            "Answer the latest extraction request concisely and preserve only "
            "the format explicitly requested now."
        )
    elif cat == "stem":
        base.append(
            "Prioritize factual correctness and explain the key mechanism or "
            "calculation requested."
        )
    elif cat == "roleplay":
        base.append(
            "Stay in the requested role while still answering the latest "
            "question naturally."
        )
    elif cat == "writing":
        base.append(
            "Produce the requested creative text directly, preserving style "
            "and formatting constraints."
        )
    elif cat == "humanities":
        base.append(
            "Give a clear, nuanced answer with enough concrete detail to be "
            "useful."
        )
    else:
        base.append("Answer the latest user request directly and accurately.")
    return base


def build_mtbench_answer_checklist(
    generated_rubrics: str,
    *,
    category: str = "",
    turn_index: int = 0,
    user_turn: str = "",
) -> str:
    """Compress evaluation rubrics into a short task checklist for answer generation.

    The policy model should see practical answer guidance, not the full scoring rubric.
    Precision-heavy categories get fewer, stricter items; open-ended categories can use
    a little more guidance for style/depth/creativity.
    """
    cat = (category or "").strip().lower()
    latest = (user_turn or "").strip().lower()
    structured = _parse_mtbench_structured_rubrics(generated_rubrics)
    if structured:
        ordered = sorted(structured, key=lambda x: abs(x[1]), reverse=True)
        positive_notes = [
            _rubric_note_from_fields(c, p, w)
            for c, p, w in ordered
            if p > 0
        ]
        negative_notes = [
            _rubric_note_from_fields(c, p, w)
            for c, p, w in ordered
            if p < 0
        ]
        notes = [x for x in positive_notes + negative_notes if x]
        limit = 5 if cat in _OPEN_ENDED_CATEGORIES else 4
        if turn_index >= 1:
            notes.insert(
                0,
                "Follow the latest user instruction first; use prior turns "
                "only as context.",
            )
        notes = _dedupe_keep_order(notes)
        if notes:
            return "\n".join(f"- {x}" for x in notes[:limit])

    has_point_format = bool(
        re.search(r"(?im)^\s*points?\s*:\s*-?\d+\s*$", generated_rubrics or "")
    )
    guidance_items = [] if has_point_format else _extract_guidance_items(
        generated_rubrics
    )
    if guidance_items:
        limit = 4 if cat in _OPEN_ENDED_CATEGORIES else 3
        items = guidance_items[:limit]
        if turn_index >= 1:
            items.insert(
                0,
                "Follow the latest user instruction first; use prior turns "
                "only as context.",
            )
        return "\n".join(f"- {x}" for x in _dedupe_keep_order(items)[:limit])

    parsed = _parse_mtbench_rubrics(generated_rubrics)
    if not parsed:
        fallback = [
            re.sub(r"^\s*\d+\.\s*", "", ln).strip()
            for ln in (generated_rubrics or "").splitlines()
            if ln.strip() and not ln.lower().startswith("points:")
        ]
        parsed = [(x, 1) for x in fallback[:5]]

    positives = [(_clean_guidance_item(c), p) for c, p in parsed if p > 0]
    positives = [(c, p) for c, p in positives if c]
    positives = sorted(positives, key=lambda x: abs(x[1]), reverse=True)

    items: List[str] = []
    if cat == "extraction":
        explicit_format = any(
            token in latest
            for token in (
                "json",
                "csv",
                "yaml",
                "xml",
                "dictionary",
                "line-by-line",
                "format",
                "no extra",
                "extraneous",
                "list only",
            )
        )
        items.append(
            "Answer the latest extraction request directly; keep the output "
            "concise and do not carry over old constraints unless asked."
        )
        keywords = (
            "format",
            "json",
            "csv",
            "yaml",
            "field",
            "list",
            "count",
            "entity",
            "exact",
        )
        if explicit_format:
            items.insert(
                0,
                "Preserve the exact output format requested by the latest "
                "user message; avoid extra prose.",
            )
            items.extend(
                c
                for c, _ in positives
                if any(k in c.lower() for k in keywords)
            )
        else:
            items.extend(
                c
                for c, _ in positives
                if not any(k in c.lower() for k in keywords)
            )
        limit = 3
    elif cat in {"math", "reasoning"}:
        items.append(
            "Solve the actual question directly, verify the reasoning, "
            "and state the final answer clearly."
        )
        items.extend(c for c, _ in positives)
        limit = 4
    elif cat == "coding":
        items.append(
            "Provide a correct, runnable solution that matches the requested "
            "language/API and handles edge cases."
        )
        items.extend(c for c, _ in positives)
        limit = 4
    elif cat == "stem":
        items.append(
            "Prioritize factual correctness, terminology, and the "
            "causal/mechanistic explanation requested."
        )
        items.extend(c for c, _ in positives)
        limit = 4
    elif cat in _OPEN_ENDED_CATEGORIES:
        items.extend(c for c, _ in positives)
        limit = 5
    else:
        items.extend(c for c, _ in positives)
        limit = 4

    if turn_index >= 1:
        items.insert(
            0,
            "Follow the latest user instruction first; use prior turns only "
            "as context.",
        )

    items = _dedupe_keep_order(items)
    if not items:
        items = _fallback_mtbench_guidance(cat, turn_index)
    return "\n".join(f"- {item}" for item in items[:limit])


def build_mtbench_rubric_guided_query(
    user_turn: str,
    generated_rubrics: str,
    *,
    category: str = "",
    turn_index: int = 0,
    guidance_mode: str = "checklist",
) -> str:
    """General-assistant rubric guidance (MT-Bench / open-domain)."""
    rubrics_block = (generated_rubrics or "").strip()
    checklist = build_mtbench_answer_checklist(
        rubrics_block,
        category=category,
        turn_index=turn_index,
        user_turn=user_turn,
    )
    cat = (category or "").strip()
    cat_line = f"Task category: {cat}\n\n" if cat else ""
    turn_hint = ""
    if _is_followup_compute_turn(cat, turn_index):
        turn_hint = (
            "Follow-up turn (reasoning/math/coding): answer the NEW "
            "question directly. Use prior turns as context. Show correct "
            "reasoning or working; give a clear final answer.\n\n"
        )
    elif turn_index >= 1:
        turn_hint = (
            "Follow-up turn: follow the NEW user instruction "
            "(rewrite, reformat, or new subtask). "
            "Prior-turn content is context.\n\n"
        )
    if guidance_mode == "raw":
        guidance_block = (
            "Below are evaluation-style rubrics. They are PROVIDED FOR "
            "REFERENCE ONLY: "
            "they may be incomplete, noisy, or wrong.\n\n"
            "[Rubrics (reference only)]\n"
            f"{rubrics_block}\n\n"
            "Optional polish: skim rubrics after drafting your normal "
            "answer; reinforce helpful items only if they fit. "
            "Negative-point lines are common pitfalls, not accusations.\n\n"
        )
    elif guidance_mode == "cot":
        guidance_block = (
            "Use these supplemental rubric notes as a private silent "
            "self-check. They are secondary to the user's actual request and "
            "may be ignored if they are noisy, wrong, or conflict with "
            "correctness/latest instruction.\n\n"
            "[Supplemental rubric notes]\n"
            f"{checklist or '- Answer the user request directly and accurately.'}\n\n"
            "Private answer procedure (do this silently, do not output it):\n"
            "1. Identify the latest user's hard requirements, especially "
            "format, length, style, role, and exact-answer constraints.\n"
            "2. Draft the answer mentally.\n"
            "3. Check the draft against the supplemental rubric notes.\n"
            "4. Fix only concrete issues that improve the answer without "
            "changing the requested format or adding extra prose.\n"
            "5. Ignore any rubric note that conflicts with the latest user "
            "instruction, factual correctness, concision, or exact output "
            "format.\n"
            "6. Output only the final answer.\n\n"
        )
    else:
        guidance_block = (
            "Before answering, skim these rubric-derived notes. They are "
            "secondary to the user's actual request and may be ignored if "
            "they are noisy, wrong, or conflict with correctness/latest "
            "instruction.\n\n"
            "[Supplemental rubric notes]\n"
            f"{checklist or '- Answer the user request directly and accurately.'}\n\n"
        )

    return (
        f"{cat_line}"
        f"{turn_hint}"
        f"{(user_turn or '').strip()}\n\n"
        "Priority order:\n"
        "1. The user's explicit instruction above (always wins).\n"
        "2. Accurate reasoning and facts.\n"
        "3. The supplemental rubrics as optional guidance—ignore anything "
        "that conflicts with (1) or (2).\n\n"
        "Do NOT reveal or quote the rubrics. Do not mention any private "
        "reasoning, checklist, self-check, or evaluation process. Reply as if "
        "answering the user normally.\n\n"
        f"{guidance_block}"
        "Provide your final response to the user directly. No preamble."
    )


_MTBENCH_CATEGORY_HINTS: Dict[str, str] = {
    "writing": "Creative writing, emails, blogs; judge tone, structure, creativity, and exact format constraints.",
    "humanities": "Humanities Q&A; judge accuracy, depth, nuance, and citation-style reasoning.",
    "roleplay": "Stay in character; judge relevance, consistency, and new roleplay instructions in later turns.",
    "reasoning": "Logic puzzles; judge correct reasoning steps, explanation depth, and final answer.",
    "stem": "Science/tech; judge factual accuracy, explanation depth, and terminology.",
    "math": "Judge mathematical correctness, clear derivation, and final numeric/symbolic answer.",
    "coding": "Judge code correctness, edge cases, and match to requested language/API.",
    "extraction": "Judge exact output format (JSON/CSV/YAML), field completeness, and no extra prose.",
}


def build_mtbench_rubrics_user_prompt(
    *,
    category: str,
    user_turn: str,
    turn_index: int,
    prior_turns: List[tuple[str, str]] | None = None,
) -> str:
    """User message for MT-Bench guidance model: latest task + context."""
    cat = (category or "general").strip()
    cat_hint = _MTBENCH_CATEGORY_HINTS.get(cat, "General assistant task.")
    lines = [
        f"Category: {cat}",
        f"Category focus: {cat_hint}",
        f"Turn index (0-based): {turn_index}",
        "",
    ]
    if prior_turns:
        lines.append("Conversation history (prior turns):")
        for i, (u, a) in enumerate(prior_turns, start=1):
            lines.append(f"[User turn {i}]\n{u.strip()}")
            lines.append(f"[Assistant turn {i}]\n{a.strip()}")
        lines.append("")
    lines.append("Latest user message (what the assistant must answer now):")
    lines.append(user_turn.strip())
    lines.append("")
    if turn_index >= 1 and _is_followup_compute_turn(cat, turn_index):
        lines.append(
            "TURN-2+ FOLLOW-UP (reasoning/math/coding/stem): rubrics should "
            "judge the answer to the NEW question above, not a generic "
            "rewrite of the previous answer.\n"
            "- Include criteria for correct final answer, valid reasoning or "
            "derivation, and relevant use of prior context.\n"
            "- Do NOT require JSON/CSV/letter-start/limerick/boxed output "
            "unless the latest user message says so."
        )
    elif turn_index >= 1:
        lines.append(
            "TURN-2+ TASK: rubrics must target the latest user message only "
            "(rewrite, rephrase, new format, self-critique, or new subtask).\n"
            "- Preserve prior content only when the latest user message asks "
            "to transform it.\n"
            "- Do NOT invent format rules absent from the latest message."
        )
    else:
        lines.append(
            "TURN-1 TASK: rubrics should judge whether the assistant answers "
            "the first user message well for this category."
        )
    lines.append("")
    lines.append(
        "Generate 5 to 7 rubrics. Each rubric must use exactly these fields:\n"
        "criterion: <concrete judgment criterion>\n"
        "points: <integer -10 to 10, not 0>\n"
        "why: <why this matters for this latest task>\n\n"
        "Use the category priors from the system prompt. Keep criteria "
        "specific to the latest user message."
    )
    return "\n".join(lines)


def build_mtbench_rubrics_messages(
    *,
    category: str,
    user_turn: str,
    turn_index: int,
    prior_turns: List[tuple[str, str]] | None = None,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": MTBENCH_RUBRICS_GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_mtbench_rubrics_user_prompt(
                category=category,
                user_turn=user_turn,
                turn_index=turn_index,
                prior_turns=prior_turns,
            ),
        },
    ]


def apply_rubric_guidance(
    policy_messages: List[Dict[str, str]],
    rubrics_text: str,
    *,
    fallback_question: str,
) -> List[Dict[str, str]]:
    """English docstring."""
    msgs = copy.deepcopy(policy_messages)
    rt = (rubrics_text or "").strip()
    if not rt:
        return msgs
    merged = False
    for j in range(len(msgs) - 1, -1, -1):
        if msgs[j].get("role") == "user":
            msgs[j]["content"] = build_rubric_guided_query(
                str(msgs[j].get("content", "") or ""),
                rt,
            )
            merged = True
            break
    if not merged:
        msgs.append(
            {
                "role": "user",
                "content": build_rubric_guided_query(
                    (fallback_question or "").strip(),
                    rt,
                ),
            }
        )
    return msgs


_FIELD_CONSTRAINT_RE = re.compile(r"^constraint:\s*(.+?)\s*$", re.IGNORECASE)
_FIELD_CHECK_RE = re.compile(r"^check:\s*(.+?)\s*$", re.IGNORECASE)

_FOLLOWBENCH_CATEGORY_HINTS: Dict[str, str] = {
    "content": "Content constraints: required fields, labels, entities, sentiment, counts.",
    "format": "Format constraints: structure, length, rhyme/meter, alphabet order, templates.",
    "style": "Style constraints: tone, voice, diction, rhetorical devices.",
    "situation": "Situation constraints: role, audience, setting, communicative purpose.",
    "mixed": "Mixed constraints from multiple categories — split each requirement.",
    "example": "Example-following: match the demonstrated output pattern exactly.",
}


def _parse_followbench_constraints(generated_rubrics: str) -> List[Tuple[str, str]]:
    lines = [
        ln.strip()
        for ln in (generated_rubrics or "").splitlines()
        if ln.strip()
    ]
    parsed: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        c_m = _FIELD_CONSTRAINT_RE.match(lines[i])
        if not c_m:
            i += 1
            continue
        constraint = c_m.group(1).strip()
        check = ""
        if i + 1 < len(lines):
            k_m = _FIELD_CHECK_RE.match(lines[i + 1])
            if k_m:
                check = k_m.group(1).strip()
                i += 2
            else:
                i += 1
        else:
            i += 1
        if constraint and not _DIRTY_GUIDANCE_RE.search(constraint):
            parsed.append((constraint, check))
    return parsed


def build_followbench_answer_checklist(
    generated_rubrics: str,
    *,
    constraint_type: str = "",
) -> str:
    """Turn FollowBench constraint rubrics into a short verification checklist."""
    structured = _parse_followbench_constraints(generated_rubrics)
    items: List[str] = []
    if structured:
        new_first = sorted(
            structured,
            key=lambda x: (0 if x[0].upper().startswith("[NEW]") else 1, x[0]),
        )
        for constraint, check in new_first:
            text = constraint
            if check:
                text = f"{constraint} ({check})"
            cleaned = _clean_guidance_item(text)
            if cleaned:
                items.append(cleaned)
    if not items:
        items = _extract_guidance_items(generated_rubrics)
    if not items:
        cat = (constraint_type or "").strip().lower()
        if cat == "example":
            items = [
                "Match the demonstrated answer format from the examples in the prompt.",
                "Answer only the final test instance; do not repeat the demo prefix.",
            ]
        else:
            items = [
                "Read the full instruction and satisfy every explicit requirement.",
                "Do not add requirements that were not asked for.",
            ]
    items = _dedupe_keep_order(items)
    return "\n".join(f"- {item}" for item in items[:8])


def build_followbench_rubric_guided_query(
    user_prompt: str,
    generated_rubrics: str,
    *,
    constraint_type: str = "",
    guidance_mode: str = "checklist",
) -> str:
    rubrics_block = (generated_rubrics or "").strip()
    checklist = build_followbench_answer_checklist(
        rubrics_block,
        constraint_type=constraint_type,
    )
    cat = (constraint_type or "").strip()
    cat_line = f"Constraint type: {cat}\n\n" if cat else ""
    if guidance_mode == "raw":
        guidance_block = (
            "Reference constraints (may be incomplete or wrong):\n\n"
            f"{rubrics_block}\n\n"
        )
    else:
        guidance_block = (
            "Private constraint checklist — verify your answer against every item:\n\n"
            f"{checklist}\n\n"
        )
    return (
        f"{cat_line}"
        f"{(user_prompt or '').strip()}\n\n"
        "Priority order:\n"
        "1. The user instruction above (always wins).\n"
        "2. Satisfy every explicit constraint in the instruction.\n"
        "3. Use the checklist below as a final verification pass.\n\n"
        "Do NOT mention the checklist or evaluation process.\n\n"
        f"{guidance_block}"
        "Provide your final response directly. No preamble."
    )


def _truncate_text(text: str, max_chars: int) -> str:
    """Keep head + tail so long FollowBench prompts (e.g. email demos) stay within context."""
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n...[truncated for length]...\n\n"
    if max_chars <= len(marker) + 80:
        return text[:max_chars]
    tail_budget = max(max_chars * 3 // 4, max_chars - 800)
    head_budget = max_chars - tail_budget - len(marker)
    if head_budget < 200:
        head_budget = 200
        tail_budget = max_chars - head_budget - len(marker)
    return f"{text[:head_budget]}{marker}{text[-tail_budget:]}"


def build_followbench_rubrics_user_prompt(
    *,
    constraint_type: str,
    user_prompt: str,
    level: int = 0,
    example_id: int = 0,
    instruction_prev: str = "",
    max_instruction_chars: int = 6000,
    max_instruction_prev_chars: int = 2000,
) -> str:
    cat = (constraint_type or "general").strip()
    cat_hint = _FOLLOWBENCH_CATEGORY_HINTS.get(cat, "General constraint-following task.")
    instruction = _truncate_text(user_prompt, max_instruction_chars)
    prev = instruction_prev.strip()
    if prev:
        if len(user_prompt.strip()) > max_instruction_chars // 2:
            prev = ""
        else:
            prev = _truncate_text(prev, max_instruction_prev_chars)
    lines = [
        f"Constraint type: {cat}",
        f"Category focus: {cat_hint}",
        f"Example id: {example_id}",
        f"Level: {level}",
        "",
        "Instruction the assistant must follow:",
        instruction,
        "",
    ]
    if prev:
        lines.extend(
            [
                "Previous level instruction (context only — current instruction is cumulative):",
                prev,
                "",
                f"Focus extra attention on constraints newly introduced at level {level}.",
                "Mark newly introduced constraints with [NEW] at the start of the constraint field.",
                "",
            ]
        )
    lines.append(
        "List 4 to 8 verifiable constraints using exactly:\n"
        "constraint: <requirement>\n"
        "check: <how to verify>\n\n"
        "Include ALL explicit requirements from the current instruction."
    )
    return "\n".join(lines)


def build_followbench_rubrics_messages(
    *,
    constraint_type: str,
    user_prompt: str,
    level: int = 0,
    example_id: int = 0,
    instruction_prev: str = "",
    max_instruction_chars: int = 6000,
    max_instruction_prev_chars: int = 2000,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": FOLLOWBENCH_RUBRICS_GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_followbench_rubrics_user_prompt(
                constraint_type=constraint_type,
                user_prompt=user_prompt,
                level=level,
                example_id=example_id,
                instruction_prev=instruction_prev,
                max_instruction_chars=max_instruction_chars,
                max_instruction_prev_chars=max_instruction_prev_chars,
            ),
        },
    ]
