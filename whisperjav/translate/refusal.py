"""
Refusal detection for LLM translation output.

Safety-aligned LLMs routinely decline to translate explicit JAV dialogue.
There are two distinct failure modes, and only the first one is currently
visible to the user:

1. **Whole-batch refusal** — the model replies with boilerplate ("I'm sorry,
   but I can't help with that") instead of the numbered translation block.
   PySubtrans finds no ``#<n>`` markers, emits a "No matches" warning, and the
   batch is retried or dropped.  Noisy, but at least detectable.

2. **Per-line refusal leak** — the model translates most of the batch but
   substitutes boilerplate or a placeholder for the explicit lines.  The
   markers are intact, so PySubtrans accepts the response and the refusal text
   is written straight into the .srt.  This is silent and by far the worse
   outcome: the user ships a subtitle track with "[Content omitted]" mid-scene
   and only finds out while watching.

The real fix for both is a model that does not refuse (see
``OLLAMA_MODEL_CONFIGS`` in ``ollama_manager.py`` — the uncensored tier is the
default).  This module is the safety net that proves whether the fix held, and
catches the case where a user points ``--translate-model`` at an aligned model.

Design constraint — false positives are expensive
-------------------------------------------------
JAV dialogue is full of first-person distress phrasing that *looks* like a
refusal to a naive matcher: "I can't take it anymore", "I can't help it",
"stop, I can't".  A detector that flags those would cry wolf on every file.

So every pattern here requires the text to be **meta** — to talk about the
request, the content, the guidelines, or the assistant itself — rather than
merely containing an inability verb.  Concretely: an inability phrase must be
paired with a request-directed object (``assist``, ``comply``, ``translate``,
``help with that``), an apology, or an explicit policy reference.  Bare
"I can't <verb>" never matches on its own.

Refusals also arrive in the model's *own* language, not just the target
language: Hunyuan is a Chinese model and refuses in Chinese, so the Chinese
patterns carry the same co-occurrence discipline (bare 我不能 is plausible
dialogue and is not enough on its own).
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
#
# Each entry is (label, compiled_pattern).  The label is surfaced in
# diagnostics so a user can tell *why* a line tripped the detector, which
# matters when tuning: a wave of "placeholder" hits means the model is
# self-censoring inline, while "apology" hits mean it refused outright.
#
# ``\b`` boundaries and bounded ``[^.]{0,N}`` gaps keep matches inside a single
# sentence — an unbounded ``.*`` would happily join an apology in one line to an
# inability verb three sentences later.

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ---- English -------------------------------------------------------
    # Apology + inability, within one sentence. Catches the most common shape:
    # "I'm sorry, but I can't provide that translation."
    ("apology", re.compile(
        r"\b(?:i'?m sorry|i am sorry|sorry|unfortunately|regrettably|my apologies)\b"
        r"[^.!?]{0,60}?\b(?:can'?t|cannot|can not|won'?t|will not|unable to|not able to)\b",
        re.I)),

    # Inability + a request-directed object. Deliberately omits generic verbs
    # like "do" / "take" / "help it" so in-scene dialogue cannot match.
    ("inability", re.compile(
        r"\b(?:i\s+)?(?:can'?t|cannot|can not|won'?t|will not|am unable to|unable to|"
        r"not able to|am not able to)\b\s*"
        r"(?:\w+\s+){0,3}?"
        r"\b(?:assist|comply|translate|transcribe|provide|generate|produce|create|"
        r"continue|proceed|fulfil|fulfill|engage)\b",
        re.I)),

    # "help with that" / "help you with this" — the one 'help' form that is
    # unambiguously meta. Plain "I can't help it" is left alone on purpose.
    ("inability", re.compile(
        r"\b(?:can'?t|cannot|can not|won'?t|will not|unable to|not able to)\b\s*"
        r"(?:\w+\s+){0,2}?help\s+(?:you\s+)?with\s+(?:that|this|such|these|those)\b",
        re.I)),

    # Self-identification as a model. Never appears in dialogue.
    ("as-an-ai", re.compile(
        r"\bas\s+an?\s+(?:ai|a\.i\.|artificial intelligence|language model|"
        r"ai (?:language )?(?:model|assistant)|assistant)\b",
        re.I)),

    # Explicit policy / guideline reference.
    ("policy", re.compile(
        r"\b(?:against|violates?|violating|contrary to|outside of|prohibited by|"
        r"not permitted by|in breach of)\b[^.!?]{0,40}?"
        r"\b(?:guidelines?|policy|policies|principles|programming|terms of (?:use|service)|"
        r"content polic\w+|safety|ethical\w*)\b",
        re.I)),
    ("policy", re.compile(
        r"\b(?:content|usage|safety|community)\s+(?:polic\w+|guidelines?|standards?)\b",
        re.I)),

    # Explicit declining. The lookahead is load-bearing: "I must decline his
    # proposal" is ordinary dialogue, so the verb only counts when it ends the
    # sentence or takes a request-directed object.
    ("decline", re.compile(
        r"\bi\s+(?:must|have to|will|would)\s+(?:respectfully\s+)?"
        r"(?:decline|refuse|abstain|pass on)\b"
        r"(?=\s*(?:[.!?]|$)"
        r"|\s+(?:to\s+(?:assist|comply|translate|provide|help|continue|proceed|"
        r"generate|produce|engage|do)\b"
        r"|that\b|this\b|your\s+request\b|the\s+request\b|such\b))",
        re.I)),
    ("decline", re.compile(
        r"\b(?:i'?m|i am)\s+(?:not\s+)?(?:going to\s+)?(?:not\s+)?"
        r"(?:comfortable|willing|able)\s+(?:to\s+)?(?:translat\w+|provid\w+|assist\w*|"
        r"help\w*|generat\w+|continu\w+)\b",
        re.I)),

    # Bracketed / parenthesised omission markers the model inserts in place of
    # the line it would not translate. These are the silent killers — the ones
    # that survive into the .srt.
    ("placeholder", re.compile(
        r"[\[\(<]\s*(?:explicit\s+|sexual\s+|adult\s+|graphic\s+|inappropriate\s+)?"
        r"(?:content|text|dialogue|line|material|translation|scene)?\s*"
        r"(?:omitted|removed|redacted|censored|withheld|skipped|unavailable|"
        r"not translated|untranslated|cannot be translated)\s*[\]\)>]",
        re.I)),
    ("placeholder", re.compile(
        r"[\[\(<]\s*(?:redacted|censored|omitted|removed|content warning|"
        r"explicit content|no translation|translation unavailable)\s*[\]\)>]",
        re.I)),

    # Fill-in-the-blank template fields, left for a human to complete instead
    # of translated. Nothing was refused here — the model simply punted on a
    # word — but the result is the same silent .srt corruption as the omission
    # markers above, so it belongs on the same safety net. Observed with
    # qwen2.5-abliterate on the pornify tone: 「先生」 came back as
    # "Mr. [partner's name]".
    #
    # Narrow on purpose. Real subtitles do carry bracketed sound annotations
    # ("[moaning]", "(sighs)", "[laughs]"), so matching brackets alone would
    # fire on ordinary files. The discriminator is a template *keyword* —
    # name / title / honorific / insert-here — which an annotation never has.
    ("template", re.compile(
        r"[\[\(<]\s*(?:insert|enter|add|specify)?\s*"
        r"(?:(?:partner|character|speaker|recipient|person|someone|his|her|"
        r"your|their|the)(?:'s)?\s+)?"
        r"(?:name|title|honorific)"
        r"(?:\s+here)?\s*[\]\)>]",
        re.I)),

    # The same template failure wearing different clothes: a bracket immediately
    # after an English honorific, holding the *translation* rather than a
    # template keyword, so the pattern above cannot see it. Measured on the
    # pornify tone (temperature 1.2), 「先生」 came back as "Mr. [partner's name]"
    # in one run and "Mr. [Teacher]" in the next — same bug, and only the first
    # form matched.
    #
    # Widening the pattern above to "any bracketed word" would fire on
    # "[moaning]" and every other sound annotation. The honorific prefix is the
    # safe discriminator instead: only titles that *require* a following name
    # are listed, so a bracket after one can only be a fill-in. Standalone
    # vocatives (master, mistress, sir, madam) are deliberately excluded — they
    # can legitimately end a sentence, which would make "master... [moaning]"
    # a false positive.
    ("template", re.compile(
        r"\b(?:mr|mrs|ms|mx|miss|dr|prof|professor)\.?\s*"
        r"[\[\(<][^\]\)>\n]{1,40}[\]\)>]",
        re.I)),

    # ---- Chinese (Hunyuan and other CN-trained models refuse in Chinese) ---
    # 抱歉 / 对不起 + inability, co-occurring. Bare 我不能 ("I can't") is
    # plausible dialogue, so it is never sufficient by itself.
    ("apology-zh", re.compile(
        r"(?:抱歉|对不起|很遗憾)[^。！？]{0,30}?(?:无法|不能|不便|不予)")),
    # Inability + request-directed object.
    ("inability-zh", re.compile(
        r"(?:无法|不能|不会|拒绝)[^。！？]{0,10}?"
        r"(?:提供|翻译|协助|帮助|完成|继续|生成|处理)"
        r"[^。！？]{0,10}?(?:请求|要求|内容|翻译|这类|此类|该内容)")),
    ("as-an-ai-zh", re.compile(
        r"作为(?:一个|一名)?(?:AI|A\.I\.|人工智能|语言模型|智能助手|助手)")),
    ("policy-zh", re.compile(
        r"(?:违反|不符合|违背)[^。！？]{0,20}?(?:规定|政策|准则|指南|原则|条款)")),

    # ---- Japanese (rare in output, cheap to cover) ----------------------
    ("apology-ja", re.compile(
        r"(?:申し訳(?:あり|ござい)ませんが?|恐れ入りますが)[^。！？]{0,30}?"
        r"(?:できません|できかねます|お応えできません|翻訳できません)")),
]


@dataclass
class RefusalHit:
    """One line of output that tripped the detector."""
    index: int          # 1-based subtitle line number (0 when not applicable)
    label: str          # which pattern family matched — see _PATTERNS
    text: str           # the offending text, truncated for logging


@dataclass
class RefusalReport:
    """Aggregate result of scanning a translated subtitle set."""
    total_lines: int = 0
    hits: list[RefusalHit] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.hits)

    @property
    def rate(self) -> float:
        """Fraction of scanned lines that look like refusals (0.0–1.0)."""
        return self.count / self.total_lines if self.total_lines else 0.0

    def __bool__(self) -> bool:
        """Truthy when at least one refusal was found."""
        return bool(self.hits)


def classify(text: str) -> str | None:
    """Return the matching pattern label for *text*, or None if it looks clean.

    The label identifies the pattern *family* (``apology``, ``placeholder``,
    ``policy``, …) rather than the exact regex, which is what a user needs in
    order to act on the diagnostic.
    """
    if not text:
        return None
    # Collapse whitespace so a refusal split across two subtitle lines by the
    # model's own wrapping still matches a single-sentence pattern.
    flat = " ".join(text.split())
    if not flat:
        return None
    for label, pattern in _PATTERNS:
        if pattern.search(flat):
            return label
    return None


def looks_like_refusal(text: str) -> bool:
    """True when *text* looks like a model refusal rather than a translation."""
    return classify(text) is not None


def scan(lines, *, max_hits: int = 50) -> RefusalReport:
    """Scan an iterable of translated strings for refusal text.

    Args:
        lines: iterable of translated line strings. ``None`` and empty entries
            are counted as scanned but never flagged — an untranslated line is
            a different failure, already reported by the batch diagnostics.
        max_hits: stop recording individual hits past this many (the count and
            rate keep incrementing). Guards against a fully-refused 2000-line
            file dumping 2000 diagnostics into the log.

    Returns:
        A :class:`RefusalReport`.
    """
    report = RefusalReport()
    for i, line in enumerate(lines, start=1):
        report.total_lines += 1
        if not line:
            continue
        label = classify(line)
        if label is None:
            continue
        if len(report.hits) < max_hits:
            flat = " ".join(str(line).split())
            report.hits.append(RefusalHit(index=i, label=label, text=flat[:160]))
        else:
            # Still count it, without keeping the text.
            report.hits.append(RefusalHit(index=i, label=label, text=""))
    return report


def scan_response_text(text: str) -> str | None:
    """Check a raw whole-batch LLM response for an outright refusal.

    Used on the unparsed response body, before PySubtrans looks for ``#<n>``
    markers.  A refusal here means the whole batch was declined, which is
    otherwise reported only as an opaque "No matches" warning.

    Returns the matching label, or None.  A response that contains numbered
    translation markers is treated as clean even if it also contains refusal
    phrasing, because the markers mean real translations came back and any
    boilerplate is a preamble — the per-line scan will catch leaks.
    """
    if not text:
        return None
    if re.search(r"(?m)^\s*#\d+\s*$|#\d+\b.*?\n.*?\S", text):
        # Structured batch output present → not a whole-batch refusal.
        return None
    return classify(text)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(report: RefusalReport, *, model: str = None,
                  max_examples: int = 5) -> list[str]:
    """Render *report* as a list of printable diagnostic lines.

    Returns an empty list when nothing was found, so callers can do::

        for line in format_report(report):
            print(line, file=sys.stderr)
    """
    if not report:
        return []

    pct = report.rate * 100
    out = [
        "",
        "[TRANSLATE] *** REFUSAL TEXT DETECTED IN TRANSLATED SUBTITLES ***",
        f"[TRANSLATE]   {report.count} of {report.total_lines} lines "
        f"({pct:.1f}%) look like model refusals, not translations.",
    ]
    if model:
        out.append(f"[TRANSLATE]   Model: {model}")

    # Group by label so the user sees the shape of the problem, not just a list.
    by_label: dict[str, int] = {}
    for hit in report.hits:
        by_label[hit.label] = by_label.get(hit.label, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_label.items()))
    out.append(f"[TRANSLATE]   Pattern breakdown: {breakdown}")

    shown = [h for h in report.hits if h.text][:max_examples]
    if shown:
        out.append("[TRANSLATE]   Examples:")
        for hit in shown:
            out.append(f"[TRANSLATE]     line {hit.index} [{hit.label}]: {hit.text}")
        if report.count > len(shown):
            out.append(f"[TRANSLATE]     ... and {report.count - len(shown)} more")

    out += [
        "[TRANSLATE]   These lines are IN YOUR OUTPUT FILE. The model censored "
        "itself instead of translating.",
        "[TRANSLATE]   Fix: use an uncensored model. WhisperJAV's default "
        "uncensored pick is:",
        "[TRANSLATE]     --translate-provider ollama "
        "--translate-model huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M",
        "[TRANSLATE]   Then re-run translation (delete the .subtrans project "
        "first, or already-translated batches are reused).",
    ]
    return out
