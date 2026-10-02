"""Tests for uncensored translation: refusal detection and the model catalog.

JAV dialogue is explicit by definition, so a safety-aligned LLM does not merely
score worse at this task — it writes refusal boilerplate into the subtitle
track. These tests cover the two halves of the fix:

- ``whisperjav.translate.refusal`` — the detector that proves whether the fix
  held, in both failure modes (whole-batch refusal, per-line leak).
- ``whisperjav.translate.ollama_manager`` — the uncensored-by-default model
  catalog and VRAM ladder, plus the invariants that keep the ladder, the
  catalog, and the GUI's curated JSON from drifting apart.

The false-positive tests matter as much as the true-positive ones. JAV dialogue
is saturated with first-person inability phrasing ("I can't take it anymore")
that a naive matcher would flag on every single file, making the diagnostic
useless. Every trap in REAL_DIALOGUE is phrasing that genuinely appears in this
material.
"""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisperjav.translate import refusal
from whisperjav.translate.ollama_manager import (
    OLLAMA_MODEL_CONFIGS,
    OllamaManager,
    _ALIGNED_LADDER,
    _UNCENSORED_LADDER,
    get_sampling_params,
    is_uncensored,
)
from whisperjav.translate.providers import PROVIDER_CONFIGS


# Real refusal shapes, in the languages the models actually refuse in.
# Hunyuan is a Chinese model and declines in Chinese, so those are not padding.
REFUSALS = [
    # English — apology + inability
    "I'm sorry, but I can't help with that.",
    "I am sorry, I cannot provide a translation for this content.",
    "Sorry, I won't be able to translate that.",
    "Unfortunately, I cannot assist with this request.",
    "My apologies, but I can't continue with this.",
    # English — inability + request-directed object
    "I cannot assist with that.",
    "I can't comply with this request.",
    "I'm unable to translate sexually explicit material.",
    "I will not generate that kind of content.",
    "I can't help you with that.",
    # English — self-identification
    "As an AI language model, I am not able to translate this.",
    "As an AI assistant, I have to decline.",
    # English — policy reference
    "This request violates my content policy.",
    "That is against my safety guidelines.",
    "This goes against usage policies.",
    # English — explicit declining
    "I must decline.",
    "I must respectfully decline that.",
    "I will refuse to comply with this.",
    "I'm not comfortable translating this.",
    "I am not willing to provide this translation.",
    # English — inline placeholders (the silent killers)
    "[Content omitted]",
    "(explicit content removed)",
    "<content censored>",
    "[Explicit dialogue not translated]",
    "(redacted)",
    # Chinese
    "抱歉，我无法翻译此内容。",
    "对不起，我不能提供这样的翻译。",
    "作为一个人工智能助手，我不能这样做。",
    "这违反了我的使用政策。",
    # Japanese
    "申し訳ありませんが、翻訳できません。",
]

# Phrasing that genuinely occurs in JAV subtitles and must NEVER be flagged.
REAL_DIALOGUE = [
    "I can't take it anymore!",
    "I can't take it, it's too much!",
    "I can't help it...",
    "I can't hold back any longer.",
    "Stop, I can't!",
    "No, I can't do this anymore.",
    "I cannot believe how good that feels.",
    "I can't breathe...",
    "I won't tell anyone, I promise.",
    "I won't let you go.",
    "I'm sorry, I was late.",
    "Sorry, that hurts a little.",
    "I'm so sorry, master.",
    "Unfortunately he never came back.",
    "That's against the rules of the club.",
    "I refuse to marry him!",
    "I must decline his proposal.",
    "I must decline the invitation politely.",
    "She declined to answer.",
    "Please, don't stop!",
    "It feels so good, I'm going to cum.",
    "Do it harder!",
    "我不能再忍了！",
    "我不能没有你。",
    "申し訳ありません、遅れました。",
    "だめ、もう我慢できない",
    "そんなこと、できません",
]


class TestRefusalRecall:
    """Every known refusal shape must be caught."""

    @pytest.mark.parametrize("text", REFUSALS)
    def test_refusal_is_detected(self, text):
        label = refusal.classify(text)
        assert label is not None, f"missed refusal: {text!r}"
        assert refusal.looks_like_refusal(text)

    def test_all_refusals_caught(self):
        missed = [t for t in REFUSALS if not refusal.looks_like_refusal(t)]
        assert not missed, f"{len(missed)} refusals missed: {missed}"


class TestRefusalPrecision:
    """Real dialogue must never be flagged — a noisy detector is a dead one."""

    @pytest.mark.parametrize("text", REAL_DIALOGUE)
    def test_dialogue_is_not_flagged(self, text):
        label = refusal.classify(text)
        assert label is None, f"false positive on dialogue {text!r} (matched {label})"

    def test_zero_false_positives(self):
        hits = [(t, refusal.classify(t)) for t in REAL_DIALOGUE
                if refusal.looks_like_refusal(t)]
        assert not hits, f"{len(hits)} false positives: {hits}"


class TestTemplateLeaks:
    """Bracketed fill-in fields the model left for a human to complete.

    Nothing was refused here — the model simply punted on a word — but the
    result is the same silent .srt corruption as the ``placeholder`` omission
    markers, so the ``template`` family rides the same safety net.

    Both shapes were measured on qwen2.5-abliterate with the pornify tone
    (temperature 1.2) translating 「イカせて、お願い先生」: one run returned
    "Mr. [partner's name]", the next "Mr. [Teacher]". The second puts the
    *translation* inside the bracket rather than a template keyword, so it needs
    its own pattern — which is why both are pinned here.
    """

    TEMPLATE_LEAKS = [
        # Template keyword inside the bracket.
        "Mr. [partner's name]",
        "[name]",
        "[insert name here]",
        "[your name]",
        "[character name]",
        "(his name)",
        "<title>",
        "[honorific]",
        # Honorific prefix outside the bracket, translation inside it.
        "Make me cum, please, Mr. [Teacher].",
        "Yes, Mrs. [Wife]",
        "Ms. [Name]",
        "Dr. [doctor]",
        "Prof. [surname]",
    ]

    # Sound annotations are legitimate subtitle content and share the bracket
    # syntax. If the detector ever fires on these it becomes useless on real
    # files, so they are pinned as hard negatives.
    ANNOTATIONS = [
        "[moaning]",
        "(sighs)",
        "[laughs]",
        "[panting]",
        "[wet sounds]",
        "[kissing noises]",
        "[the word]",
        "Ahh... Mmm...",
        "I'm cumming!",
        # Standalone vocatives deliberately left out of the honorific list —
        # they can end a sentence, so a following annotation is plausible.
        "Yes, master... [moaning]",
        "Please, sir... (panting)",
        "Mistress... [whimpering]",
    ]

    @pytest.mark.parametrize("text", TEMPLATE_LEAKS)
    def test_template_leak_is_flagged(self, text):
        assert refusal.classify(text) == "template", f"missed template leak: {text!r}"

    @pytest.mark.parametrize("text", ANNOTATIONS)
    def test_annotation_is_not_flagged(self, text):
        label = refusal.classify(text)
        assert label is None, f"false positive on annotation {text!r} (matched {label})"


class TestClassifyEdgeCases:
    def test_empty_and_none_are_clean(self):
        assert refusal.classify("") is None
        assert refusal.classify(None) is None
        assert refusal.classify("   \n\t  ") is None

    def test_wrapped_refusal_still_matches(self):
        """A refusal the model split across lines must still match.

        Patterns are single-sentence by design (bounded [^.!?] gaps), so
        classify() collapses whitespace first. Without that, the model's own
        line wrapping would defeat the apology+inability pairing.
        """
        assert refusal.classify("I'm sorry, but I\ncannot translate this.") is not None

    def test_label_identifies_the_family(self):
        assert refusal.classify("[Content omitted]") == "placeholder"
        assert refusal.classify("As an AI language model, no.") == "as-an-ai"
        assert refusal.classify("I'm sorry, I cannot do that.") == "apology"


class TestScan:
    def test_counts_and_rate(self):
        lines = ["Do it harder!", "[Content omitted]", "I can't take it anymore!"]
        report = refusal.scan(lines)
        assert report.total_lines == 3
        assert report.count == 1
        assert report.rate == pytest.approx(1 / 3)
        assert bool(report) is True
        assert report.hits[0].index == 2, "index must be 1-based subtitle position"
        assert report.hits[0].label == "placeholder"

    def test_clean_track_is_falsy_and_reports_nothing(self):
        report = refusal.scan(REAL_DIALOGUE)
        assert report.count == 0
        assert bool(report) is False
        assert report.rate == 0.0
        assert refusal.format_report(report) == []

    def test_untranslated_lines_counted_not_flagged(self):
        """None/empty is a different failure, already reported elsewhere."""
        report = refusal.scan(["ok", None, "", "also ok"])
        assert report.total_lines == 4
        assert report.count == 0

    def test_empty_input(self):
        report = refusal.scan([])
        assert report.total_lines == 0
        assert report.count == 0
        assert report.rate == 0.0, "must not divide by zero"

    def test_max_hits_caps_retained_text_not_the_count(self):
        """A fully-refused 2000-line file must not dump 2000 diagnostics."""
        report = refusal.scan(["[Content omitted]"] * 120, max_hits=10)
        assert report.count == 120, "count must keep incrementing past the cap"
        assert sum(1 for h in report.hits if h.text) == 10
        assert report.rate == 1.0

    def test_accepts_a_generator(self):
        """core.py passes a generator over SubtitleLine.text."""
        report = refusal.scan(line for line in ["ok", "[redacted]"])
        assert report.total_lines == 2
        assert report.count == 1


class TestScanResponseText:
    """Whole-batch refusal: no numbered markers came back at all."""

    def test_bare_refusal_detected(self):
        text = ("I'm sorry, but I can't help with that. The content appears to be "
                "sexually explicit, which is against my content policy.")
        assert refusal.scan_response_text(text) is not None

    def test_structured_batch_is_clean(self):
        text = ("#1\nOriginal>\nもっと激しくして\nTranslation>\nDo it harder.\n\n"
                "#2\nOriginal>\nだめ\nTranslation>\nNo!\n")
        assert refusal.scan_response_text(text) is None

    def test_preamble_plus_translations_is_clean(self):
        """Markers present means real translations came back.

        Boilerplate alongside them is a preamble, not a batch refusal — the
        per-line scan is what catches leaks inside the translations.
        """
        text = "I'll translate this for you.\n\n#1\nTranslation>\nDo it harder.\n"
        assert refusal.scan_response_text(text) is None

    def test_empty_is_clean(self):
        assert refusal.scan_response_text("") is None
        assert refusal.scan_response_text(None) is None


class TestFormatReport:
    def test_names_the_actionable_fix(self):
        report = refusal.scan(["[Content omitted]", "ok"])
        out = "\n".join(refusal.format_report(report, model="qwen2.5:7b"))
        assert "REFUSAL TEXT DETECTED" in out
        assert "qwen2.5:7b" in out, "must name the offending model"
        # The whole point of the diagnostic is a copy-pasteable fix.
        assert "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M" in out
        assert "--translate-model" in out

    def test_recommended_fix_is_actually_uncensored(self):
        """Guard against the advice drifting out of sync with the catalog."""
        out = "\n".join(refusal.format_report(refusal.scan(["[redacted]"])))
        recommended = [tok for tok in out.split() if tok.startswith("huihui_ai/")]
        assert recommended, "no model recommended"
        for model in recommended:
            assert is_uncensored(model), f"recommended {model} is not uncensored"
            assert model in OLLAMA_MODEL_CONFIGS, f"recommended {model} not in catalog"

    def test_breakdown_and_examples(self):
        report = refusal.scan(["[Content omitted]", "I'm sorry, I cannot translate."])
        out = "\n".join(refusal.format_report(report))
        assert "placeholder=1" in out
        assert "apology=1" in out
        assert "line 1" in out and "line 2" in out

    def test_example_count_is_capped_with_an_and_more_line(self):
        report = refusal.scan(["[Content omitted]"] * 9)
        out = "\n".join(refusal.format_report(report, max_examples=3))
        assert "and 6 more" in out


class TestUncensoredDefaults:
    """The default path must not hand a user a model that refuses."""

    def test_ollama_provider_default_is_uncensored(self):
        model = PROVIDER_CONFIGS['ollama']['model']
        assert is_uncensored(model), f"default ollama model {model} is aligned"
        assert model in OLLAMA_MODEL_CONFIGS

    def test_recommendation_is_uncensored_at_every_vram_size(self):
        mgr = OllamaManager()
        for vram in (0, 2, 4, 6, 8, 10, 12, 14, 16, 24, 48):
            rec = mgr.recommend_model(vram)
            assert rec.uncensored, f"{vram} GB -> aligned model {rec.name}"
            assert is_uncensored(rec.name)

    def test_aligned_tier_requires_explicit_opt_out(self):
        mgr = OllamaManager()
        rec = mgr.recommend_model(8, prefer_uncensored=False)
        assert rec.uncensored is False
        assert "SAFETY-ALIGNED" in rec.note, "opting out must be warned about"

    def test_uncensored_note_says_so(self):
        rec = OllamaManager().recommend_model(8)
        assert "uncensored" in rec.note.lower()

    def test_eight_gb_card_gets_the_subtitle_safe_model(self):
        """The target hardware (RTX 4060, 8 GB) must land on qwen2.5-abliterate.

        Deliberately NOT the Hunyuan-MT specialist, which scores higher on MT
        benchmarks but on a 14-cue JAV probe returned 12 cues and shifted every
        line after cue 8 into the wrong slot — fluent text on wrong timings,
        silent and non-deterministic. It also ignored glossary renderings it
        had been given verbatim ("about to reach orgasm" for イッちゃう).
        """
        rec = OllamaManager().recommend_model(8)
        assert rec.name == "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"
        assert rec.quality == "very_good"


class TestVramFloorsUseReportedNotNameplateSizes:
    """Ladder floors are nameplate GB; cards never report nameplate.

    The bug this pins: ``recommend_model(8)`` passed, but no card reports 8.
    An RTX 4060 Laptop reports 8188 MiB to nvidia-smi and 7797 MiB
    (= 7.61 GiB) to ``torch.cuda.mem_get_info`` — the driver keeps the
    difference — so a bare ``vram_gb >= 8`` dropped an 8 GB card to the 4 GB
    tier and handed it the 4B instead of the translation specialist. Passing a
    round 8 in tests hid it completely.
    """

    # (label, torch CUDA-addressable GiB, nvidia-smi board GiB, expected model)
    REAL_CARDS = [
        ("RTX 4060 Laptop 8 GB", 7.6143798828125, 7.99609375,
         "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"),
        ("8 GB desktop",         7.79,  8.00,
         "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"),
        ("12 GB",                11.63, 12.00,
         "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"),
        ("16 GB",                15.48, 15.9921875,
         "huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M"),
        ("24 GB",                23.55, 24.00,
         "huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M"),
        ("6 GB",                 5.72,  6.00,
         "huihui_ai/qwen3-abliterated:4b"),
        ("4 GB",                 3.81,  4.00,
         "huihui_ai/qwen3-abliterated:4b"),
    ]

    @pytest.mark.parametrize("label,torch_gib,smi_gib,expected", REAL_CARDS)
    def test_reported_vram_lands_on_the_right_tier(self, label, torch_gib,
                                                   smi_gib, expected):
        mgr = OllamaManager()
        assert mgr.recommend_model(torch_gib).name == expected, (
            f"{label}: torch reports {torch_gib} GiB and got the wrong tier")
        assert mgr.recommend_model(smi_gib).name == expected, (
            f"{label}: nvidia-smi reports {smi_gib} GiB and got the wrong tier")

    @pytest.mark.parametrize("label,torch_gib,smi_gib,expected", REAL_CARDS)
    def test_both_detection_paths_agree(self, label, torch_gib, smi_gib,
                                        expected):
        """_detect_vram_gb() has two backends that report different totals.

        torch returns CUDA-addressable memory, nvidia-smi returns the board
        total. They differ by the driver reserve (391 MiB on this box). Which
        one runs depends on whether torch imports, so if the two disagree on
        the tier the recommendation becomes environment-dependent.
        """
        mgr = OllamaManager()
        assert mgr.recommend_model(torch_gib).name == \
            mgr.recommend_model(smi_gib).name, (
                f"{label}: tier depends on which detection backend ran")

    def test_slack_cannot_promote_a_card_past_a_tier_it_cannot_hold(self):
        """The slack must stay well inside the gap between floors.

        A 6 GB card is the largest thing below the 8 GB floor and a 12 GB card
        the largest below 16; neither may be promoted, or the model spills into
        shared memory and translation crawls.
        """
        mgr = OllamaManager()
        assert mgr.recommend_model(6.0).name == "huihui_ai/qwen3-abliterated:4b"
        assert mgr.recommend_model(12.0).name == \
            "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M"

    def test_note_does_not_overstate_the_card(self):
        """``:.0f`` printed "8 GB VRAM" for a 7.5 GiB card.

        Worse, it printed "8 GB VRAM" *while* picking the 4B, so the output
        contradicted itself and read like a different bug than the real one.
        """
        rec = OllamaManager().recommend_model(7.6143798828125)
        assert "7.6 GB VRAM" in rec.note, rec.note
        assert "8 GB VRAM" not in rec.note, rec.note

    def test_low_vram_label_matches_the_tier_it_picks(self):
        """Whatever gets the bottom tier is what may be called low-VRAM."""
        mgr = OllamaManager()
        bottom = "huihui_ai/qwen2.5-abliterate:3b-instruct-q4_K_M"
        for vram in (0, 0.5, 1, 2, 3, 3.5, 3.81, 4, 6, 7.61, 8, 12, 16, 24):
            rec = mgr.recommend_model(vram)
            says_low = "CPU/low VRAM" in rec.note
            assert says_low == (rec.name == bottom), (
                f"{vram} GB: note says low-VRAM={says_low} but picked "
                f"{rec.name}")


class TestIsUncensored:
    @pytest.mark.parametrize("model", [
        "huihui_ai/hy-mt1.5-abliterated:7b",
        "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M",
        "dolphin-llama3:8b-256k-v2.9-q4_K_M",
    ])
    def test_known_uncensored(self, model):
        assert is_uncensored(model)

    @pytest.mark.parametrize("model", ["qwen2.5:7b", "gemma3:12b", "qwen2.5:14b"])
    def test_known_aligned(self, model):
        assert not is_uncensored(model)

    @pytest.mark.parametrize("model,expected", [
        ("some/thing-abliterated:8b", True),
        ("vendor/model-uncensored:latest", True),
        ("dolphin-mixtral:8x7b", True),
        ("mistral:7b", False),
        ("llama3.1:8b", False),
    ])
    def test_unknown_model_name_heuristic(self, model, expected):
        """Unlisted models fall back to community tag names, for warnings only."""
        assert is_uncensored(model) is expected


class TestSamplingParameters:
    def test_hunyuan_uses_its_published_sampling(self):
        """Tencent published these; they are not our cross-model median.

        top_p 0.6 with top_k 20 is a materially tighter nucleus than the global
        0.92/50 default, and mixing the two sets is worse than either.
        """
        params = get_sampling_params("huihui_ai/hy-mt1.5-abliterated:7b")
        assert params['top_k'] == 20
        assert params['top_p'] == 0.6
        assert params['repeat_penalty'] == 1.05
        assert 'min_p' not in params, "min_p was never validated for this model"

    def test_models_without_an_override_get_the_global_default(self):
        params = get_sampling_params("huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M")
        assert params['top_k'] == 50
        assert params['top_p'] == 0.92

    def test_unknown_model_gets_the_global_default(self):
        assert get_sampling_params("nobody/nothing:latest")['top_k'] == 50

    def test_returns_a_copy(self):
        """Callers mutate this dict; the catalog must not be corrupted."""
        first = get_sampling_params("huihui_ai/hy-mt1.5-abliterated:7b")
        first['top_k'] = 999
        assert get_sampling_params("huihui_ai/hy-mt1.5-abliterated:7b")['top_k'] == 20

    def test_repeat_penalty_is_gentle_for_jav(self):
        """JAV dialogue is legitimately repetitive; a harsh penalty suppresses it."""
        assert get_sampling_params("huihui_ai/hy-mt1.5-abliterated:7b")['repeat_penalty'] <= 1.05


class TestCatalogInvariants:
    """Keep the ladder, the catalog, and the GUI's JSON from drifting apart."""

    @pytest.mark.parametrize("ladder,want_uncensored", [
        (_UNCENSORED_LADDER, True),
        (_ALIGNED_LADDER, False),
    ])
    def test_ladder_entries_exist_with_the_right_tier(self, ladder, want_uncensored):
        for floor, model in ladder:
            cfg = OLLAMA_MODEL_CONFIGS.get(model)
            assert cfg is not None, f"ladder model {model} missing from catalog"
            assert cfg.get('uncensored', False) is want_uncensored, \
                f"{model} is in the wrong tier"
            assert cfg['min_vram_gb'] == floor, \
                f"{model} min_vram_gb {cfg['min_vram_gb']} != ladder floor {floor}"

    @pytest.mark.parametrize("ladder", [_UNCENSORED_LADDER, _ALIGNED_LADDER])
    def test_ladder_is_descending_and_reaches_zero(self, ladder):
        floors = [floor for floor, _ in ladder]
        assert floors == sorted(floors, reverse=True), "ladder must be largest-first"
        assert floors[-1] == 0, "ladder must have a 0 GB fallback or selection can fail"
        assert len(set(floors)) == len(floors), "duplicate VRAM floors are ambiguous"

    def test_every_catalog_entry_declares_its_alignment(self):
        for model, cfg in OLLAMA_MODEL_CONFIGS.items():
            assert 'uncensored' in cfg, f"{model} does not declare 'uncensored'"
            assert isinstance(cfg['uncensored'], bool)

    def test_every_catalog_entry_is_fully_specified(self):
        required = ('num_ctx', 'batch_size', 'temperature', 'download_size',
                    'quality', 'min_vram_gb')
        for model, cfg in OLLAMA_MODEL_CONFIGS.items():
            missing = [k for k in required if k not in cfg]
            assert not missing, f"{model} missing {missing}"

    def test_download_sizes_are_decimal_gb_strings(self):
        """Sizes must match what `ollama pull` prints, or the prompt misleads."""
        for model, cfg in OLLAMA_MODEL_CONFIGS.items():
            size = cfg['download_size']
            assert size.endswith(' GB'), f"{model} size {size!r} is not '<n> GB'"
            assert float(size[:-3]) > 0

    def test_uncensored_tier_covers_every_vram_size(self):
        """No VRAM value may fall through to an aligned model."""
        mgr = OllamaManager()
        for vram in range(0, 49):
            assert mgr.recommend_model(vram).uncensored, f"{vram} GB fell through"


class TestCuratedConfigMatchesCatalog:
    """config/ollama_models.json feeds the GUI dropdown; it must not drift."""

    @staticmethod
    def _load():
        path = (Path(__file__).parent.parent / 'whisperjav' / 'config'
                / 'ollama_models.json')
        return json.loads(path.read_text(encoding='utf-8'))

    def test_json_is_valid_and_non_empty(self):
        curated = self._load()
        assert isinstance(curated, list) and curated

    def test_required_gui_fields_present(self):
        """app.js reads model/size/label; a missing key breaks the dropdown."""
        for entry in self._load():
            for key in ('model', 'size', 'label', 'rank', 'uncensored'):
                assert key in entry, f"{entry.get('model')} missing {key!r}"

    def test_top_ranked_entry_is_uncensored(self):
        curated = sorted(self._load(), key=lambda e: e['rank'])
        assert curated[0]['uncensored'] is True
        assert is_uncensored(curated[0]['model'])

    def test_default_provider_model_is_offered_in_the_gui(self):
        models = [e['model'] for e in self._load()]
        assert PROVIDER_CONFIGS['ollama']['model'] in models

    def test_ranks_are_unique(self):
        ranks = [e['rank'] for e in self._load()]
        assert len(set(ranks)) == len(ranks)

    def test_every_curated_entry_is_tuned_in_the_catalog(self):
        """An unlisted model silently falls back to untuned defaults."""
        for entry in self._load():
            assert entry['model'] in OLLAMA_MODEL_CONFIGS, \
                f"{entry['model']} is offered in the GUI but has no tuning"

    def test_sizes_and_alignment_agree_with_the_catalog(self):
        for entry in self._load():
            cfg = OLLAMA_MODEL_CONFIGS[entry['model']]
            assert entry['size'] == cfg['download_size'], \
                f"{entry['model']}: JSON {entry['size']} != catalog {cfg['download_size']}"
            assert entry['uncensored'] == cfg['uncensored'], \
                f"{entry['model']}: alignment flag disagrees with the catalog"
            assert entry['min_vram_gb'] == cfg['min_vram_gb']

    def test_aligned_entries_are_labelled_as_such(self):
        """A user picking from the dropdown must see the risk in the label."""
        for entry in self._load():
            if not entry['uncensored']:
                assert 'ALIGNED' in entry['label'].upper(), \
                    f"{entry['model']} is aligned but its label does not say so"


# ---------------------------------------------------------------------------
# core.py wiring
# ---------------------------------------------------------------------------
# The detector is only worth anything if it is actually reached during a run.
# These drive translate_subtitle() with PySubtrans mocked out, the same way
# tests/test_translation_e2e_output.py does, and assert on what lands on stderr.

class _Line:
    """Minimal stand-in for PySubtrans SubtitleLine."""

    def __init__(self, number, text):
        self.number = number
        self.text = text


def _run_translate(tmp_path, translated_lines=None, on_translate=None,
                   model="qwen2.5:7b"):
    """Drive translate_subtitle() with PySubtrans mocked; return stderr text."""
    import io
    import sys

    from PySubtrans.TranslationEvents import TranslationEvents
    from whisperjav.translate.core import translate_subtitle

    test_srt = tmp_path / "in.srt"
    test_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nテスト\n", encoding="utf-8")
    output_srt = tmp_path / "in.english.srt"

    mock_translator = MagicMock()
    mock_translator.events = TranslationEvents()
    mock_translator.resume = False

    mock_project = MagicMock()
    mock_project.events = MagicMock()
    mock_project.events.batch_translated = MagicMock()
    mock_project.existing_project = False

    subtitles = MagicMock()
    subtitles.linecount = len(translated_lines) if translated_lines else 0
    subtitles.any_translated = bool(translated_lines)
    subtitles.all_translated = bool(translated_lines)
    subtitles.translated = translated_lines if translated_lines is not None else []
    subtitles.outputpath = None
    mock_project.subtitles = subtitles

    def _translate(translator):
        if on_translate is not None:
            on_translate(translator)

    mock_project.TranslateSubtitles = _translate
    mock_project.SaveTranslation = MagicMock(return_value=str(output_srt))
    mock_project.SaveProject = MagicMock()

    captured = io.StringIO()
    old_stderr = sys.stderr
    try:
        sys.stderr = captured
        with patch("PySubtrans.init_options") as m1, \
             patch("PySubtrans.init_translation_provider") as m2, \
             patch("PySubtrans.init_project") as m3, \
             patch("PySubtrans.init_translator") as m4:
            m1.return_value = MagicMock()
            m2.return_value = MagicMock()
            m3.return_value = mock_project
            m4.return_value = mock_translator
            translate_subtitle(
                input_path=str(test_srt),
                output_path=output_srt,
                provider_config={"pysubtrans_name": "Custom Server"},
                model=model,
                api_key="k",
            )
    finally:
        sys.stderr = old_stderr
    return captured.getvalue()


class TestCoreWiringPerLineScan:
    """Per-line leak: markers intact, refusal text lands in the .srt."""

    def test_leaked_refusals_are_reported(self, tmp_path):
        lines = [
            _Line(1, "Do it harder!"),
            _Line(2, "[Content omitted]"),
            _Line(3, "I'm sorry, but I cannot translate this line."),
            _Line(4, "I can't take it anymore!"),   # real dialogue, must not count
        ]
        out = _run_translate(tmp_path, translated_lines=lines)
        assert "REFUSAL TEXT DETECTED" in out
        assert "2 of 4 lines" in out, f"wrong count in:\n{out}"
        assert "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M" in out
        assert "qwen2.5:7b" in out, "must name the model that refused"

    def test_clean_translation_produces_no_refusal_warning(self, tmp_path):
        lines = [
            _Line(1, "Do it harder!"),
            _Line(2, "I can't take it anymore!"),
            _Line(3, "It feels so good."),
        ]
        out = _run_translate(tmp_path, translated_lines=lines)
        assert "REFUSAL TEXT DETECTED" not in out

    def test_no_translated_lines_does_not_crash(self, tmp_path):
        out = _run_translate(tmp_path, translated_lines=[])
        assert "REFUSAL TEXT DETECTED" not in out
        assert "Translation result (ground truth)" in out

    def test_scan_failure_cannot_break_the_run(self, tmp_path):
        """A broken diagnostic must never turn a good run into a failure."""
        class Exploding:
            @property
            def text(self):
                raise RuntimeError("boom")

        out = _run_translate(tmp_path, translated_lines=[Exploding()])
        assert "Translation result (ground truth)" in out
        assert "Saving translated subtitles" in out, "run must continue past the scan"


class TestCoreWiringWholeBatchRefusal:
    """Whole-batch refusal: response carries no numbered translations at all."""

    @staticmethod
    def _response(text):
        return {"choices": [{"message": {"content": text}}]}

    def test_refused_batch_is_named_as_an_alignment_problem(self, tmp_path):
        refusal_text = ("I'm sorry, but I can't help with that. This request "
                        "violates my content policy.")

        def on_translate(translator):
            translator.client._process_api_response(self._response(refusal_text), {})

        out = _run_translate(tmp_path, on_translate=on_translate)
        assert "MODEL REFUSED THIS BATCH" in out
        # The old advice ("smaller --max-batch-size") is actively wrong here.
        assert "MODEL ALIGNMENT problem" in out
        assert "Shrinking batches will not help" in out

    def test_valid_batch_response_is_not_flagged(self, tmp_path):
        good = "#1\nOriginal>\nテスト\nTranslation>\nTest.\n"

        def on_translate(translator):
            translator.client._process_api_response(self._response(good), {})

        out = _run_translate(tmp_path, on_translate=on_translate)
        assert "MODEL REFUSED THIS BATCH" not in out

    def test_refusal_in_reasoning_field_is_caught(self, tmp_path):
        """Thinking models put output in 'reasoning' with empty 'content'."""
        def on_translate(translator):
            translator.client._process_api_response(
                {"choices": [{"message": {"content": "",
                                          "reasoning": "I cannot assist with that."}}]},
                {},
            )

        out = _run_translate(tmp_path, on_translate=on_translate)
        assert "MODEL REFUSED THIS BATCH" in out

    def test_watcher_delegates_the_response_unchanged(self, tmp_path):
        """The watcher observes only — it must not alter what PySubtrans parses."""
        seen = {}

        def on_translate(translator):
            original = self._response("I'm sorry, I cannot translate this.")
            translator.client._process_api_response(original, {"marker": 1})
            # The MagicMock records the delegated call for us.
            seen['calls'] = translator.client._process_api_response

        out = _run_translate(tmp_path, on_translate=on_translate)
        assert "MODEL REFUSED THIS BATCH" in out
        assert seen['calls'] is not None

    def test_no_matches_warning_stops_blaming_batch_size(self, tmp_path):
        """When a refusal was seen, 'No matches' must not misdiagnose format."""
        def on_translate(translator):
            translator.client._process_api_response(
                self._response("I'm sorry, but I can't help with that."), {})
            translator.events.warning.send(None, message="No matches found in response")

        out = _run_translate(tmp_path, on_translate=on_translate)
        assert "were REFUSALS, not format errors" in out
        assert "Consider: smaller --max-batch-size" not in out

    def test_no_matches_without_a_refusal_keeps_the_original_advice(self, tmp_path):
        def on_translate(translator):
            translator.events.warning.send(None, message="No matches found in response")

        out = _run_translate(tmp_path, on_translate=on_translate)
        assert "Consider: smaller --max-batch-size" in out
        assert "were REFUSALS, not format errors" not in out


# ─────────────────────────────────────────────────────────────────────────────
# VRAM release after translation
#
# Served by a real HTTP stub rather than mocks: the code under test speaks the
# /api/ps and /api/generate contracts, and the whole point of this fix is that
# the previous version got those contracts' *semantics* wrong (it named a model
# that was never loaded). Mocking the transport would have let that bug pass.
# ─────────────────────────────────────────────────────────────────────────────

class _OllamaStub:
    """Minimal stand-in for an Ollama server. Records unload requests."""

    def __init__(self):
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.resident = []
        self.unloaded = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                body = _json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/api/ps':
                    self._send({'models': stub.resident})
                elif self.path == '/':
                    self._send({'status': 'ok'})
                else:
                    self._send({'error': 'not found'}, 404)

            def do_POST(self):
                n = int(self.headers.get('Content-Length', 0))
                payload = _json.loads(self.rfile.read(n) or b'{}')
                if self.path == '/api/generate' and payload.get('keep_alive') == 0:
                    stub.unloaded.append(payload.get('model'))
                    self._send({'done': True})
                else:
                    self._send({'error': 'not found'}, 404)

        self._srv = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self._srv.server_address[1]}'
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()


@pytest.fixture
def ollama_stub():
    stub = _OllamaStub()
    try:
        yield stub
    finally:
        stub.close()


DEFAULT_OLLAMA_MODEL = PROVIDER_CONFIGS['ollama']['model']


def _release(stub, translate_model=None, provider='ollama'):
    """Call release_translation_vram() against the stub, return evicted names."""
    from types import SimpleNamespace

    from whisperjav.main import release_translation_vram

    stub.unloaded.clear()
    release_translation_vram(SimpleNamespace(
        translate_provider=provider,
        translate_model=translate_model,
        ollama_url=stub.url,
    ))
    return list(stub.unloaded)


class TestListRunningModels:
    def test_parses_api_ps(self, ollama_stub):
        ollama_stub.resident = [{
            'name': 'whisperjav-abc1234567', 'model': 'whisperjav-abc1234567',
            'size': 5137025024, 'size_vram': 5137025024,
            'expires_at': '2026-08-16T14:38:31.83753-07:00', 'details': {},
        }]
        got = OllamaManager(base_url=ollama_stub.url).list_running_models()
        assert [m['name'] for m in got] == ['whisperjav-abc1234567']

    def test_unreachable_server_returns_empty_not_raise(self):
        """Callers use this in best-effort cleanup; it must never raise."""
        assert OllamaManager(base_url='http://127.0.0.1:1').list_running_models() == []

    def test_empty_ps_is_an_empty_list(self, ollama_stub):
        ollama_stub.resident = []
        assert OllamaManager(base_url=ollama_stub.url).list_running_models() == []


class TestReleaseTranslationVram:
    def test_evicts_the_tuned_variant(self, ollama_stub):
        """The whisperjav-<hash> variant is what actually occupies VRAM.

        _ensure_optimized_variant() routes translation through it, so the base
        model name is never the resident one in the normal case.
        """
        ollama_stub.resident = [{'name': 'whisperjav-abc1234567', 'size_vram': 1}]
        assert _release(ollama_stub) == ['whisperjav-abc1234567']

    def test_never_asks_for_a_model_it_did_not_load(self, ollama_stub):
        """Regression: the old code unloaded a hardcoded 'gemma3:12b'.

        That model may not even be installed, so the request errored and the
        real model stayed resident until Ollama's idle timeout.
        """
        ollama_stub.resident = [{'name': 'whisperjav-abc1234567', 'size_vram': 1}]
        evicted = _release(ollama_stub)
        assert 'gemma3:12b' not in evicted
        assert evicted == ['whisperjav-abc1234567']

    def test_evicts_the_provider_default_when_no_model_flag_given(self, ollama_stub):
        """--translate-model is usually unset; the default supplies the model."""
        ollama_stub.resident = [{'name': DEFAULT_OLLAMA_MODEL, 'size_vram': 1}]
        assert _release(ollama_stub) == [DEFAULT_OLLAMA_MODEL]

    def test_evicts_an_explicitly_requested_model(self, ollama_stub):
        name = 'huihui_ai/qwen3-abliterated:4b'
        ollama_stub.resident = [{'name': name, 'size_vram': 1}]
        assert _release(ollama_stub, translate_model=name) == [name]

    def test_leaves_another_sessions_model_alone(self, ollama_stub):
        """Evicting a bystander model would break someone else's work."""
        ollama_stub.resident = [{'name': 'llama3:70b', 'size_vram': 1}]
        assert _release(ollama_stub) == []

    def test_reports_when_nothing_of_ours_was_resident(self, ollama_stub, capsys):
        """Silence here would leave the user guessing why the next file is slow."""
        ollama_stub.resident = [{'name': 'llama3:70b', 'size_vram': 1}]
        _release(ollama_stub)
        assert 'llama3:70b' in capsys.readouterr().out

    def test_mixed_residency_evicts_only_ours(self, ollama_stub):
        ollama_stub.resident = [{'name': 'whisperjav-deadbeef01', 'size_vram': 1},
                                {'name': 'llama3:70b', 'size_vram': 1}]
        assert _release(ollama_stub) == ['whisperjav-deadbeef01']

    def test_nothing_resident_means_no_request(self, ollama_stub):
        ollama_stub.resident = []
        assert _release(ollama_stub) == []

    def test_non_ollama_provider_is_a_noop(self, ollama_stub):
        """Cloud providers hold no local VRAM; don't touch a local server."""
        ollama_stub.resident = [{'name': 'whisperjav-abc1234567', 'size_vram': 1}]
        assert _release(ollama_stub, provider='openai') == []

    def test_malformed_ps_entries_are_skipped(self, ollama_stub):
        """Falls back to the 'model' key and tolerates junk entries."""
        ollama_stub.resident = [{}, {'model': 'whisperjav-fallbackname'}, {'name': ''}]
        assert _release(ollama_stub) == ['whisperjav-fallbackname']

    def test_survives_args_missing_optional_attributes(self):
        """Called from three code paths whose namespaces are not identical."""
        from types import SimpleNamespace

        from whisperjav.main import release_translation_vram

        release_translation_vram(SimpleNamespace(translate_provider='ollama'))
        release_translation_vram(SimpleNamespace())

    def test_unreachable_server_does_not_raise(self):
        from types import SimpleNamespace

        from whisperjav.main import release_translation_vram

        release_translation_vram(SimpleNamespace(
            translate_provider='ollama', translate_model=None,
            ollama_url='http://127.0.0.1:1'))


class TestReleaseIsWiredIntoEveryTranslationPath:
    """Every path that loads the LLM must also release it.

    The sync and async paths translate inside the per-file loop while the ASR
    model stays resident, so they release after each file. The ensemble path
    translates in a separate loop after all transcription finishes, so it
    releases once at the end — reloading per file there would be pure waste.
    """

    def test_all_three_translate_call_sites_release(self):
        import re

        source = Path('whisperjav/main.py').read_text(encoding='utf-8')
        calls = len(re.findall(r'^\s*translated_path = translate_with_config\(',
                               source, re.MULTILINE))
        releases = len(re.findall(r'^\s*release_translation_vram\(args\)',
                                  source, re.MULTILINE))
        assert calls == 3, f'translation call sites changed ({calls})'
        assert releases == calls, (
            f'{calls} translation paths but {releases} release calls — a path '
            f'that translates without releasing leaks the LLM into VRAM')

    def test_per_file_releases_are_in_finally_blocks(self):
        """A failed translation leaves the model loaded just the same.

        The two interleaved paths must release from a ``finally``, not from the
        success branch. The ensemble path runs after its own loop and needs no
        guard, so exactly two of the three calls are finally-guarded.
        """
        lines = Path('whisperjav/main.py').read_text(encoding='utf-8').split('\n')
        guarded = 0
        for i, line in enumerate(lines):
            if line.strip() != 'release_translation_vram(args)':
                continue
            # Walk back over the comment block to the enclosing statement.
            j = i - 1
            while j >= 0 and (not lines[j].strip() or lines[j].strip().startswith('#')):
                j -= 1
            if lines[j].strip() == 'finally:':
                guarded += 1
        assert guarded == 2, (
            f'{guarded} of the per-file releases are finally-guarded, expected 2 — '
            f'a release in the success branch is skipped when translation raises')

    def test_no_hardcoded_aligned_model_remains_in_the_unload_path(self):
        """Guard against the fallback creeping back in.

        Tokenizes rather than greps so the explanatory comments and docstrings
        that name the old fallback don't count as offenders — only a real
        string literal in executable position does.
        """
        import io
        import tokenize

        source = Path('whisperjav/main.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)

        offenders = []
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if 'gemma3:12b' not in tok.string:
                continue
            if tok.type == tokenize.STRING and any(
                    d in tok.string for d in docstrings):
                continue
            offenders.append(f'line {tok.start[0]}: {tok.string[:80]}')
        assert offenders == [], offenders


class TestJavGlossaryInjection:
    """The glossary is what actually stops euphemism laundering.

    Both bundled tones already instruct "avoid euphemisms". Measured against
    hy-mt1.5 that was not enough: 「中に出して」 came back as "let it happen"
    (dropping 中に — inside — entirely) and 「イッちゃう」 as "I am about to
    reach orgasm". Naming the specific renderings fixed both, so the glossary
    reaching the model is a correctness property, not a nicety.
    """

    def _instructions(self, body="do the thing", retry=True):
        text = "### prompt\nTranslate.\n\n### instructions\n" + body + "\n"
        if retry:
            text += "\n### retry_instructions\nTry again.\n"
        return text

    def test_glossary_file_ships(self):
        from whisperjav.translate.service import _load_jav_glossary
        g = _load_jav_glossary()
        assert g, "bundled jav_glossary.txt is missing or empty"
        # The renderings that were actually wrong before the fix.
        assert "cum inside me" in g
        assert "I'm cumming" in g

    def test_injected_into_instructions_section(self):
        from whisperjav.translate.service import _inject_glossary
        out = _inject_glossary(self._instructions())
        assert "JAV TERMINOLOGY" in out
        # Must land inside ### instructions, NOT under ### retry_instructions.
        # PySubtrans splits on ### headings, so a glossary filed under retry
        # would only be seen after a validation failure.
        assert out.index("JAV TERMINOLOGY") < out.index("### retry_instructions")

    def test_section_order_preserved(self):
        import re
        from whisperjav.translate.service import _inject_glossary
        out = _inject_glossary(self._instructions())
        assert re.findall(r'^### (\w+)', out, re.M) == [
            'prompt', 'instructions', 'retry_instructions']

    def test_original_instruction_body_survives(self):
        from whisperjav.translate.service import _inject_glossary
        out = _inject_glossary(self._instructions(body="KEEP THIS LINE"))
        assert "KEEP THIS LINE" in out

    def test_idempotent(self):
        """Applied twice must not duplicate — a gist may already bundle it."""
        from whisperjav.translate.service import _inject_glossary
        once = _inject_glossary(self._instructions())
        assert _inject_glossary(once) == once

    def test_no_instructions_section_is_left_alone(self):
        """Injecting blind could land the glossary in the wrong section.

        A wrong-section glossary is worse than none: it would apply only on
        retries, making quality depend on whether validation happened to fail.
        """
        from whisperjav.translate.service import _inject_glossary
        text = "### prompt\nTranslate.\n\n### retry_instructions\nAgain.\n"
        assert _inject_glossary(text) == text

    def test_works_without_retry_section(self):
        from whisperjav.translate.service import _inject_glossary
        out = _inject_glossary(self._instructions(retry=False))
        assert "JAV TERMINOLOGY" in out

    def test_missing_glossary_file_degrades_quietly(self):
        """A packaging slip must not break translation outright."""
        from whisperjav.translate import service
        with patch.object(service, '_load_jav_glossary', return_value=None):
            text = self._instructions()
            assert service._inject_glossary(text) == text

    def test_reaches_the_resolved_instruction_file(self):
        """End of the chain: what gets written is what the model reads.

        Guards the reason this lives in service.py rather than in
        defaults/*.txt — get_instruction_content() prefers a remote gist, so a
        glossary edited into the bundled file does nothing when the gist is up.
        """
        from whisperjav.translate import service
        with patch.object(service, 'get_instruction_content',
                          return_value=self._instructions()):
            path = service._resolve_instruction_file('pornify')
        assert "JAV TERMINOLOGY" in Path(path).read_text(encoding='utf-8')

    def test_reaches_the_cli_resolved_instruction_file(self):
        """The CLI has its own resolver, and it is the one that actually runs.

        ``service._resolve_instruction_file`` (covered above) is not what
        ``whisperjav-translate`` calls — ``cli.resolve_instruction_file_or_content``
        is, because it additionally honours ``--instructions-file``. It
        duplicated the resolver and omitted the injection, so every CLI run
        shipped the tone file with no glossary while every unit test above
        still passed. Measured before the fix: both hy-mt1.5 and
        qwen2.5-abliterate rendered おちんちん/おまんこ as "penis"/"vagina".
        """
        import argparse
        from whisperjav.translate import cli

        with patch.object(cli, 'get_instruction_content',
                          return_value=self._instructions()):
            path = cli.resolve_instruction_file_or_content(
                argparse.Namespace(instructions_file=None),
                {'tone': 'standard', 'source': 'japanese'},
            )
        text = Path(path).read_text(encoding='utf-8')
        assert "JAV TERMINOLOGY" in text
        # Same section constraint as the service path: under retry_instructions
        # the glossary would only apply when validation happened to fail.
        assert 0 < text.index("JAV TERMINOLOGY") < text.index("### retry_instructions")

    def test_cli_leaves_explicit_instructions_file_untouched(self, tmp_path):
        """A user-supplied prompt is returned verbatim.

        Injecting into a file the user pointed us at would silently edit their
        prompt. Only the tone-derived path gets the glossary.
        """
        import argparse
        from whisperjav.translate import cli

        own = tmp_path / "mine.txt"
        own.write_text(self._instructions(body="MY OWN RULES"), encoding='utf-8')
        path = cli.resolve_instruction_file_or_content(
            argparse.Namespace(instructions_file=str(own)), {'tone': 'standard'})
        assert Path(path) == own
        assert "JAV TERMINOLOGY" not in own.read_text(encoding='utf-8')


class TestCuratedSamplingReachesTheRequest:
    """Per-request options override a Modelfile in Ollama.

    ``_ensure_optimized_variant()`` bakes curated sampling into the
    ``whisperjav-<hash>`` variant, but the request used to send the tone
    default ``top_p 0.9``, which overrode the variant's validated 0.6 on every
    call — so the tuning was inert. ``top_k``/``repeat_penalty`` were never sent
    at all and survived only via the variant, meaning they silently vanished on
    the fallback path where variant creation fails.
    """

    MODEL = "huihui_ai/hy-mt1.5-abliterated:7b"

    def test_curated_top_p_beats_tone_default(self):
        from whisperjav.translate.service import _build_provider_options
        from whisperjav.translate.ollama_manager import get_sampling_params
        expected = get_sampling_params(self.MODEL)['top_p']
        opts = _build_provider_options(tone='pornify', model=self.MODEL)
        assert opts['top_p'] == expected, (
            "tone default overrode the model author's validated top_p")
        assert opts['top_p'] != 0.9, "still sending the tone default"

    def test_top_k_and_repeat_penalty_are_sent(self):
        """So sampling is identical whether or not the variant was created."""
        from whisperjav.translate.service import _build_provider_options
        opts = _build_provider_options(tone='pornify', model=self.MODEL)
        assert opts['top_k'] == 20
        assert opts['repeat_penalty'] == 1.05

    def test_explicit_override_still_wins(self):
        from whisperjav.translate.service import _build_provider_options
        opts = _build_provider_options(tone='pornify', model=self.MODEL,
                                       top_p=0.95)
        assert opts['top_p'] == 0.95

    def test_settings_still_win_over_curated(self):
        from whisperjav.translate.service import _build_provider_options
        opts = _build_provider_options(tone='pornify', model=self.MODEL,
                                       settings_model_params={'top_p': 0.8})
        assert opts['top_p'] == 0.8

    def test_unknown_model_falls_back_to_defaults(self):
        from whisperjav.translate.service import _build_provider_options
        opts = _build_provider_options(tone='pornify', model='nobody/nothing:1b')
        assert 0.0 <= opts['top_p'] <= 1.0
        assert 'temperature' in opts

    def test_no_model_keeps_legacy_behaviour(self):
        """Callers that pass no model must not crash or change meaning."""
        from whisperjav.translate.service import _build_provider_options
        opts = _build_provider_options(tone='pornify')
        assert opts == {'temperature': 1.2, 'top_p': 0.9}

    def test_tone_still_controls_temperature(self):
        from whisperjav.translate.service import _build_provider_options
        hot = _build_provider_options(tone='pornify', model=self.MODEL)
        cold = _build_provider_options(tone='standard', model=self.MODEL)
        assert hot['temperature'] > cold['temperature']

    def test_service_passes_the_resolved_model(self):
        """The wiring, not just the helper.

        _build_provider_options only consults the curated table if
        translate_with_config actually hands it the model name; it did not
        before this fix, which is why the override went unnoticed.
        """
        import re
        src = Path('whisperjav/translate/service.py').read_text(encoding='utf-8')
        # Anchor on the assignment, not the bare name — the `def` appears first
        # in the file and its signature would match instead.
        call = re.search(
            r'provider_options\s*=\s*_build_provider_options\(\s*\n(.*?)\n\s*\)',
            src, re.S)
        assert call, "could not locate the _build_provider_options call site"
        assert 'model=resolved_model' in call.group(1), (
            "call site does not pass the model, so curated sampling is ignored")
