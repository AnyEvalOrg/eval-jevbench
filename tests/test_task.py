import json
import os
import asyncio

import httpx
import pytest
from inspect_ai import eval

from jevbench.task import (
    INVALID_REASONS,
    _gateway_decide,
    _load_records,
    _load_samples,
    _model_id_for_gateway,
    _score_decision,
    _translate_question,
    jevbench,
)


def test_data_loads_all_public_samples_with_exact_ids():
    samples = _load_samples()
    assert len(samples) == 231
    assert len({sample.id for sample in samples}) == 231
    assert samples[0].id == "original-policy-01-0"
    assert any(sample.id == "easy-intent-00" for sample in samples)
    assert samples[-1].id == "hard-sol-c-multi_hop-13"


def test_sample_input_does_not_render_gold_fields():
    for sample in _load_samples():
        lowered = sample.input.lower()
        assert sample.input.startswith("State:\n")
        assert "\n\nInstructions:\n" in sample.input
        assert "\n\nLabels:\n" in sample.input
        assert "\n\nExpected:\n" not in sample.input
        assert "\n\nGold:\n" not in sample.input
        assert "\n\nTarget:\n" not in sample.input
        assert f"answer: {sample.target}" not in lowered


def test_question_translation_literal_shapes():
    records = _load_records()
    noul = next(record for record in records if record["question"]["type"] == "noul")
    choice = next(record for record in records if record["question"]["type"] == "choice")
    score = next(record for record in records if record["question"]["type"] == "score")

    assert _translate_question(noul["question"], noul["labels"]) == {
        "type": "boolean",
        "instructions": noul["question"]["instructions"],
        "criteria": {
            "true": noul["question"]["criteria"]["true"],
            "false": noul["question"]["criteria"]["false"],
        },
    }
    assert _translate_question(choice["question"], choice["labels"]) == {
        "type": "choice",
        "instructions": choice["question"]["instructions"],
        "criteria": {label: choice["question"]["criteria"][label] for label in choice["labels"]},
    }
    assert _translate_question(score["question"], score["labels"]) == {
        "type": "score",
        "instructions": score["question"]["instructions"],
        "criteria": list(score["question"]["criteria"]),
    }


def test_scorer_valid_and_renormalisation_band():
    score = _score_decision(
        raw_probs={"no": 0.2, "yes": 0.79},
        expected="yes",
        labels=["no", "yes"],
        question_type="noul",
    )
    assert score.value == "C"
    assert score.answer == "yes"
    assert sum(score.metadata["probs"].values()) == pytest.approx(1.0)
    assert score.metadata["brier"] == pytest.approx(
        score.metadata["probs"]["no"] ** 2 + (score.metadata["probs"]["yes"] - 1) ** 2
    )
    assert score.metadata["top_confidence"] == pytest.approx(score.metadata["probs"]["yes"])
    assert score.metadata["correct"] is True


@pytest.mark.parametrize(
    ("raw_probs", "reason"),
    [
        (None, "missing_answer"),
        (["yes", 1.0], "wrong_type"),
        ({"yes": 1.0}, "label_set_mismatch"),
        ({"no": -0.1, "yes": 1.1}, "out_of_range"),
        ({"no": 0.5, "yes": 0.3}, "sum_out_of_band"),
    ],
)
def test_scorer_invalid_cases(raw_probs, reason):
    score = _score_decision(
        raw_probs=raw_probs,
        expected="yes",
        labels=["no", "yes"],
        question_type="noul",
    )
    assert score.value == "I"
    assert score.metadata["invalid_reason"] == reason
    assert score.metadata["invalid_reason"] in INVALID_REASONS


def test_scorer_distribution_metrics():
    score = _score_decision(
        raw_probs={"0": 0.1, "1": 0.7, "2": 0.2},
        expected="1",
        labels=["0", "1", "2"],
        question_type="score",
        gold_distribution={"0": 0.0, "1": 1.0, "2": 0.0},
    )
    assert score.value == "C"
    assert score.metadata["tvd_to_gold"] == pytest.approx(0.3)
    assert score.metadata["ordinal_mae"] == pytest.approx(0.3)


def test_502_no_retry_scores_invalid():
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "trustedrouter/trev-1.0", "architecture": {"modality": "text->decision"}}]},
            )
        return httpx.Response(
            502,
            headers={"x-should-retry": "false"},
            json={"model": "trustedrouter/trev-1.0"},
        )

    result = asyncio.run(
        _gateway_decide(
            state_value="state",
            question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
            labels=["no", "yes"],
            model_id="trustedrouter/trev-1.0",
            base_url="https://api.trustedrouter.com/v1",
            api_key="test",
            timeout_s=5,
            transport=httpx.MockTransport(handler),
        )
    )

    assert calls.count("/v1/decide") == 1
    assert result.ok is False
    assert result.invalid_reason == "http_502_no_retry"
    assert result.probs_source == "native"


def test_model_prefix_stripping():
    assert _model_id_for_gateway("openai/trustedrouter/trev-1.0") == "trustedrouter/trev-1.0"
    assert _model_id_for_gateway("trustedrouter/google/gemma-4-31b-it") == "google/gemma-4-31b-it"
    assert _model_id_for_gateway("google/gemma-4-31b-it") == "google/gemma-4-31b-it"


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("JEVBENCH_LIVE") != "1", reason="set JEVBENCH_LIVE=1 to call TrustedRouter")
def test_live_smoke_three_question_types():
    os.environ["HOME"] = os.environ.get("JEVBENCH_TEST_HOME", "/tmp/jevbench-home")
    os.environ.setdefault("INSPECT_TRACE_FILE", "/tmp/jevbench-inspect-trace.log")
    logs = eval(
        jevbench(timeout_s=120),
        model=os.environ.get("INSPECT_EVAL_MODEL", "openai/trustedrouter/trev-1.0"),
        sample_id=["original-policy-01-0", "easy-intent-00", "original-ordinal-01-0"],
        display="none",
        trace=False,
        log_dir=os.environ.get("INSPECT_LOG_DIR", "/tmp/jevbench-inspect-logs"),
        log_level="warning",
    )
    assert logs
    assert logs[0].status == "success"
