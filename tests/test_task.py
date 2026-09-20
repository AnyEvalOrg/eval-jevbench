import json
import os
import asyncio
import importlib

import httpx
import openai
import pytest
from inspect_ai import eval

import jevbench.task as task_module
from jevbench.task import (
    INVALID_REASONS,
    _MODEL_CATALOG_CACHE,
    _gateway_decide,
    _load_records,
    _load_samples,
    _model_id_for_gateway,
    _openai_provider_client,
    _private_model_modalities,
    _score_decision,
    _translate_question,
    jevbench,
)


def _mock_private_catalog(monkeypatch, handler):
    def factory(timeout_s):
        return httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(task_module, "_new_private_httpx_client", factory)


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


def test_hard_probability_gold_probs_become_private_score_metadata():
    records = _load_records("hard")
    gold_records = [
        record
        for record in records
        if isinstance(record.get("provenance"), dict)
        and isinstance(record["provenance"].get("gold_probs"), dict)
    ]
    samples = _load_samples("hard")
    gold_samples = [sample for sample in samples if sample.metadata.get("gold_distribution") is not None]

    assert len(gold_records) == 10
    assert {sample.id for sample in gold_samples} == {record["id"] for record in gold_records}

    for sample in gold_samples:
        labels = sample.metadata["labels"]
        gold = sample.metadata["gold_distribution"]
        assert set(gold) == set(labels)
        assert sum(gold.values()) == pytest.approx(1.0)
        assert "gold_probs" not in str(sample.input)
        assert "gold_probs" not in str(sample.metadata.get("provenance") or {})

    scored = _score_decision(
        raw_probs=gold_samples[0].metadata["gold_distribution"],
        expected=gold_samples[0].target,
        labels=gold_samples[0].metadata["labels"],
        question_type=gold_samples[0].metadata["question_type"],
        gold_distribution=gold_samples[0].metadata["gold_distribution"],
    )
    assert scored.metadata["tvd_to_gold"] == pytest.approx(0.0)

    no_gold = next(sample for sample in samples if sample.metadata.get("gold_distribution") is None)
    no_gold_score = _score_decision(
        raw_probs={label: 1.0 / len(no_gold.metadata["labels"]) for label in no_gold.metadata["labels"]},
        expected=no_gold.target,
        labels=no_gold.metadata["labels"],
        question_type=no_gold.metadata["question_type"],
        gold_distribution=no_gold.metadata.get("gold_distribution"),
    )
    assert "tvd_to_gold" not in no_gold_score.metadata


class FakeApi:
    def __init__(self, client):
        self.client = client


class FakeModel:
    def __init__(self, client):
        self.api = FakeApi(client)


def _openai_sdk_httpx_module():
    probe = openai.AsyncOpenAI(api_key="probe-key", base_url="https://sdk.example/v1")
    try:
        probe_client_type = type(probe._client)
        for cls in probe_client_type.__mro__:
            if cls.__name__ == "AsyncClient":
                module = importlib.import_module(cls.__module__.split(".", 1)[0])
                assert issubclass(probe_client_type, module.AsyncClient)
                return module
    finally:
        asyncio.run(probe.close())
    raise AssertionError(f"could not find SDK HTTP client base for {probe_client_type!r}")


def _sdk_provider_client(sdk_httpx, handler):
    http_client = sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(handler))
    provider_client = openai.AsyncOpenAI(
        api_key="sdk-key",
        base_url="https://api.trustedrouter.com/v1",
        http_client=http_client,
    )
    # The regression must use the OpenAI SDK's own HTTP stack (httpx or httpx2),
    # because SDK status errors carry that response class through provider calls.
    assert type(provider_client._client) is type(http_client)
    assert isinstance(provider_client._client, sdk_httpx.AsyncClient)
    return provider_client


def test_provider_client_decide_uses_sdk_base_auth_and_hooks(monkeypatch):
    _MODEL_CATALOG_CACHE.clear()
    calls = []
    private_model_calls = []
    provider_hook_paths = []

    async def hook(request: httpx.Request) -> None:
        provider_hook_paths.append(request.url.path)

    async def models_handler(request: httpx.Request) -> httpx.Response:
        private_model_calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "authorization": request.headers.get("authorization"),
            }
        )
        return httpx.Response(
            200,
            json={"data": [{"id": "trustedrouter/trev-1.0", "architecture": {"modality": "text->decision"}}]},
        )

    _mock_private_catalog(monkeypatch, models_handler)

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "authorization": request.headers.get("authorization"),
                "json": json.loads(request.content.decode()) if request.content else None,
            }
        )
        return httpx.Response(
            200,
            json={
                "model": "trustedrouter/trev-1.0",
                "answers": {"decision": {"probability": 0.75}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [hook]},
    )
    provider_client = openai.AsyncOpenAI(
        api_key="sdk-key",
        base_url="https://sdk.example/v1",
        http_client=http_client,
    )

    try:
        result = asyncio.run(
            _gateway_decide(
                state_value="state",
                question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
                labels=["no", "yes"],
                model_id="trustedrouter/trev-1.0",
                timeout_s=5,
                provider_client=provider_client,
            )
        )
    finally:
        asyncio.run(provider_client.close())

    decide_calls = [call for call in calls if call["path"] == "/v1/decide"]
    assert len(decide_calls) == 1
    assert decide_calls[0]["method"] == "POST"
    assert decide_calls[0]["url"] == "https://sdk.example/v1/decide"
    assert decide_calls[0]["authorization"] == "Bearer sdk-key"
    assert decide_calls[0]["json"]["model"] == "trustedrouter/trev-1.0"
    assert provider_hook_paths == ["/v1/decide"]
    assert [call["path"] for call in private_model_calls] == ["/v1/models"]
    assert private_model_calls[0]["authorization"] == "Bearer sdk-key"
    assert _openai_provider_client(FakeModel(provider_client)) is provider_client
    assert result.ok is True
    assert result.probs == {"yes": 0.75, "no": 0.25}
    assert result.probs_source == "native"
    assert result.transport == "provider_client"


def test_502_no_retry_scores_invalid_on_provider_client(monkeypatch):
    _MODEL_CATALOG_CACHE.clear()
    calls = []

    async def models_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "trustedrouter/trev-1.0", "architecture": {"modality": "text->decision"}}]},
        )

    _mock_private_catalog(monkeypatch, models_handler)

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            502,
            headers={"x-should-retry": "false"},
            json={"model": "trustedrouter/trev-1.0"},
        )

    provider_client = openai.AsyncOpenAI(
        api_key="sdk-key",
        base_url="https://api.trustedrouter.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        result = asyncio.run(
            _gateway_decide(
                state_value="state",
                question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
                labels=["no", "yes"],
                model_id="trustedrouter/trev-1.0",
                timeout_s=5,
                provider_client=provider_client,
            )
        )
    finally:
        asyncio.run(provider_client.close())

    assert calls.count("/v1/decide") == 1
    assert result.ok is False
    assert result.invalid_reason == "http_502_no_retry"
    assert result.probs_source == "native"
    assert result.transport == "provider_client"


def test_sdk_httpx2_502_no_retry_scores_invalid_once_on_provider_client(monkeypatch):
    _MODEL_CATALOG_CACHE.clear()
    sdk_httpx = _openai_sdk_httpx_module()
    calls = []

    async def models_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "trustedrouter/trev-1.0", "architecture": {"modality": "text->decision"}}]},
        )

    _mock_private_catalog(monkeypatch, models_handler)

    async def handler(request):
        calls.append(request.url.path)
        return sdk_httpx.Response(
            502,
            headers={"x-should-retry": "false"},
            json={"model": "trustedrouter/trev-1.0"},
        )

    provider_client = _sdk_provider_client(sdk_httpx, handler)
    try:
        result = asyncio.run(
            _gateway_decide(
                state_value="state",
                question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
                labels=["no", "yes"],
                model_id="trustedrouter/trev-1.0",
                timeout_s=5,
                provider_client=provider_client,
            )
        )
    finally:
        asyncio.run(provider_client.close())

    assert calls.count("/v1/decide") == 1
    assert result.ok is False
    assert result.invalid_reason == "http_502_no_retry"
    assert result.probs_source == "native"
    assert result.transport == "provider_client"


def test_sdk_httpx2_500_retries_then_scores_http_5xx_on_provider_client(monkeypatch):
    _MODEL_CATALOG_CACHE.clear()
    sdk_httpx = _openai_sdk_httpx_module()
    calls = []

    async def models_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "trustedrouter/trev-1.0", "architecture": {"modality": "text->decision"}}]},
        )

    _mock_private_catalog(monkeypatch, models_handler)

    async def handler(request):
        calls.append(request.url.path)
        return sdk_httpx.Response(500, json={"model": "trustedrouter/trev-1.0"})

    provider_client = _sdk_provider_client(sdk_httpx, handler)
    try:
        result = asyncio.run(
            _gateway_decide(
                state_value="state",
                question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
                labels=["no", "yes"],
                model_id="trustedrouter/trev-1.0",
                timeout_s=5,
                provider_client=provider_client,
            )
        )
    finally:
        asyncio.run(provider_client.close())

    assert calls.count("/v1/decide") == 3
    assert result.ok is False
    assert result.invalid_reason == "http_5xx"
    assert result.probs_source == "native"


def test_sdk_httpx2_400_raises_on_provider_client(monkeypatch):
    _MODEL_CATALOG_CACHE.clear()
    sdk_httpx = _openai_sdk_httpx_module()
    calls = []

    async def models_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    _mock_private_catalog(monkeypatch, models_handler)

    async def handler(request):
        calls.append(request.url.path)
        return sdk_httpx.Response(400, json={"error": {"message": "bad request"}})

    provider_client = _sdk_provider_client(sdk_httpx, handler)
    try:
        with pytest.raises(Exception):
            asyncio.run(
                _gateway_decide(
                    state_value="state",
                    question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
                    labels=["no", "yes"],
                    model_id="trustedrouter/trev-1.0",
                    timeout_s=5,
                    provider_client=provider_client,
                )
            )
    finally:
        asyncio.run(provider_client.close())

    assert calls.count("/v1/decide") == 1


def test_sdk_httpx2_timeout_scores_invalid_timeout_on_provider_client(monkeypatch):
    _MODEL_CATALOG_CACHE.clear()
    sdk_httpx = _openai_sdk_httpx_module()
    calls = []

    async def models_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    _mock_private_catalog(monkeypatch, models_handler)

    async def handler(request):
        calls.append(request.url.path)
        raise sdk_httpx.TimeoutException("timed out", request=request)

    provider_client = _sdk_provider_client(sdk_httpx, handler)
    try:
        result = asyncio.run(
            _gateway_decide(
                state_value="state",
                question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
                labels=["no", "yes"],
                model_id="trustedrouter/trev-1.0",
                timeout_s=5,
                provider_client=provider_client,
            )
        )
    finally:
        asyncio.run(provider_client.close())

    assert calls.count("/v1/decide") == 3
    assert result.ok is False
    assert result.invalid_reason == "timeout"
    assert result.transport == "provider_client"


def test_private_client_fallback_records_transport_and_unknown_source():
    _MODEL_CATALOG_CACHE.clear()
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers.get("authorization")))
        if request.url.path.endswith("/models"):
            return httpx.Response(500, json={"error": "catalog unavailable"})
        return httpx.Response(
            200,
            json={
                "model": "trustedrouter/trev-1.0",
                "answers": {"decision": {"probability": 0.6}},
            },
        )

    result = asyncio.run(
        _gateway_decide(
            state_value="state",
            question={"type": "noul", "instructions": "Allowed?", "criteria": {"true": "yes", "false": "no"}},
            labels=["no", "yes"],
            model_id="trustedrouter/trev-1.0",
            base_url="https://fallback.example/v1",
            api_key="fallback-key",
            timeout_s=5,
            transport=httpx.MockTransport(handler),
        )
    )

    assert ("/v1/decide", "Bearer fallback-key") in calls
    assert _openai_provider_client(FakeModel(object())) is None
    assert result.ok is True
    assert result.probs_source == "unknown"
    assert result.transport == "private_client"


def test_model_catalog_is_cached_once_per_base_url():
    _MODEL_CATALOG_CACHE.clear()
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(
            200,
            json={"data": [{"id": "trustedrouter/trev-1.0", "architecture": {"modality": "text->decision"}}]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = await _private_model_modalities(client, "https://cache.example/v1", "key-a")
            second = await _private_model_modalities(client, "https://cache.example/v1/", "key-b")
        return first, second

    first, second = asyncio.run(run())

    assert first == second == {"trustedrouter/trev-1.0": "text->decision"}
    assert calls == [("https://cache.example/v1/models", "Bearer key-a")]


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


def test_non_finite_probabilities_are_invalid_never_correct():
    import math
    from jevbench.task import _validate_distribution, _score_decision
    assert _validate_distribution({"no": math.nan, "yes": 1.0}, ["no", "yes"]) == (None, "out_of_range")
    assert _validate_distribution({"no": math.inf, "yes": 0.0}, ["no", "yes"]) == (None, "out_of_range")
    score = _score_decision(raw_probs={"no": math.nan, "yes": 1.0}, expected="no", labels=["no", "yes"],
                            question_type="noul", gold_distribution=None, result_metadata={}, preset_invalid_reason=None)
    assert score.value != "C" and score.metadata["invalid_reason"] == "out_of_range"


def test_gold_rationales_never_reach_sample_metadata():
    from jevbench.task import _load_samples
    for sample in _load_samples():
        prov = sample.metadata.get("provenance") or {}
        assert set(prov) <= {"source", "license", "label_basis", "exclude_reason"}, sample.id
    # The actual gold text of every hard record must be absent, not just the key.
    from jevbench.task import _load_records
    samples = {s.id: s for s in _load_samples()}
    for record in _load_records("hard"):
        prov = record.get("provenance") or {}
        blob = str(samples[record["id"]].metadata) + str(samples[record["id"]].input)
        # surface_answer is one of the task's own labels (the tempting wrong one), so
        # its text legitimately appears in the label list; only the prose is checked.
        for key in ("rationale", "why_hard"):
            text = prov.get(key)
            if isinstance(text, str) and len(text) > 20:
                assert text not in blob, (record["id"], key)
