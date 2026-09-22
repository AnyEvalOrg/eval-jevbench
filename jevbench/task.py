from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from importlib import import_module, resources
from typing import Any

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

TIERS = ("original", "easy", "hard")
DATA_FILES = {
    "original": "original.jsonl",
    "easy": "easy.jsonl",
    "hard": "hard.jsonl",
}
# Denominator exclusions were withdrawn: a returned response can hide behind a
# redirect, a body read, or a hook, so this layer cannot prove a call never
# reached the model. The consumer that observes response delivery must make
# that decision; every invalid reason here scores INCORRECT.
INVALID_REASONS = {
    "missing_answer",
    "wrong_type",
    "label_set_mismatch",
    "out_of_range",
    "sum_out_of_band",
    "http_502_no_retry",
    "http_5xx",
    "timeout",
    "transport_error",
}
_MODEL_CATALOG_CACHE: dict[str, dict[str, str]] = {}
_MODEL_CATALOG_LOCK: asyncio.Lock | None = None


@dataclass
class DecisionResult:
    ok: bool
    probs: dict[str, Any] | None
    probs_source: str
    usage: dict[str, Any]
    latency_s: float
    served_model: str
    transport: str
    invalid_reason: str | None = None


@dataclass(frozen=True)
class _SDKStatusResponse:
    status_code: int
    headers: Any
    response: Any
    exc: BaseException

    def json(self) -> Any:
        json_fn = getattr(self.response, "json", None)
        if not callable(json_fn):
            raise ValueError("SDK status response has no JSON body")
        return json_fn()

    def raise_for_status(self) -> None:
        raise self.exc


def _checked_gold_distribution(
    value: Any, labels: list[str], source: str
) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    distribution, invalid_reason = _validate_distribution(value, labels)
    if invalid_reason is not None:
        raise ValueError(f"{source} is not a valid gold distribution: {invalid_reason}")
    return distribution


def _gold_distribution(record: dict[str, Any], labels: list[str]) -> dict[str, float] | None:
    for key in ("gold_distribution", "expected_distribution", "distribution"):
        distribution = _checked_gold_distribution(
            record.get(key), labels, f"{record.get('id', '<unknown>')}.{key}"
        )
        if distribution is not None:
            return distribution
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        # Hard probability-family golds are scoring metadata only. They must never
        # be rendered into the sample input or copied into public provenance.
        return _checked_gold_distribution(
            provenance.get("gold_probs"),
            labels,
            f"{record.get('id', '<unknown>')}.provenance.gold_probs",
        )
    return None


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _render_input(record: dict[str, Any]) -> str:
    labels = [str(label) for label in record["labels"]]
    return (
        f"State:\n{_state_text(record['state'])}\n\n"
        f"Instructions:\n{record['question']['instructions']}\n\n"
        f"Labels:\n{json.dumps(labels, ensure_ascii=False)}"
    )


def _load_records(tier: str | None = None) -> list[dict[str, Any]]:
    if tier is not None and tier not in DATA_FILES:
        raise ValueError(f"tier must be one of {', '.join(TIERS)} or None")

    tiers = [tier] if tier else list(TIERS)
    records: list[dict[str, Any]] = []
    data_root = resources.files("jevbench.data")
    for tier_name in tiers:
        with (data_root / DATA_FILES[tier_name]).open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record["_tier"] = tier_name
                records.append(record)
    return records


_PUBLIC_PROVENANCE_KEYS = ("source", "license", "label_basis", "exclude_reason")


def _public_provenance(provenance: Any) -> dict[str, Any] | None:
    if not isinstance(provenance, dict):
        return None
    return {k: provenance[k] for k in _PUBLIC_PROVENANCE_KEYS if k in provenance}


def _sample_from_record(record: dict[str, Any]) -> Sample:
    labels = [str(label) for label in record["labels"]]
    metadata: dict[str, Any] = {
        "tier": record["_tier"],
        "family": record["family"],
        "group": record.get("group"),
        "split": record.get("split"),
        "question_type": record["question"]["type"],
        "labels": labels,
        # Only the provenance fields that describe WHERE a task came from. The hard
        # tier's provenance also carries the authors' gold rationale and a surface
        # answer, and Inspect logs retain sample metadata, so those would publish.
        "provenance": _public_provenance(record.get("provenance")),
        "state": record["state"],
        "question": record["question"],
    }
    for key in ("topic",):
        if key in record:
            metadata[key] = record[key]
    gold_distribution = _gold_distribution(record, labels)
    if gold_distribution is not None:
        metadata["gold_distribution"] = gold_distribution

    return Sample(
        id=record["id"],
        input=_render_input(record),
        target=str(record["expected"]),
        metadata=metadata,
    )


def _dataset_name(tier: str | None) -> str:
    """AnyEval refuses to publish a run whose reproducibility record has no dataset
    name; the public JevBench files are the dataset, named by the tier filter."""
    return "jevbench-public" if tier is None else f"jevbench-public-{tier}"


def _load_samples(tier: str | None = None) -> list[Sample]:
    return [_sample_from_record(record) for record in _load_records(tier)]


def _translate_question(question: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    qtype = question["type"]
    instructions = question["instructions"]
    criteria = question.get("criteria")

    if qtype == "noul":
        translated = {"type": "boolean", "instructions": instructions}
        if isinstance(criteria, dict):
            tf = {k: criteria[k] for k in ("true", "false") if k in criteria}
            if tf:
                translated["criteria"] = tf
        return translated

    if qtype == "choice":
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {
                label: criteria.get(label, label) if isinstance(criteria, dict) else label
                for label in labels
            },
        }

    if qtype == "score":
        if isinstance(criteria, dict):
            levels = [criteria[str(i)] for i in range(len(labels))]
        else:
            levels = list(criteria or [])
        return {"type": "score", "instructions": instructions, "criteria": levels}

    raise ValueError(f"unknown question type: {qtype}")


def _model_id_for_gateway(model_name: str) -> str:
    # Not a universal namespace conversion: Inspect's openai-api/gateway/openai/...
    # leaves gateway/openai/... here, which AnyEval's model pin refuses (pre-existing).
    # Both names can be Inspect providers OR gateway providers. Strip a prefix
    # only when another provider segment remains, preserving provider/model ids.
    if model_name.startswith("openai/") and "/" in model_name[len("openai/") :]:
        model_name = model_name[len("openai/") :]
    if model_name.startswith("trustedrouter/") and "/" in model_name[len("trustedrouter/") :]:
        model_name = model_name[len("trustedrouter/") :]
    return model_name


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _decide_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/decide"


def _openai_provider_client(model: Any) -> Any | None:
    try:
        import openai
    except ImportError:
        return None

    client = getattr(getattr(model, "api", None), "client", None)
    return client if isinstance(client, openai.AsyncOpenAI) else None


def _status_response(response: Any) -> Any | None:
    # OpenAI SDK 3.x may vend its own httpx2 response class. Treat SDK
    # responses structurally so billed provider-client failures are classified.
    status_code = getattr(response, "status_code", None)
    headers = getattr(response, "headers", None)
    return response if isinstance(status_code, int) and callable(getattr(headers, "get", None)) else None


def _sdk_status_response(exc: BaseException) -> Any | None:
    response = getattr(exc, "response", None)
    if (duck_response := _status_response(response)) is not None:
        return duck_response

    try:
        import openai
    except ImportError:
        return None

    if isinstance(exc, openai.APIStatusError):
        # Some SDK status exceptions expose the status separately from the
        # response object; use both pieces without depending on httpx/httpx2.
        status_code = getattr(exc, "status_code", None)
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if isinstance(status_code, int) and callable(getattr(headers, "get", None)):
            return _SDKStatusResponse(status_code, headers, exc.response, exc)
    return None


_MISSING = object()


def _header_value(headers: Any, name: str, default: str = "") -> str:
    value = headers.get(name, _MISSING)
    if value is not _MISSING:
        return str(value)
    lower_name = name.lower()
    for key, item in getattr(headers, "items", lambda: [])():
        if str(key).lower() == lower_name:
            return str(item)
    return default


def _is_sdk_transport_error(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:
        return False

    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))


def _transport_invalid_reason(exc: BaseException) -> str | None:
    # The SDK also wraps hook bugs as connection errors. Inspect the cause.
    # AnyEval's policy refusal is a plain ValueError with no provenance marker;
    # let it propagate, since its message could also come from a response hook.
    if _is_sdk_transport_error(exc):
        if exc.__cause__ is not None:
            return _transport_invalid_reason(exc.__cause__)
        import openai

        return "timeout" if isinstance(exc, openai.APITimeoutError) else "transport_error"
    # SDK 3.x uses httpx2; private requests still use httpx.
    for package in ("httpx", "httpx2"):
        try:
            http = import_module(package)
        except ImportError:
            continue
        if isinstance(exc, http.TimeoutException):
            return "timeout"
        if isinstance(exc, http.TransportError):
            return "transport_error"
    return None


def _modalities_from_payload(payload: Any) -> dict[str, str]:
    if isinstance(payload, dict):
        data = payload.get("data", [])
    elif isinstance(payload, list):
        data = payload
    else:
        data = []
    modalities: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name")
        modality = (item.get("architecture") or {}).get("modality")
        if isinstance(model_id, str) and isinstance(modality, str):
            modalities[model_id] = modality
    return modalities


async def _private_model_modalities(
    client: httpx.AsyncClient, base_url: str, api_key: str
) -> dict[str, str]:
    cache_key = base_url.rstrip("/")
    if cache_key in _MODEL_CATALOG_CACHE:
        return _MODEL_CATALOG_CACHE[cache_key]

    global _MODEL_CATALOG_LOCK
    if _MODEL_CATALOG_LOCK is None:
        _MODEL_CATALOG_LOCK = asyncio.Lock()

    async with _MODEL_CATALOG_LOCK:
        if cache_key in _MODEL_CATALOG_CACHE:
            return _MODEL_CATALOG_CACHE[cache_key]
        try:
            response = await client.get(
                _models_url(base_url),
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code in {401, 402, 403}:
                response.raise_for_status()
            if response.status_code >= 500:
                _MODEL_CATALOG_CACHE[cache_key] = {}
                return {}
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            _MODEL_CATALOG_CACHE[cache_key] = {}
            return {}

        modalities = _modalities_from_payload(payload)
        _MODEL_CATALOG_CACHE[cache_key] = modalities
        return modalities


def _new_private_httpx_client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))


async def _provider_model_modalities(provider_client: Any, timeout_s: float) -> dict[str, str]:
    base_url = str(provider_client.base_url)
    api_key = str(getattr(provider_client, "api_key", "") or "")
    # AnyEval journals every request on Inspect's provider client as billable.
    # Keep catalog discovery on a private httpx client; only /decide uses the
    # provider client and its accounting hooks.
    try:
        async with _new_private_httpx_client(timeout_s) as client:
            return await _private_model_modalities(client, base_url, api_key)
    except Exception:
        _MODEL_CATALOG_CACHE[base_url.rstrip("/")] = {}
        return {}


def _extract_probs(answer: dict[str, Any], qtype: str) -> dict[str, Any] | None:
    if qtype == "noul":
        if "probability" not in answer:
            return None
        p = answer["probability"]
        return {"yes": p, "no": 1 - p} if isinstance(p, int | float) and not isinstance(p, bool) else {"yes": p, "no": None}
    if qtype in {"choice", "score"}:
        probs = answer.get("probabilities")
        return dict(probs) if isinstance(probs, dict) else None
    raise ValueError(f"unknown question type: {qtype}")


async def _gateway_decide(
    *,
    state_value: Any,
    question: dict[str, Any],
    labels: list[str],
    model_id: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
    provider_client: Any | None = None,
    reasoning: bool | str | None = None,
) -> DecisionResult:
    async def decide_with_client(client: httpx.AsyncClient | None) -> DecisionResult:
        if provider_client is not None:
            modalities = await _provider_model_modalities(provider_client, timeout_s)
            transport_name = "provider_client"
        else:
            if client is None or base_url is None or api_key is None:
                raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required without an Inspect provider client")
            modalities = await _private_model_modalities(client, base_url, api_key)
            transport_name = "private_client"

        body = {
            "state": state_value,
            "model": model_id,
            "questions": {"decision": _translate_question(question, labels)},
        }
        if reasoning is not None:
            body["reasoning"] = True if reasoning is True else {"effort": reasoning}
        last_status = 0
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                if provider_client is not None:
                    response = await provider_client.post(
                        "/decide",
                        body=body,
                        cast_to=httpx.Response,
                        options={"max_retries": 0, "timeout": timeout_s},
                    )
                else:
                    assert client is not None and base_url is not None and api_key is not None
                    response = await client.post(
                        _decide_url(base_url),
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=body,
                    )
            except Exception as exc:
                response = _sdk_status_response(exc) if provider_client is not None else None
                if response is None:
                    invalid_reason = _transport_invalid_reason(exc)
                    if invalid_reason is None:
                        raise
                    if attempt < 2:
                        await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    return DecisionResult(False, None, "unknown", {}, time.perf_counter() - t0, model_id, transport_name, invalid_reason)

            last_status = response.status_code
            if last_status in {400, 401, 402, 403}:
                response.raise_for_status()
            # Pre-existing: only 502 honors no-retry; a usage-bearing 503 with
            # x-should-retry: false still makes three attempts. Out of scope here.
            if last_status == 502 and _header_value(response.headers, "x-should-retry").lower() == "false":
                served_model = _served_model(response, model_id)
                return DecisionResult(False, None, _probs_source(modalities, served_model), {}, time.perf_counter() - t0, served_model, transport_name, "http_502_no_retry")
            if 500 <= last_status <= 599:
                if attempt < 2:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                served_model = _served_model(response, model_id)
                return DecisionResult(False, None, _probs_source(modalities, served_model), {}, time.perf_counter() - t0, served_model, transport_name, "http_5xx")

            response.raise_for_status()
            payload = response.json()
            served_model = str(payload.get("model") or model_id)
            answer = ((payload.get("answers") or {}).get("decision") or {})
            probs = _extract_probs(answer, question["type"]) if isinstance(answer, dict) else None
            return DecisionResult(
                ok=probs is not None,
                probs=probs,
                probs_source=_probs_source(modalities, served_model),
                usage=payload.get("usage") or {},
                latency_s=time.perf_counter() - t0,
                served_model=served_model,
                transport=transport_name,
                invalid_reason=None if probs is not None else "missing_answer",
            )

        return DecisionResult(False, None, "unknown", {}, time.perf_counter() - t0, model_id, transport_name, "http_5xx" if last_status else "timeout")

    if provider_client is not None:
        return await decide_with_client(None)

    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        return await decide_with_client(client)


def _served_model(response: Any, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback
    return str(payload.get("model") or fallback) if isinstance(payload, dict) else fallback


def _probs_source(modalities: dict[str, str], served_model: str) -> str:
    modality = modalities.get(served_model)
    if modality is None:
        return "unknown"
    return "native" if modality == "text->decision" else "verbalized"


def _validate_distribution(raw: Any, labels: list[str]) -> tuple[dict[str, float] | None, str | None]:
    if raw is None:
        return None, "missing_answer"
    if not isinstance(raw, dict):
        return None, "wrong_type"
    if set(raw) != set(labels):
        return None, "label_set_mismatch"

    probs: dict[str, float] = {}
    for label in labels:
        value = raw[label]
        if not isinstance(value, int | float) or isinstance(value, bool):
            return None, "wrong_type"
        value = float(value)
        # NaN passes every comparison below, renormalises to NaN and then wins an
        # argmax; a non-finite probability is not a forecast (review finding).
        if not math.isfinite(value):
            return None, "out_of_range"
        if value < 0.0 or value > 1.0:
            return None, "out_of_range"
        probs[label] = value

    total = sum(probs.values())
    if abs(total - 1.0) > 0.02 or total <= 0.0:
        return None, "sum_out_of_band"
    if total != 1.0:
        probs = {label: value / total for label, value in probs.items()}
    return probs, None


def _brier(probs: dict[str, float], expected: str, labels: list[str]) -> float:
    return sum((probs[label] - (1.0 if label == expected else 0.0)) ** 2 for label in labels)


def _tvd(probs: dict[str, float], gold: dict[str, float], labels: list[str]) -> float:
    return 0.5 * sum(abs(probs.get(label, 0.0) - gold.get(label, 0.0)) for label in labels)


def _ordinal_mae(probs: dict[str, float], expected: str) -> float | None:
    try:
        expected_i = int(expected)
        return sum(prob * abs(int(label) - expected_i) for label, prob in probs.items())
    except ValueError:
        return None


def _score_decision(
    *,
    raw_probs: Any,
    expected: str,
    labels: list[str],
    question_type: str,
    gold_distribution: dict[str, float] | None = None,
    result_metadata: dict[str, Any] | None = None,
    preset_invalid_reason: str | None = None,
) -> Score:
    result_metadata = result_metadata or {}
    metadata = dict(result_metadata)
    if preset_invalid_reason is not None:
        metadata["invalid_reason"] = preset_invalid_reason
        metadata["correct"] = False
        return Score(value=INCORRECT, answer=None, metadata=metadata)

    probs, invalid_reason = _validate_distribution(raw_probs, labels)
    if invalid_reason is not None:
        metadata["invalid_reason"] = invalid_reason
        metadata["correct"] = False
        return Score(value=INCORRECT, answer=None, metadata=metadata)

    assert probs is not None
    answer = max(labels, key=lambda label: probs[label])
    correct = answer == expected
    metadata.update(
        {
            "probs": probs,
            "brier": _brier(probs, expected, labels),
            "top_confidence": probs[answer],
            "correct": correct,
        }
    )
    if gold_distribution is not None:
        metadata["tvd_to_gold"] = _tvd(probs, gold_distribution, labels)
    if question_type == "score":
        ordinal_mae = _ordinal_mae(probs, expected)
        if ordinal_mae is not None:
            metadata["ordinal_mae"] = ordinal_mae
    return Score(value=CORRECT if correct else INCORRECT, answer=answer, metadata=metadata)


@metric
def mean_brier():
    def metric(scores):
        values = []
        for item in scores:
            score = getattr(item, "score", item)
            if score.metadata and score.metadata.get("brier") is not None:
                values.append(score.metadata["brier"])
        return sum(values) / len(values) if values else 0.0

    return metric


@solver
def trustedrouter_decision_solver(
    timeout_s: float = 120.0, reasoning: bool | str | None = None
):
    # Fail before evaluation so a typo cannot masquerade as a deliberated run.
    if not (
        reasoning is None
        or reasoning is True
        or (isinstance(reasoning, str) and reasoning in ("low", "medium", "high"))
    ):
        raise ValueError('reasoning must be None, True, or one of "low", "medium", "high"')

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        labels = [str(label) for label in state.metadata["labels"]]
        model = get_model()
        model_id = _model_id_for_gateway(str(model.name))
        provider_client = _openai_provider_client(model)
        base_url = None
        api_key = None
        if provider_client is None:
            # Only providers without Inspect's OpenAI SDK client need env auth;
            # AnyEval accounting is attached to the provider client when present.
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.trustedrouter.com/v1")
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for JevBench gateway calls without an Inspect provider client")
        result = await _gateway_decide(
            state_value=state.metadata["state"],
            question=state.metadata["question"],
            labels=labels,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            provider_client=provider_client,
            reasoning=reasoning,
        )
        state.metadata["jevbench_result"] = {
            "probs": result.probs,
            "probs_source": result.probs_source,
            "usage": result.usage,
            "latency_s": result.latency_s,
            "served_model": result.served_model,
            "transport": result.transport,
            "reasoning": reasoning,
            "invalid_reason": result.invalid_reason,
        }
        state.completed = True
        return state

    return solve


@scorer(metrics=[accuracy(), mean_brier()])
def jevbench_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata.get("jevbench_result") or {}
        result_metadata = {
            key: result.get(key)
            for key in ("probs_source", "usage", "latency_s", "served_model", "transport", "reasoning")
            if key in result
        }
        return _score_decision(
            raw_probs=result.get("probs"),
            expected=target.text,
            labels=[str(label) for label in state.metadata["labels"]],
            question_type=state.metadata["question_type"],
            gold_distribution=state.metadata.get("gold_distribution"),
            result_metadata=result_metadata,
            preset_invalid_reason=result.get("invalid_reason"),
        )

    return score


@task
def jevbench(
    tier: str | None = None, timeout_s: float = 120, reasoning: bool | str | None = None
) -> Task:
    return Task(
        dataset=MemoryDataset(_load_samples(tier), name=_dataset_name(tier), location="jevbench.data"),
        solver=trustedrouter_decision_solver(timeout_s=timeout_s, reasoning=reasoning),
        scorer=jevbench_scorer(),
        name="jevbench",
    )


@task
def jevbench_easy(timeout_s: float = 120, reasoning: bool | str | None = None) -> Task:
    return jevbench(tier="easy", timeout_s=timeout_s, reasoning=reasoning)


@task
def jevbench_hard(timeout_s: float = 120, reasoning: bool | str | None = None) -> Task:
    return jevbench(tier="hard", timeout_s=timeout_s, reasoning=reasoning)


@task
def jevbench_original(timeout_s: float = 120, reasoning: bool | str | None = None) -> Task:
    return jevbench(tier="original", timeout_s=timeout_s, reasoning=reasoning)
