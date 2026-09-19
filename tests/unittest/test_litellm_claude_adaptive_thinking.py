import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import openai
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler

# Environment variables that LiteLLMAIHandler.__init__ reads or mutates: the AWS
# credential path (entered when AWS_USE_IMDS is set) writes the AWS_* variables,
# and OPENAI_API_KEY influences the litellm.api_key fallback.
_HANDLER_ENV_VARS = (
    "AWS_USE_IMDS",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION_NAME",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _restore_litellm_globals():
    """LiteLLMAIHandler.__init__ mutates global litellm/openai state and, when
    AWS_USE_IMDS is set, os.environ; snapshot and restore both, and drop
    AWS_USE_IMDS so the AWS credential path never runs in these tests."""
    saved = (litellm.api_key, getattr(litellm, "openai_key", None), openai.api_key)
    saved_env = {name: os.environ.get(name) for name in _HANDLER_ENV_VARS}
    os.environ.pop("AWS_USE_IMDS", None)
    try:
        yield
    finally:
        litellm.api_key = saved[0]
        litellm.openai_key = saved[1]
        openai.api_key = saved[2]
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _settings(
    reasoning_effort="medium",
    enabled=False,
    extended_enabled=False,
    extended_budget_tokens=2048,
    extended_max_output_tokens=4096,
    adaptive_override=None,
):
    flags = {
        "claude_adaptive_thinking_models_override": adaptive_override or [],
        "enable_claude_adaptive_thinking": enabled,
        "enable_claude_extended_thinking": extended_enabled,
        "extended_thinking_budget_tokens": extended_budget_tokens,
        "extended_thinking_max_output_tokens": extended_max_output_tokens,
    }
    config = SimpleNamespace(
        reasoning_effort=reasoning_effort,
        ai_timeout=120,
        custom_reasoning_model=False,
        max_model_tokens=32000,
        verbosity_level=0,
        get=lambda key, default=None: flags.get(key, default),
    )
    # Bedrock requests now resolve credentials per request rather than from the
    # process environment, so a bedrock model needs them supplied here.
    aws = {
        "aws.AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "aws.AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws.AWS_REGION_NAME": "us-east-1",
    }
    return SimpleNamespace(
        config=config,
        litellm=SimpleNamespace(get=lambda key, default=None: default),
        get=lambda key, default=None: aws.get(key, default),
    )


def _response():
    response = MagicMock()
    payload = {"choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]}
    response.__getitem__.side_effect = payload.__getitem__
    response.dict.return_value = payload
    return response


async def _run_completion(monkeypatch, model, reasoning_effort="medium", enabled=False,
                          extended_enabled=False, extended_override=None, adaptive_override=None):
    # An ambient selector would be refused before the request credentials are read.
    for variable in ("AWS_PROFILE_NAME", "AWS_ROLE_NAME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: _settings(reasoning_effort, enabled, extended_enabled, adaptive_override=adaptive_override),
    )
    with patch(
        "pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
        new_callable=AsyncMock,
    ) as completion:
        completion.return_value = _response()
        handler = LiteLLMAIHandler()
        if extended_override is not None:
            handler.claude_extended_thinking_models = extended_override
        await handler.chat_completion(model=model, system="sys", user="usr")
        return completion.call_args.kwargs


@pytest.mark.parametrize(
    "model, expected",
    [
        ("anthropic/claude-opus-4-8", True),
        ("bedrock/us.anthropic.claude-opus-4-7-v1:0", True),
        ("vertex_ai/claude-sonnet-5", True),
        ("anthropic/claude-opus-5", True),
        ("bedrock/us.anthropic.claude-opus-5", True),
        ("anthropic/claude-fable-5", True),
        ("anthropic/claude-fable-5-1", True),
        ("bedrock/us.anthropic.claude-fable-5-1", True),
        ("anthropic/claude-opus-4-6", False),
        ("anthropic/claude-sonnet-50", False),
        ("anthropic/claude-opus-50", False),
        ("anthropic/my-opus-4-8", False),
    ],
)
def test_detects_adaptive_thinking_models_across_providers(model, expected):
    assert LiteLLMAIHandler._is_claude_adaptive_thinking_model(model) is expected


@pytest.mark.asyncio
async def test_enabled_adaptive_thinking_sends_anthropic_payload(monkeypatch):
    kwargs = await _run_completion(
        monkeypatch,
        "anthropic/claude-opus-4-8",
        reasoning_effort="high",
        enabled=True,
    )

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_adaptive_thinking_effort_is_isolated_from_later_settings(monkeypatch):
    active_settings = _settings(reasoning_effort="low", enabled=True)
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: active_settings)
    handler = LiteLLMAIHandler()
    active_settings = _settings(reasoning_effort="high", enabled=True)

    with patch(
        "pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
        new_callable=AsyncMock,
    ) as completion:
        completion.return_value = _response()
        await handler.chat_completion(model="anthropic/claude-opus-4-8", system="sys", user="usr")

    assert completion.call_args.kwargs["output_config"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_adaptive_thinking_enablement_is_isolated_from_later_settings(monkeypatch):
    active_settings = _settings(reasoning_effort="low", enabled=False)
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: active_settings)
    handler = LiteLLMAIHandler()
    active_settings = _settings(reasoning_effort="high", enabled=True)

    with patch(
        "pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
        new_callable=AsyncMock,
    ) as completion:
        completion.return_value = _response()
        await handler.chat_completion(model="anthropic/claude-opus-4-8", system="sys", user="usr")

    assert "thinking" not in completion.call_args.kwargs
    assert "output_config" not in completion.call_args.kwargs


@pytest.mark.asyncio
async def test_extended_thinking_limits_are_isolated_from_later_settings(monkeypatch):
    model = "anthropic/claude-opus-4-6"
    active_settings = _settings(
        extended_enabled=True,
        extended_budget_tokens=1024,
        extended_max_output_tokens=2048,
    )
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: active_settings)
    handler = LiteLLMAIHandler()
    handler.claude_extended_thinking_models = [model]
    active_settings = _settings(
        extended_enabled=True,
        extended_budget_tokens=4096,
        extended_max_output_tokens=8192,
    )

    with patch(
        "pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
        new_callable=AsyncMock,
    ) as completion:
        completion.return_value = _response()
        await handler.chat_completion(model=model, system="sys", user="usr")

    assert completion.call_args.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert completion.call_args.kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_adaptive_thinking_accepts_max_effort_without_enum_dependency(monkeypatch):
    kwargs = await _run_completion(
        monkeypatch,
        "anthropic/claude-sonnet-5",
        reasoning_effort="max",
        enabled=True,
    )

    assert kwargs["output_config"] == {"effort": "max"}


@pytest.mark.asyncio
async def test_adaptive_thinking_omits_unsupported_effort(monkeypatch):
    kwargs = await _run_completion(
        monkeypatch,
        "anthropic/claude-opus-4-8",
        reasoning_effort="minimal",
        enabled=True,
    )

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "output_config" not in kwargs


@pytest.mark.asyncio
async def test_adaptive_thinking_is_opt_in_and_does_not_touch_older_models(monkeypatch):
    disabled = await _run_completion(
        monkeypatch,
        "anthropic/claude-opus-4-8",
        reasoning_effort="high",
        enabled=False,
    )
    older_model = await _run_completion(
        monkeypatch,
        "anthropic/claude-opus-4-6",
        reasoning_effort="high",
        enabled=True,
    )

    assert "thinking" not in disabled
    assert "output_config" not in disabled
    assert "thinking" not in older_model
    assert "output_config" not in older_model


@pytest.mark.asyncio
async def test_adaptive_only_model_in_extended_override_does_not_get_budget_tokens(monkeypatch):
    """An adaptive-only model wrongly placed in claude_extended_thinking_models_override must not
    receive the legacy budget_tokens payload, even with adaptive thinking left disabled.

    Reported by @IsmaelMartinez on #2531: the TOML comment warned about this configuration but the
    code did not enforce it, so the request was still shaped in a way the provider rejects (400).
    """
    kwargs = await _run_completion(
        monkeypatch,
        "anthropic/claude-opus-5",
        reasoning_effort="high",
        enabled=False,
        extended_enabled=True,
        extended_override=["anthropic/claude-opus-5"],
    )

    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


@pytest.mark.asyncio
async def test_non_adaptive_model_in_extended_override_still_gets_extended_thinking(monkeypatch):
    """The adaptive-only guard must not disturb ordinary extended-thinking models."""
    kwargs = await _run_completion(
        monkeypatch,
        "anthropic/claude-opus-4-6",
        reasoning_effort="high",
        enabled=False,
        extended_enabled=True,
        extended_override=["anthropic/claude-opus-4-6"],
    )

    assert kwargs["thinking"]["type"] == "enabled"
    assert "budget_tokens" in kwargs["thinking"]


@pytest.mark.asyncio
async def test_opaque_arn_with_adaptive_enabled_warns_and_skips_payload(monkeypatch):
    """An opaque application inference profile ARN carries no model name, so the adaptive
    payload must be skipped and a warning logged instead of failing silently (see #3216)."""
    logger = MagicMock()
    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.get_logger",
               return_value=logger):
        kwargs = await _run_completion(
            monkeypatch,
            "bedrock/converse/arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/abc123def456",
            enabled=True,
        )

    assert "thinking" not in kwargs
    assert "output_config" not in kwargs
    logger.warning.assert_called_once()
    assert "abc123def456" in logger.warning.call_args.args[0]
    assert "litellm.model_id" in logger.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_family_embedding_arn_still_gets_adaptive_payload(monkeypatch):
    """An ARN that embeds the model family normalises to a matching id, so adaptive thinking
    applies; the miss is specific to opaque suffixes (see #3216)."""
    kwargs = await _run_completion(
        monkeypatch,
        "bedrock/converse/arn:aws:bedrock:eu-central-1:000000000000:inference-profile/us.anthropic.claude-sonnet-5",
        enabled=True,
    )

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_extended_enabled_with_no_matching_model_warns(monkeypatch):
    """Extended thinking enabled with an empty override list leaves no matching model, so the
    warning fires even though the adaptive model gate is not involved (see #3216)."""
    logger = MagicMock()
    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.get_logger",
               return_value=logger):
        kwargs = await _run_completion(
            monkeypatch,
            "anthropic/claude-opus-4-6",
            enabled=False,
            extended_enabled=True,
            extended_override=[],
        )

    assert "thinking" not in kwargs
    logger.warning.assert_called_once()
    assert "opus-4-6" in logger.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_non_arn_model_warns_without_bedrock_advice(monkeypatch):
    """The Bedrock remedy must stay gated to Bedrock ids: a plain provider id that reaches the
    warning branch only gets the generic message, never the litellm.model_id ARN advice."""
    logger = MagicMock()
    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.get_logger",
               return_value=logger):
        kwargs = await _run_completion(
            monkeypatch,
            "openai/gpt-4o",
            enabled=True,
        )

    assert "thinking" not in kwargs
    logger.warning.assert_called_once()
    message = logger.warning.call_args.args[0]
    assert "gpt-4o" in message
    assert "litellm.model_id" not in message
    assert "arn" not in message


_OPAQUE_ARN = "arn:aws:bedrock:eu-central-1:123456789012:application-inference-profile/abc123def456"
_OPAQUE_MODEL = f"bedrock/converse/{_OPAQUE_ARN}"


@pytest.fixture
def _forget_registered_models():
    """The override registers ids in the process-global litellm model map; drop them again."""
    known = set(litellm.model_cost)
    try:
        yield
    finally:
        for name in set(litellm.model_cost) - known:
            litellm.model_cost.pop(name, None)


def _litellm_mapped_thinking(model):
    """Return the thinking payload LiteLLM's Bedrock Converse mapping keeps for an adaptive request."""
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    provider_model, provider, _, _ = litellm.get_llm_provider(model=model)
    assert provider == "bedrock"
    return AmazonConverseConfig().map_openai_params(
        non_default_params={"thinking": {"type": "adaptive"}},
        optional_params={},
        model=provider_model,
        drop_params=False,
    ).get("thinking")


@pytest.mark.asyncio
async def test_opaque_arn_is_not_adaptive_without_the_override(monkeypatch):
    kwargs = await _run_completion(monkeypatch, _OPAQUE_MODEL, enabled=True)

    assert "thinking" not in kwargs


@pytest.mark.asyncio
async def test_override_declares_an_opaque_arn_adaptive(monkeypatch, _forget_registered_models):
    kwargs = await _run_completion(
        monkeypatch,
        _OPAQUE_MODEL,
        reasoning_effort="high",
        enabled=True,
        adaptive_override=[f"  {_OPAQUE_MODEL}  "],
    )

    assert kwargs["model"] == _OPAQUE_MODEL
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_override_needs_adaptive_thinking_enabled(monkeypatch, _forget_registered_models):
    kwargs = await _run_completion(monkeypatch, _OPAQUE_MODEL, enabled=False, adaptive_override=[_OPAQUE_MODEL])

    assert "thinking" not in kwargs


@pytest.mark.asyncio
async def test_override_is_additive_and_leaves_other_models_alone(monkeypatch, _forget_registered_models):
    named = await _run_completion(
        monkeypatch, "bedrock/us.anthropic.claude-sonnet-5", enabled=True, adaptive_override=[_OPAQUE_MODEL]
    )
    unlisted = await _run_completion(
        monkeypatch, "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", enabled=True,
        adaptive_override=[_OPAQUE_MODEL],
    )

    assert named["thinking"] == {"type": "adaptive"}
    assert "thinking" not in unlisted


@pytest.mark.parametrize("bad_override", ["not-a-list", [1], [""], [_OPAQUE_MODEL, None]])
def test_malformed_override_falls_back_to_built_in_detection(monkeypatch, bad_override):
    monkeypatch.setattr(
        litellm_handler, "get_settings", lambda: _settings(enabled=True, adaptive_override=bad_override)
    )
    logger = MagicMock()
    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.get_logger", return_value=logger):
        handler = LiteLLMAIHandler()

    assert handler.claude_adaptive_thinking_models_override == []
    assert any(
        "claude_adaptive_thinking_models_override" in call.args[0] for call in logger.warning.call_args_list
    )
    assert handler._model_uses_adaptive_thinking(_OPAQUE_MODEL) is False
    assert handler._model_uses_adaptive_thinking("anthropic/claude-opus-5") is True


@pytest.mark.parametrize("model", [_OPAQUE_MODEL, f"bedrock/{_OPAQUE_ARN}"])
def test_litellm_keeps_the_adaptive_payload_for_an_overridden_arn(monkeypatch, _forget_registered_models, model):
    """LiteLLM rewrites the adaptive payload to budget_tokens for Bedrock ids its model map does
    not know, which Bedrock rejects; the override must make LiteLLM's own check agree."""
    unlisted = model.replace("abc123def456", "unlisted000000")
    assert _litellm_mapped_thinking(model) != {"type": "adaptive"}

    monkeypatch.setattr(
        litellm_handler, "get_settings", lambda: _settings(enabled=True, adaptive_override=[model])
    )
    LiteLLMAIHandler()

    assert _litellm_mapped_thinking(model) == {"type": "adaptive"}
    assert _litellm_mapped_thinking(unlisted) != {"type": "adaptive"}


def test_override_registers_only_bedrock_ids_with_litellm(monkeypatch, _forget_registered_models):
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: _settings(enabled=True, adaptive_override=[_OPAQUE_MODEL, "anthropic/my-private-claude"]),
    )
    LiteLLMAIHandler()

    assert litellm.model_cost[_OPAQUE_MODEL]["supports_adaptive_thinking"] is True
    assert "anthropic/my-private-claude" not in litellm.model_cost
