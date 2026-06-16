from __future__ import annotations

import json
from types import SimpleNamespace

from agent.claude_code_cli_adapter import (
    ClaudeCodeCLIClient,
    create_claude_code_cli_completion,
    normalize_claude_code_model,
    split_system_and_prompt,
)
from agent.transports import get_transport
from hermes_cli.models import get_default_model_for_provider, provider_model_ids


def test_claude_code_cli_catalog_defaults_to_bare_route():
    models = provider_model_ids("claude-code-cli")
    assert models[:5] == ["default", "sonnet", "opus", "haiku", "fable"]
    assert get_default_model_for_provider("claude-code-cli") == "default"


def test_normalize_claude_code_model_maps_hermes_ids_to_cli_selectors():
    assert normalize_claude_code_model("") == ""
    assert normalize_claude_code_model("default") == ""
    assert normalize_claude_code_model("claude-code-default") == ""
    assert normalize_claude_code_model("sonnet") == "sonnet"
    assert normalize_claude_code_model("opus") == "opus"
    assert normalize_claude_code_model("haiku") == "haiku"
    assert normalize_claude_code_model("fable") == "fable"
    assert normalize_claude_code_model("anthropic/claude-sonnet-4.6") == "sonnet"
    assert normalize_claude_code_model("anthropic/claude-opus-4-8") == "opus"
    assert normalize_claude_code_model("claude-haiku-4.5") == "haiku"
    assert normalize_claude_code_model("claude-fable-5") == "fable"


def test_split_system_and_prompt_flattens_transcript():
    system, prompt = split_system_and_prompt([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": "Hi"},
        {"role": "tool", "content": "tool output"},
    ])

    assert system == "Be terse."
    assert "User:\nHello" in prompt
    assert "Assistant:\nHi" in prompt
    assert "Tool result:\ntool output" in prompt
    assert prompt.endswith("Assistant:")


def test_create_completion_invokes_claude_print_json(monkeypatch, tmp_path):
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("#!/bin/sh\nexit 0\n")
    fake_claude.chmod(0o755)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "type": "result",
                "result": "CLI_OK",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }),
            stderr="",
        )

    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", str(fake_claude))
    monkeypatch.setenv("HERMES_CLAUDE_CODE_CLI_MAX_BUDGET_USD", "0.25")
    monkeypatch.delenv("HERMES_CLAUDE_CODE_CLI_ALLOW_TOOLS", raising=False)
    monkeypatch.setattr("agent.claude_code_cli_adapter.subprocess.run", fake_run)

    response = create_claude_code_cli_completion(
        model="default",
        messages=[
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Say OK"},
        ],
        timeout=12,
    )

    argv, kwargs = calls[0]
    assert argv[:3] == [str(fake_claude), "-p", "User:\nSay OK\n\nAssistant:"]
    assert "--output-format" in argv and "json" in argv
    assert "--no-session-persistence" in argv
    assert "--model" not in argv
    assert "--system-prompt" in argv and "System" in argv
    assert "--tools" in argv and "" in argv
    assert "--max-budget-usd" in argv and "0.25" in argv
    assert kwargs["timeout"] == 12.0
    assert response.choices[0].message.content == "CLI_OK"
    assert response.usage.total_tokens == 5


def test_create_completion_passes_explicit_model_to_claude(monkeypatch, tmp_path):
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("#!/bin/sh\nexit 0\n")
    fake_claude.chmod(0o755)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"result": "MODEL_OK"}),
            stderr="",
        )

    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", str(fake_claude))
    monkeypatch.setenv("HERMES_CLAUDE_CODE_CLI_MAX_BUDGET_USD", "0.25")
    monkeypatch.setattr("agent.claude_code_cli_adapter.subprocess.run", fake_run)

    response = create_claude_code_cli_completion(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "Say OK"}],
    )

    argv, _kwargs = calls[0]
    assert "--model" in argv
    model_index = argv.index("--model")
    assert argv[model_index + 1] == "sonnet"
    assert response.model == "sonnet"
    assert response.choices[0].message.content == "MODEL_OK"


def test_claude_code_cli_transport_reuses_chat_completion_normalizer():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="TRANSPORT_OK",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    transport = get_transport("claude_code_cli")
    assert transport is not None
    assert transport.validate_response(response) is True
    normalized = transport.normalize_response(response)
    assert normalized.content == "TRANSPORT_OK"
    assert normalized.finish_reason == "stop"
    assert normalized.usage.total_tokens == 3


def test_client_exposes_chat_completions_create(monkeypatch, tmp_path):
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("#!/bin/sh\nexit 0\n")
    fake_claude.chmod(0o755)

    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"result": "CLIENT_OK"}),
            stderr="",
        )

    monkeypatch.setattr("agent.claude_code_cli_adapter.subprocess.run", fake_run)
    client = ClaudeCodeCLIClient(command=str(fake_claude))
    response = client.chat.completions.create(model="sonnet", messages=[{"role": "user", "content": "x"}])
    assert response.choices[0].message.content == "CLIENT_OK"
