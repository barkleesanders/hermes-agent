"""Claude Code CLI adapter for Hermes provider fallback.

This transport intentionally goes through the first-party ``claude -p`` path
instead of reusing Claude Code OAuth tokens against Anthropic's public Messages
API.  That keeps subscription routing, account gating, and CLI-side behavior in
Claude Code's own process — the stable path users verify with ``claude -p``.

The adapter presents an OpenAI Chat Completions-shaped response so the existing
AIAgent loop can consume it without special casing the downstream result shape.
It is text-only and disables Claude Code built-in tools by default; Hermes tools
remain owned by Hermes, avoiding a second autonomous tool runner inside a model
fallback call.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Iterable


_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_MAX_BUDGET_USD = "0.50"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_claude_code_command(command: str | None = None) -> str:
    """Return the Claude Code CLI command or raise a clear error."""
    candidate = (
        str(command or "").strip()
        or os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CLI_PATH", "").strip()
        or "claude"
    )
    # ``shutil.which`` is correct for bare command names.  Preserve explicit
    # paths so subprocess can report executable/permission errors accurately.
    if os.path.sep not in candidate:
        resolved = shutil.which(candidate)
        if not resolved:
            raise RuntimeError(
                f"Claude Code CLI command '{candidate}' was not found. "
                "Install Claude Code or set HERMES_CLAUDE_CODE_COMMAND."
            )
        return resolved
    if not os.path.exists(candidate):
        raise RuntimeError(
            f"Claude Code CLI command path '{candidate}' does not exist. "
            "Set HERMES_CLAUDE_CODE_COMMAND to a valid claude binary."
        )
    return candidate


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI/Hermes message content into safe text for ``claude -p``."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "")
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item_type in {"image_url", "input_image", "image"}:
                    parts.append("[Image content omitted in Claude Code CLI text fallback]")
                else:
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def split_system_and_prompt(messages: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """Convert chat messages into ``--system-prompt`` plus a transcript prompt."""
    system_parts: list[str] = []
    prompt_parts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        text = _content_to_text(message.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
            continue

        label = {
            "assistant": "Assistant",
            "tool": "Tool result",
            "function": "Tool result",
            "developer": "Developer",
            "user": "User",
        }.get(role, role.title())
        if role == "assistant" and message.get("tool_calls"):
            try:
                calls = json.dumps(message.get("tool_calls"), ensure_ascii=False)
            except Exception:
                calls = str(message.get("tool_calls"))
            text = (text + "\n" if text else "") + f"[Assistant requested tool calls: {calls}]"
        if text:
            prompt_parts.append(f"{label}:\n{text}")

    prompt_parts.append("Assistant:")
    return "\n\n".join(system_parts).strip(), "\n\n".join(prompt_parts).strip()


def _usage_from_payload(payload: dict[str, Any]) -> SimpleNamespace | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=int(usage.get("total_tokens") or input_tokens + output_tokens),
    )


def _redact_cli_error(text: str) -> str:
    # Keep diagnostics useful while avoiding accidental credential/session leaks.
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", text or "")
    text = re.sub(r"[A-Za-z0-9_=-]{32,}", "[REDACTED]", text)
    return text.strip()


def create_claude_code_cli_completion(**kwargs: Any) -> SimpleNamespace:
    """Run ``claude -p`` and return an OpenAI Chat Completions-shaped object."""
    model = str(kwargs.get("model") or "").strip()
    if model.lower() in {"", "default", "auto", "claude-code-default"}:
        model = ""
    messages = kwargs.get("messages") or []
    system_prompt, prompt = split_system_and_prompt(messages)
    if not prompt:
        prompt = "Assistant:"

    command = resolve_claude_code_command(kwargs.get("__claude_code_command"))
    argv = [command, "-p", prompt, "--output-format", "json", "--no-session-persistence"]

    if model:
        argv.extend(["--model", model])
    if system_prompt:
        argv.extend(["--system-prompt", system_prompt])

    if not _truthy(os.getenv("HERMES_CLAUDE_CODE_CLI_ALLOW_TOOLS")):
        # Primary source: `claude -p --help` documents `--tools ''` as disabling
        # built-in Claude Code tools. Hermes, not the fallback subprocess, owns
        # tool execution in this path.
        argv.extend(["--tools", ""])

    max_budget = os.getenv("HERMES_CLAUDE_CODE_CLI_MAX_BUDGET_USD", _DEFAULT_MAX_BUDGET_USD).strip()
    if max_budget:
        argv.extend(["--max-budget-usd", max_budget])

    raw_extra_args = os.getenv("HERMES_CLAUDE_CODE_CLI_ARGS", "").strip()
    if raw_extra_args:
        argv.extend(shlex.split(raw_extra_args))
    argv.extend(list(kwargs.get("__claude_code_args") or []))

    timeout = kwargs.get("timeout") or os.getenv("HERMES_CLAUDE_CODE_CLI_TIMEOUT_SECONDS") or _DEFAULT_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(timeout)
    except (TypeError, ValueError):
        timeout_seconds = _DEFAULT_TIMEOUT_SECONDS

    proc = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        detail = _redact_cli_error(stderr or stdout)
        raise RuntimeError(
            f"Claude Code CLI failed with exit code {proc.returncode}"
            + (f": {detail}" if detail else "")
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Claude Code CLI returned non-JSON output: "
            f"{_redact_cli_error(stdout[:1000])}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Claude Code CLI returned invalid JSON payload")
    if payload.get("is_error"):
        detail = payload.get("result") or payload.get("api_error_status") or payload
        raise RuntimeError(f"Claude Code CLI returned an error: {_redact_cli_error(str(detail))}")

    content = str(payload.get("result") or "")
    usage = _usage_from_payload(payload)
    stop_reason = str(payload.get("stop_reason") or "end_turn")
    finish_reason = "length" if stop_reason in {"max_tokens", "model_context_window_exceeded"} else "stop"

    message = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        model=model or payload.get("model") or "claude-code-cli",
        usage=usage,
        provider_data={"claude_code_cli": True, "stop_reason": stop_reason},
    )


class ClaudeCodeCLICompletionsAdapter:
    def __init__(self, command: str | None = None, args: list[str] | None = None):
        self._command = command
        self._args = list(args or [])

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if self._command:
            kwargs["__claude_code_command"] = self._command
        if self._args:
            kwargs["__claude_code_args"] = self._args
        return create_claude_code_cli_completion(**kwargs)


class ClaudeCodeCLIClient:
    """OpenAI-client-compatible wrapper around ``claude -p``."""

    def __init__(self, *, command: str | None = None, args: list[str] | None = None):
        self.api_key = "claude-code-cli"
        self.base_url = "claude-code-cli://claude"
        self.command = command
        self.args = list(args or [])
        completions = ClaudeCodeCLICompletionsAdapter(command=command, args=args)
        self.chat = SimpleNamespace(completions=completions)

    def close(self) -> None:
        return None
