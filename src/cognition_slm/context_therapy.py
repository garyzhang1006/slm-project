"""Inspectable context-care diagnostics for long LLM conversations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .tokenizer import ByteTokenizer


ALLOWED_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
MAX_MESSAGES = 512
MAX_MESSAGE_CHARS = 100_000
MAX_TOTAL_CHARS = 2_000_000
_SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


@dataclass(frozen=True)
class ContextMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ContextObservation:
    code: str
    severity: str
    message: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class RepairAction:
    code: str
    action: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "action": self.action, "reason": self.reason}


@dataclass(frozen=True)
class ContextAssessment:
    state: str
    message_count: int
    estimated_tokens: int
    token_budget: int | None
    pressure: float | None
    observations: tuple[ContextObservation, ...]
    actions: tuple[RepairAction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "message_count": self.message_count,
            "estimated_tokens": self.estimated_tokens,
            "token_budget": self.token_budget,
            "pressure": self.pressure,
            "observations": [item.to_dict() for item in self.observations],
            "actions": [item.to_dict() for item in self.actions],
        }


def _safe_excerpt(text: str, limit: int = 120) -> str:
    excerpt = re.sub(r"\s+", " ", text).strip()[:limit]
    for pattern in _SECRET_PATTERNS:
        excerpt = pattern.sub("[REDACTED]", excerpt)
    return excerpt


def parse_messages(raw_messages: Iterable[ContextMessage | Mapping[str, Any]]) -> tuple[ContextMessage, ...]:
    messages: list[ContextMessage] = []
    total_chars = 0
    for index, raw in enumerate(raw_messages):
        if isinstance(raw, ContextMessage):
            if not isinstance(raw.role, str) or not isinstance(raw.content, str):
                raise ValueError(f"message {index}: role and content must be text")
            message = ContextMessage(role=raw.role.strip().lower(), content=raw.content)
        elif isinstance(raw, Mapping):
            role = raw.get("role")
            content = raw.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(f"message {index}: role and content must be text")
            message = ContextMessage(role=role.strip().lower(), content=content)
        else:
            raise ValueError(f"message {index}: expected object")
        if message.role not in ALLOWED_ROLES:
            raise ValueError(f"message {index}: unsupported role {message.role!r}")
        if not message.content.strip():
            raise ValueError(f"message {index}: content must be non-empty text")
        if len(message.content) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message {index}: content exceeds {MAX_MESSAGE_CHARS} characters")
        total_chars += len(message.content)
        if total_chars > MAX_TOTAL_CHARS:
            raise ValueError(f"context exceeds {MAX_TOTAL_CHARS} characters")
        messages.append(message)
        if len(messages) > MAX_MESSAGES:
            raise ValueError(f"context cannot contain more than {MAX_MESSAGES} messages")
    if not messages:
        raise ValueError("context must contain at least one message")
    return tuple(messages)


def estimate_tokens(messages: Iterable[ContextMessage], tokenizer: ByteTokenizer | None = None) -> int:
    tokenizer = tokenizer or ByteTokenizer()
    return sum(
        len(tokenizer.encode(f"<{message.role}>\n{message.content}\n", add_bos=False, add_eos=False))
        for message in messages
    )


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _pressure_observation(estimated: int, budget: int) -> ContextObservation | None:
    pressure = estimated / budget
    if pressure >= 1.0:
        return ContextObservation(
            "context_over_budget",
            "critical",
            "Visible context exceeds supplied token budget.",
            (f"estimated_tokens={estimated}, token_budget={budget}",),
        )
    if pressure >= 0.85:
        return ContextObservation(
            "context_near_budget",
            "warning",
            "Visible context is close to supplied token budget.",
            (f"estimated_tokens={estimated}, token_budget={budget}",),
        )
    if pressure >= 0.65:
        return ContextObservation(
            "context_pressure",
            "info",
            "Visible context is using a large share of supplied token budget.",
            (f"estimated_tokens={estimated}, token_budget={budget}",),
        )
    return None


def _repetition_observation(messages: tuple[ContextMessage, ...]) -> ContextObservation | None:
    normalized = [_normalized(message.content) for message in messages]
    counts = Counter(item for item in normalized if len(item) >= 20)
    repeated = [item for item, count in counts.items() if count > 1]
    if not repeated:
        return None
    return ContextObservation(
        "repeated_context",
        "warning",
        "Multiple visible turns repeat the same substantial text.",
        tuple(_safe_excerpt(item) for item in repeated[:3]),
    )


def _actions_for(observations: tuple[ContextObservation, ...]) -> tuple[RepairAction, ...]:
    codes = {item.code for item in observations}
    actions: list[RepairAction] = []
    if "context_over_budget" in codes or "context_near_budget" in codes or "context_pressure" in codes:
        actions.append(
            RepairAction(
                "compress_context",
                "Compress old and repeated turns while preserving current goals, hard constraints, decisions, open questions, and verified evidence.",
                "High token pressure increases the chance that important requirements are lost.",
            )
        )
    if "repeated_context" in codes:
        actions.append(
            RepairAction(
                "deduplicate_context",
                "Keep one canonical copy of repeated text and record where it came from.",
                "Repeated turns consume budget without adding new evidence.",
            )
        )
    if not actions:
        actions.append(
            RepairAction(
                "continue",
                "Continue with visible context and keep claims tied to evidence.",
                "No pressure or repetition threshold was crossed.",
            )
        )
    return tuple(actions)


class ContextTherapist:
    """Diagnose visible context strain without inferring private model states."""

    def __init__(self, tokenizer: ByteTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or ByteTokenizer()

    def assess(
        self,
        messages: Iterable[ContextMessage | Mapping[str, Any]],
        *,
        token_budget: int | None = None,
    ) -> ContextAssessment:
        parsed = parse_messages(messages)
        if token_budget is not None and token_budget < 1:
            raise ValueError("token_budget must be positive")
        estimated = estimate_tokens(parsed, self.tokenizer)
        observations: list[ContextObservation] = []
        if token_budget is not None:
            pressure_observation = _pressure_observation(estimated, token_budget)
            if pressure_observation is not None:
                observations.append(pressure_observation)
        repetition_observation = _repetition_observation(parsed)
        if repetition_observation is not None:
            observations.append(repetition_observation)
        if any(item.code == "context_over_budget" for item in observations):
            state = "overloaded"
        elif observations:
            state = "strained"
        else:
            state = "stable"
        pressure = estimated / token_budget if token_budget is not None else None
        frozen_observations = tuple(observations)
        return ContextAssessment(
            state=state,
            message_count=len(parsed),
            estimated_tokens=estimated,
            token_budget=token_budget,
            pressure=pressure,
            observations=frozen_observations,
            actions=_actions_for(frozen_observations),
        )
