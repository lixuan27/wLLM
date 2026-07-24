"""Output objects matching the shape app code consumes.

Apps read ``out.request_output.prompt_token_ids`` and iterate
``out.request_output.outputs`` for completions carrying ``token_ids``,
``text``, and a ``multimodal_output`` mapping (layer tables, tts marker
embeddings, audio chunks, sample rates). These dataclasses pin that
contract so both this engine and external ones stay interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionOutput:
    index: int = 0
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    multimodal_output: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = False


@dataclass
class OmniOutput:
    request_id: str
    request_output: RequestOutput
    stage_id: int = 0

    @property
    def finished(self) -> bool:
        return self.request_output.finished
