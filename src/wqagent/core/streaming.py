"""Best-effort token streaming for the model calls a UI can show live.

Streaming must never cost correctness: when the model has no ``stream``, the
provider refuses, or the chunks cannot be folded back into one message, the
caller gets exactly what a plain ``invoke()`` would have returned.

The deltas are for humans, not for the agent: callers are expected to emit them
as *transient* events (see ``write_event``) and to drive the run from the
complete message this module returns.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

__all__ = ["merge_chunks", "streaming_invoke"]


def _chunk_text(chunk: Any) -> str:
    """The text of one streamed chunk (a string, or a list of content blocks)."""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") in ("text", "output_text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def merge_chunks(chunks: Sequence[Any]) -> Any:
    """Fold streamed chunks into one message, tool-call deltas included.

    LangChain's ``AIMessageChunk.__add__`` does the real work; a chunk type that
    refuses to be added (a fake model, a plain ``AIMessage``) ends the fold and
    the message built so far is returned.
    """
    total = chunks[0]
    for chunk in chunks[1:]:
        try:
            total = total + chunk
        except Exception:  # noqa: BLE001 - not addable: keep what we have
            break
    return total


def streaming_invoke(
    model: Any,
    messages: Sequence[Any],
    on_delta: Callable[[str], None],
) -> Any:
    """Invoke *model*, reporting text deltas; returns the complete message.

    Falls back to ``model.invoke(messages)`` whenever streaming is unavailable or
    fails; a partially streamed attempt is discarded so the caller never sees a
    truncated answer.
    """
    stream = getattr(model, "stream", None)
    if callable(stream):
        chunks: list[Any] = []
        try:
            for chunk in stream(messages):
                chunks.append(chunk)
                text = _chunk_text(chunk)
                if text:
                    try:
                        on_delta(text)
                    except Exception:  # noqa: BLE001 - the UI must not break the run
                        pass
        except Exception:  # noqa: BLE001 - fall back to one plain call
            chunks = []
        if chunks:
            merged = merge_chunks(chunks)
            if _chunk_text(merged).strip():
                return merged
    return model.invoke(messages)
