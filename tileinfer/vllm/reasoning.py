"""The ATEM reasoning parser — MuseGlimmer's channel protocol, for ``--reasoning-parser atem``.

The checkpoint's chat template (``chat_template.jinja``, ATEM protocol) ends the generation
prompt with a bare ``<|start|>assistant``; the *model* then chooses its recipient and emits

    `` to=self`` REASONING ``<|eom|>`` `` to=user`` ANSWER ``<|eot|>``

``<|start|>`` / ``<|message|>`` / ``<|eom|>`` / ``<|eot|>`` are special tokens that the serving
stack strips, so what a parser sees in the output *text* is the plain markers ``" to=self"`` and
``" to=user"`` (leading space included — they are rendered inline after the stripped
``<|start|>assistant``).  Everything between ``" to=self"`` and the switch marker is the
reasoning channel (``message.reasoning``); everything after it is the user-facing content.
Tool channels (`` to=<tool>``) are out of scope here: without a tool parser the tool-call text
stays in the content, same as with no parser at all.

Degenerate case seen in practice: the model may answer *inside* the ``to=self`` channel and
never switch.  Those outputs are reported entirely as ``reasoning`` — an honest rendering of a
protocol violation, and better than pretending the reasoning is user-facing content.

Streaming derives the channel state from ``previous_text`` on every call (the parser instance
is per-request, but deriving state from the text keeps a resumed request consistent).  A tail
that could still grow into ``" to=user"`` is held back so the switch is never split across a
chunk boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Optional, Tuple

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

__all__ = ["AtemReasoningParser"]

#: The channel markers as they appear in the *stripped* output text.
SELF = " to=self"
USER = " to=user"
_MARKERS = (SELF, USER)
_LONGEST_MARKER = max(len(marker) for marker in _MARKERS)


def _find_channel_start(text: str) -> Optional[Tuple[str, int]]:
    """``(marker, end_index)`` of the first channel marker in ``text``, or ``None``."""
    hits = [(text.find(marker) + len(marker), marker) for marker in _MARKERS]
    starts = [hit for hit in hits if hit[0] >= len(hit[1])]
    if not starts:
        return None
    end, marker = min(starts)  # earliest marker wins; ties are impossible (disjoint starts)
    return marker, end


def _is_marker_prefix(tail: str) -> bool:
    """Could ``tail`` still grow into one of the markers?"""
    return any(marker.startswith(tail) for marker in _MARKERS)


class AtemReasoningParser(ReasoningParser):
    """Split ATEM channel output into ``reasoning`` (to=self) and ``content`` (to=user)."""

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Approximate: decode the output tail and look for the `` to=user`` switch.

        Only consulted by the structured-output engines (which this deployment does not enable),
        so the decode-per-call cost is acceptable and the tail window keeps it bounded.
        """
        if not input_ids:
            return False
        tail_ids = list(input_ids[-16:])
        try:
            tail = self.model_tokenizer.decode(tail_ids, skip_special_tokens=True)
        except Exception:  # noqa: BLE001 - never break a serving step on a decode quirk
            return True
        return USER in tail

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """The token ids after the `` to=user`` marker (all ids when there is no switch).

        Approximation: token/text boundaries are re-aligned by decoding once and re-encoding
        the prefix, which can be off by a token around the marker.  Adequate for the structured
        outputs path this feeds; the serving path uses the text-based methods below.
        """
        if not input_ids:
            return []
        try:
            text = self.model_tokenizer.decode(input_ids, skip_special_tokens=True)
        except Exception:  # noqa: BLE001
            return list(input_ids)
        found = _find_channel_start(text)
        if found is None or found[0] != SELF:  # only the self channel may switch later
            return list(input_ids)
        switch = text.find(USER, found[1])
        if switch < 0:
            return list(input_ids)
        prefix_text = text[: switch + len(USER)]
        prefix_ids = self.model_tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_ids) >= len(input_ids):
            return []
        return list(input_ids[len(prefix_ids):])

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional[DeltaMessage]:
        if not delta_text:
            return None
        cur_len = len(current_text)
        prev_len = len(previous_text)

        # Where does the *current* text stand channel-wise?  (Derived from the text, not
        # carried in instance state, so a resumed request stays consistent.)  Emission is the
        # delta region ``(prev_len, cur_len]`` classified against the current boundaries.
        start = _find_channel_start(current_text)
        if start is None:
            # No marker yet.  While the whole text could still grow into a marker, hold the
            # chunk back; once it cannot, flow it through as plain content (identity
            # behaviour — a marker that arrives after content already flowed is treated the
            # same way, see the degenerate-note in the module docstring).
            if _is_marker_prefix(current_text):
                return None
            return DeltaMessage(reasoning=None, content=delta_text)

        marker, marker_end = start
        if marker == USER:
            # Direct to=user: everything after the marker is content, forever.
            from_idx = max(prev_len, marker_end)
            content_part = current_text[from_idx:cur_len] if cur_len > from_idx else ""
            return DeltaMessage(reasoning=None, content=content_part or None)

        # In the to=self channel: reasoning from the marker until the " to=user" switch.
        switch = current_text.find(USER, marker_end)
        if switch < 0:
            # No switch yet — hold back a tail that could still grow into the switch marker,
            # so " to=user" is never split across a chunk boundary.
            hold = 0
            for width in range(min(cur_len - marker_end, _LONGEST_MARKER - 1), 0, -1):
                if _is_marker_prefix(current_text[cur_len - width:]):
                    hold = width
                    break
            from_idx = max(prev_len, marker_end)
            emit_up_to = cur_len - hold
            reasoning_part = (
                current_text[from_idx:emit_up_to] if emit_up_to > from_idx else ""
            )
            return DeltaMessage(reasoning=reasoning_part or None, content=None)

        boundary = switch + len(USER)
        from_idx = max(prev_len, marker_end)
        reasoning_part = current_text[from_idx:switch] if switch > from_idx else ""
        content_from = max(prev_len, boundary)
        content_part = current_text[content_from:cur_len] if cur_len > content_from else ""
        return DeltaMessage(
            reasoning=reasoning_part or None,
            content=content_part or None,
        )

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming split: ``(" to=self" ... " to=user")`` -> reasoning/content."""
        del request
        if not model_output:
            return None, None
        start = _find_channel_start(model_output)
        if start is None:
            return None, model_output
        marker, marker_end = start
        if marker == USER:
            return None, model_output[marker_end:]
        switch = model_output.find(USER, marker_end)
        if switch < 0:
            # The model never left the reasoning channel (observed in practice): report the
            # whole text as reasoning rather than passing a channel header to the user.
            return model_output[marker_end:], None
        return (
            model_output[marker_end:switch],
            model_output[switch + len(USER):],
        )
