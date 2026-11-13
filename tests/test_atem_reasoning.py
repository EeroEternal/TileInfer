"""The ATEM reasoning parser: channel splitting, boundary-by-boundary.

The parser strips MuseGlimmer's ATEM channel markers (``" to=self"`` reasoning,
``" to=user"`` content) from chat output.  What is pinned here is the part that breaks
quietly in production: the *streaming* split must reassemble — character chunk by character
chunk — to exactly the non-streaming split, including the switch marker arriving split across
chunk boundaries (`` to=us|er``), the degenerate no-switch output, and unmarked output
flowing through untouched.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm.reasoning")

from vllm.reasoning import ReasoningParserManager  # noqa: E402

from tileinfer.vllm.reasoning import AtemReasoningParser, SELF, USER  # noqa: E402


class _FakeTokenizer:
    """Just enough tokenizer for the parser: decode() over a fixed vocabulary."""

    VOCAB = {
        0: "",
        1: " to=self",
        2: " to=user",
        3: "思考",
        4: "答案",
        5: "普通",
        6: "输出",
    }

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(self.VOCAB.get(int(i), "") for i in ids)

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        out = []
        for token_id, token_text in self.VOCAB.items():
            while token_text and token_text in text:
                text = text.replace(token_text, "", 1)
                out.append(token_id)
        return out


def _parser() -> AtemReasoningParser:
    return AtemReasoningParser(_FakeTokenizer())


# ---------------------------------------------------------------------------------------
# non-streaming
# ---------------------------------------------------------------------------------------


def test_split_reasoning_then_content():
    reasoning, content = _parser().extract_reasoning(" to=self想一想 to=user说出来", None)
    assert reasoning == "想一想"
    assert content == "说出来"


def test_direct_user_channel_has_no_reasoning():
    reasoning, content = _parser().extract_reasoning(" to=user直接回答", None)
    assert reasoning is None
    assert content == "直接回答"


def test_degenerate_self_channel_without_a_switch_is_all_reasoning():
    reasoning, content = _parser().extract_reasoning(" to=self只有推理没有切换", None)
    assert reasoning == "只有推理没有切换"
    assert content is None


def test_unmarked_output_flows_through_as_content():
    reasoning, content = _parser().extract_reasoning("普通输出", None)
    assert reasoning is None
    assert content == "普通输出"


def test_empty_output_is_a_pair_of_nones():
    assert _parser().extract_reasoning("", None) == (None, None)


def test_registration_under_the_name_atem():
    # The plugin's install() registers this; do it here too so the contract holds standalone.
    ReasoningParserManager.register_module("atem", module=AtemReasoningParser)
    assert ReasoningParserManager.reasoning_parsers.get("atem") is AtemReasoningParser


# ---------------------------------------------------------------------------------------
# streaming: reassembles to the non-streaming split, boundary by boundary
# ---------------------------------------------------------------------------------------


def _stream(text: str, width: int = 1):
    """Feed ``text`` through the streaming parser ``width`` chars at a time; return the join."""
    parser = _parser()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    previous = ""
    for i in range(0, len(text), width):
        chunk = text[i : i + width]
        current = previous + chunk
        delta = parser.extract_reasoning_streaming(previous, current, chunk, [], [], [])
        if delta is not None:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
        previous = current
    return "".join(reasoning_parts), "".join(content_parts)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_streaming_reassembles_the_full_split(width):
    text = " to=self想一想过程 to=user说出来结果"
    got = _stream(text, width)
    assert got == ("想一想过程", "说出来结果")


def test_streaming_switch_marker_split_across_chunks():
    """The switch marker's own characters must never leak into either channel."""
    text = " to=self推理中 to=user回答"
    # ' to=us' | 'er' boundary sits inside one of the chunks at these widths.
    got = _stream(text, 7)
    assert got == ("推理中", "回答")


def test_streaming_direct_user_channel():
    got = _stream(" to=user直接回答", 3)
    assert got == ("", "直接回答")


def test_streaming_degenerate_self_channel_keeps_every_char():
    text = " to=self只有推理"
    got = _stream(text, 2)
    reasoning, _ = _parser().extract_reasoning(text, None)
    # The held-back tail is a prefix of the (absent) switch marker only when the text ends
    # mid-prefix; this text does not, so streaming must match non-streaming exactly.
    assert got == (reasoning, "")


def test_streaming_unmarked_output_flows_through():
    got = _stream("普通输出文本", 4)
    assert got == ("", "普通输出文本")


# ---------------------------------------------------------------------------------------
# token-id helpers (structured-output path)
# ---------------------------------------------------------------------------------------


def test_is_reasoning_end_checks_the_decoded_tail():
    parser = _parser()
    assert parser.is_reasoning_end([1, 3, 2]) is True  # ... to=user in the tail
    assert parser.is_reasoning_end([1, 3]) is False  # reasoning only, no switch
    assert parser.is_reasoning_end([]) is False


def test_extract_content_ids_splits_at_the_switch():
    parser = _parser()
    # 1 = " to=self", 3 = "思考", 2 = " to=user", 4 = "答案"
    assert parser.extract_content_ids([1, 3, 2, 4]) == [4]
    assert parser.extract_content_ids([1, 3]) == [1, 3]  # no switch: everything is content


def test_markers_spell_the_protocol_exactly():
    assert SELF == " to=self"
    assert USER == " to=user"
