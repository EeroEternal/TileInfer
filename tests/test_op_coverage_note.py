"""The operator-coverage ledger must not rot away from the family table.

``docs/notes/op-coverage-qwen36.md`` answers one question for every operator Qwen3.6 calls: is it
**bound** by us, a vLLM **fallback** on an out-of-tree platform, or **MISSING** (raises on the first
real step).  That is a claim about *two* things that live in different files - the ledger's rows and
``tileinfer/vllm/ops/families/gdn_moe.py::required_names()`` - so the claim is checkable, and this
file checks it:

1. the class column only ever says ``bound``/``fallback``/``MISSING`` (no invented fourth state);
2. the totals line equals the rows (a row added without the summary is a silent lie);
3. **every** name the family promises to bind is documented (a new binding forces a new row);
4. no name in the MISSING block is also in the family table - if someone binds the OOT MoE kernel
   the ratchet fails until the ledger is rewritten, which is exactly the review this file exists to
   force.  That has happened: the last MISSING operator (`unquantized_fused_moe`) is bound, and the
   two MISSING assertions below now enforce the *closed* state (zero MISSING rows, an empty block,
   and both directions of the block/rows agreement) instead of requiring one;
5. the MISSING block and the MISSING rows agree with each other.

The ledger's line numbers point at ``runtime/vllm-api/`` (gitignored), so nothing here reads the
sources: only the note's own structure is pinned.  Adding a row is expected; adding a *class* or a
MISSING name without the corresponding code change is the thing that fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tileinfer.vllm.ops.families import gdn_moe

NOTE = Path(__file__).resolve().parents[1] / "docs" / "notes" / "op-coverage-qwen36.md"

#: ``bound``, ``**bound**``, ``fallback (not taken)``, ``MISSING`` - the class cell's first word.
_CLASS = re.compile(r"^(bound|fallback|missing)\b", re.IGNORECASE)

#: The summary line the note ends its tables with.
_TOTALS = re.compile(
    r"Totals:\s*(\d+)\s*bound\s*/\s*(\d+)\s*fallback\s*/\s*(\d+)\s*MISSING,\s*over\s*(\d+)\s*rows",
    re.IGNORECASE,
)

#: The canonical MISSING list: ``* `Name` (why)`` between the BEGIN/END markers.
_BEGIN = "<!-- BEGIN MISSING -->"
_END = "<!-- END MISSING -->"

#: The first back-ticked token of a table cell - the operator's name as the sources spell it.
_TICKED = re.compile(r"`([^`]+)`")


def _text() -> str:
    assert NOTE.is_file(), f"the ledger is gone: {NOTE}"
    return NOTE.read_text(encoding="utf-8")


def _rows() -> list[tuple[str, str, str, str]]:
    """``[(operator cell, class, call site, evidence)]`` for every row of every op table."""
    rows: list[tuple[str, str, str, str]] = []
    for line in _text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:  # not one of the five-column op rows (header separator, prose)
            continue
        match = _CLASS.match(cells[2].replace("*", ""))
        if match is None:  # the header row itself ("class")
            continue
        rows.append((cells[0], match.group(1).lower(), cells[1], cells[3]))
    return rows


def _missing_block() -> list[str]:
    text = _text()
    start, end = text.index(_BEGIN), text.index(_END)
    names = []
    for line in text[start:end].splitlines():
        if not line.lstrip().startswith("*"):
            continue
        name = _TICKED.search(line)
        if name is not None:
            names.append(name.group(1).strip())
    return names


def test_the_ledger_has_rows_and_only_three_classes():
    rows = _rows()
    assert len(rows) >= 30, f"only {len(rows)} op rows parsed; the table format changed"
    assert {row[1] for row in rows} <= {"bound", "fallback", "missing"}
    # Every row has a call site and an evidence cell: a row without evidence is a guess.
    for operator, cls, call_site, evidence in rows:
        assert call_site, f"{operator}: no call site"
        assert evidence and evidence != "-", f"{operator}: no evidence"


def test_the_totals_line_matches_the_rows():
    counts = {"bound": 0, "fallback": 0, "missing": 0}
    for _operator, cls, _call_site, _evidence in _rows():
        counts[cls] += 1

    match = _TOTALS.search(_text())
    assert match is not None, "the note lost its `Totals: N bound / N fallback / N MISSING` line"
    bound, fallback, missing, total = (int(group) for group in match.groups())
    assert (bound, fallback, missing) == (
        counts["bound"],
        counts["fallback"],
        counts["missing"],
    ), f"the summary says {bound}/{fallback}/{missing} but the rows count {counts}"
    assert total == sum(counts.values()), f"{total} rows claimed, {sum(counts.values())} present"


def test_every_op_the_family_promises_is_documented():
    """``required_names()`` is the promise; the ledger is where each name is justified."""
    documented = " ".join(row[0] for row in _rows())
    missing_from_note = [name for name in gdn_moe.required_names() if name not in documented]
    assert not missing_from_note, (
        "these names are in gdn_moe.required_names() but no ledger row mentions them: "
        f"{missing_from_note}. Add a row (and its evidence) before promising the binding."
    )


def test_the_missing_set_does_not_collide_with_the_family_table():
    """A MISSING row that the family also binds is stale in one direction or the other.

    The gap this note opened with (the OOT routed-expert kernel) is closed, so the block is allowed
    to be empty - but the *collision* rule still bites the moment someone re-adds a MISSING row for
    a name the family binds.
    """
    missing_rows = [row for row in _rows() if row[1] == "missing"]
    for operator, _cls, _call_site, _evidence in missing_rows:
        name = _TICKED.search(operator)
        assert name is not None, f"MISSING row {operator!r} does not name the operator"
        assert name.group(1) not in gdn_moe.required_names(), (
            f"{name.group(1)} is now in required_names() - it is bound, so the ledger's MISSING "
            "row (and the section around it) must be rewritten, not left standing"
        )


def test_the_missing_block_lists_exactly_the_missing_rows():
    """Empty block <=> zero MISSING rows; non-empty block <=> the same names on both sides."""
    block = _missing_block()

    row_names = []
    for operator, cls, _call_site, _evidence in _rows():
        if cls != "missing":
            continue
        name = _TICKED.search(operator)
        assert name is not None
        row_names.append(name.group(1))

    assert len(block) == len(set(block)), f"duplicate entries in the MISSING block: {block}"
    if not row_names:
        assert not block, (
            "the MISSING block still lists "
            f"{block} but no row is MISSING any more: empty the block (the operator is bound now)"
        )
        return
    assert block, "there are MISSING rows but the block lists nothing"
    for name in block:
        assert any(name in row for row in row_names), (
            f"{name!r} is in the MISSING block but no MISSING row offers its call site/evidence"
        )
    for name in row_names:
        assert any(name in entry for entry in block), (
            f"{name!r} is a MISSING row but the MISSING block does not list it"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
