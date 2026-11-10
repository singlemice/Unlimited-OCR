"""Tests for the Unlimited-OCR 2 -> Markdown post-processing pipeline.

Covers utils/process_unlimitedOCR2.py (pure stdlib, runs identically on every
platform). These guard the conversion rules we rely on for evaluation.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

from process_unlimitedOCR2 import (  # noqa: E402
    convert,
    _parse_blocks,
    _build_groups,
    _render_blocks,
    _single_root_titles,
)


def test_convert_basic_title_and_text():
    raw = ("<PAGE><BLOCK>title<LEVEL1>[0,0,100,10]<FORK>Hello"
           "<BLOCK>text[0,20,100,40]<FORK>World body")
    md = convert(raw)
    assert "# Hello" in md
    assert "World body" in md


def test_convert_splits_pages_keeps_order():
    raw = "<PAGE><BLOCK>text[0,0,10,10]<FORK>A<PAGE><BLOCK>text[0,0,10,10]<FORK>B"
    md = convert(raw)
    # Two pages -> two text blocks, A before B.
    assert md.index("A") < md.index("B")


def test_convert_image_label_renders_markdown_image():
    raw = "<PAGE><BLOCK>image[10,20,30,40]<FORK>"
    md = convert(raw)
    assert "![](page1_10_20_30_40.jpg)" in md


def test_convert_coordinates_clamped_to_1000():
    raw = "<PAGE><BLOCK>image[1100,2000,3000,4000]<FORK>"
    md = convert(raw)
    assert "![](page1_1000_1000_1000_1000.jpg)" in md


def test_convert_cross_table_merge_drops_repeated_header():
    frag1 = '<PAGE><BLOCK>cross_table[0,0,10,10]<FORK><table><tr><td>h1</td></tr><tr><td>a</td></tr></table>'
    frag2 = '<PAGE><BLOCK>append_table[0,0,10,10]<FORK><table><tr><td>h1</td></tr><tr><td>b</td></tr></table>'
    md = convert(frag1 + frag2)
    # Repeated header 'h1' should appear only once.
    assert md.count("h1") == 1
    assert "a" in md and "b" in md


def test_convert_removes_supported_stray_special_tokens():
    # The stray-token cleaner strips ``<|...|>``-bounded markers (e.g. ``<|/det|>``).
    raw = "<PAGE><BLOCK>text[0,0,10,10]<FORK>clean<|/det|>more"
    md = convert(raw)
    assert "<|/det|>" not in md
    assert "clean" in md and "more" in md


def test_convert_keeps_plain_text_with_pipe():
    # A plain '|' in text is not a stray token and must be preserved.
    raw = "<PAGE><BLOCK>text[0,0,10,10]<FORK>a | b"
    md = convert(raw)
    assert "a | b" in md


def test_convert_markdown_code_fence_stripped():
    raw = "```markdown\n<PAGE><BLOCK>text[0,0,10,10]<FORK>hi\n```"
    md = convert(raw)
    assert "```" not in md
    assert "hi" in md


def test_single_root_titles_collapses_multiple_top_headings():
    items = ["# A", "# B", "# C"]
    out = _single_root_titles(items)
    levels = [len(s) - len(s.lstrip("#")) for s in out if s.lstrip().startswith("#")]
    # Only one top-level heading should remain.
    assert levels.count(1) == 1


def test_groups_union_find_groups_cross_chain():
    blocks = _parse_blocks(
        "<PAGE><BLOCK>cross_text[0,0,10,10]<FORK>a"
        "<BLOCK>append_text[0,0,10,10]<FORK>b"
    )
    groups = _build_groups(blocks)
    # cross_ + append_ of the same base type should be one group.
    assert len(groups) == 1
