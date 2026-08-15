"""Regression tests for inline-cell editing permissions.

History: the KV-cache column (and other model-level inference knobs like
ctx/gpu/threads) must REMAIN editable even while the read-only "default"
preset is locked. Only the *sampling* parameters (temp/topk/topp/rp) are
governed by the preset and blocked on the locked default. A previous bug
locked every parameter column (kv included) behind the default preset, which
silently blocked picking q8_0 / q4_0 forever. These tests pin the corrected
behaviour so it cannot regress.
"""
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, ".")
from llamacpp_loader.gui.app import MainWindow


def _make_fake(preset_locked):
    """Build a bare MainWindow instance without running __init__ (no Tk)."""
    fake = object.__new__(MainWindow)
    captured = {}

    def start(row_id, col, x, y):
        captured["called"] = (row_id, col)

    fake._start_cell_edit = start
    fake._status_bar = MagicMock()
    fake._preset_is_locked = lambda name: preset_locked
    return fake, captured


def test_kv_editable_when_default_locked():
    fake, cap = _make_fake(preset_locked=True)
    MainWindow._maybe_edit_cell(fake, "mymodel", "kv")
    assert cap.get("called") == ("mymodel", "kv")


def test_kv_editable_when_unlocked():
    fake, cap = _make_fake(preset_locked=False)
    MainWindow._maybe_edit_cell(fake, "mymodel", "kv")
    assert cap.get("called") == ("mymodel", "kv")


def test_other_inference_knobs_editable_when_locked():
    fake, cap = _make_fake(preset_locked=True)
    for col in ("ctx", "gpu", "threads"):
        cap.clear()
        MainWindow._maybe_edit_cell(fake, "mymodel", col)
        assert cap.get("called") == ("mymodel", col), col


def test_sampling_locked_when_default_locked():
    fake, cap = _make_fake(preset_locked=True)
    MainWindow._maybe_edit_cell(fake, "mymodel", "temp")
    assert "called" not in cap


def test_non_editable_column_blocked():
    fake, cap = _make_fake(preset_locked=False)
    MainWindow._maybe_edit_cell(fake, "mymodel", "model")
    assert "called" not in cap
