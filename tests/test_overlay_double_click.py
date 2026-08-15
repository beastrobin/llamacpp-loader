"""Regression tests for the green "changed" cell double-click detection.

Changed (green) KV cells used to be uneditable because Tk's native
<Double-1> synthesis is defeated when the overlay's <Button-1> handler calls
event_generate. The overlay now detects double-clicks itself via
_is_overlay_double_click — this pins that logic down.
"""
import pytest

from llamacpp_loader.gui.app import MainWindow


def _make() -> MainWindow:
    fake = object.__new__(MainWindow)
    fake._overlay_last_click = None
    return fake


def test_first_click_is_not_double():
    fake = _make()
    assert fake._is_overlay_double_click("r1", "kv", 1000) is False
    # state recorded for the same cell
    assert fake._overlay_last_click == ("r1", "kv", 1000)


def test_second_click_same_cell_is_double():
    fake = _make()
    fake._is_overlay_double_click("r1", "kv", 1000)
    assert fake._is_overlay_double_click("r1", "kv", 1200) is True
    # state resets after a double-click fires
    assert fake._overlay_last_click is None


def test_different_cell_is_not_double():
    fake = _make()
    fake._is_overlay_double_click("r1", "kv", 1000)
    assert fake._is_overlay_double_click("r2", "kv", 1100) is False
    assert fake._overlay_last_click == ("r2", "kv", 1100)


def test_too_slow_is_not_double():
    fake = _make()
    assert fake._is_overlay_double_click("r1", "kv", 1000) is False
    # 1000ms later (> 400ms threshold) is treated as a fresh first click
    assert fake._is_overlay_double_click("r1", "kv", 2000) is False


def test_boundary_threshold():
    fake = _make()
    fake._is_overlay_double_click("r1", "kv", 1000)
    # exactly 399ms -> still a double; 400ms -> not
    assert fake._is_overlay_double_click("r1", "kv", 1399) is True
    fake._is_overlay_double_click("r1", "kv", 2000)
    assert fake._is_overlay_double_click("r1", "kv", 2400) is False
