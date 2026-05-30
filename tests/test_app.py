"""AppTest-based tests for app.py.

Uses streamlit.testing.v1.AppTest to exercise the app without a running server.
All tests use a 90-second timeout.
"""

import re
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def _get_summary(at) -> str:
    """Return the view-summary markdown block (index 11)."""
    return at.markdown[11].value


def _bin_count(at) -> int:
    """Extract the bin count from the summary markdown."""
    m = re.search(r"<b>(\d+)</b> bin", _get_summary(at))
    assert m, "Could not find bin count in summary"
    return int(m.group(1))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_app_runs_without_exception():
    """App runs without raising an exception."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    assert not at.exception


def test_app_default_title():
    """App shows 'Ranked Placement Atlas' in markdown."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    title_md = at.markdown[1].value
    assert "Ranked Placement Atlas" in title_md


def test_app_default_bin_count():
    """View summary contains '100 bins' for synthetic demo data."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    assert _bin_count(at) == 100


def test_app_method_pills_present():
    """The method_pills widget exists with default 'Majority'."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    method_pill = at.pills[2]   # index 2 = method_pills
    assert method_pill.key == "method_pills"
    assert method_pill.value == "Majority"


def test_app_sort_radio_present():
    """The sort_radio widget exists with default 'Similarity'."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    sort_r = at.radio[0]   # index 0 = sort_radio
    assert sort_r.key == "sort_radio"
    assert sort_r.value == "Similarity"


def test_app_deselect_segment():
    """Set segments_pills to ['NA'] then run; bin count in summary < 100."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    at.pills[0].set_value(["NA"])   # index 0 = segments_pills
    at.run(timeout=90)
    assert not at.exception
    assert _bin_count(at) < 100


def test_app_change_method():
    """Set method_pills to 'Weighted' then run; summary contains 'Weighted'."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    at.pills[2].set_value("Weighted")   # index 2 = method_pills
    at.run(timeout=90)
    assert not at.exception
    assert "Weighted" in _get_summary(at)


def test_app_reset_button():
    """After deselecting a segment, clicking the reset button restores count to 100."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    # Filter to just NA segment
    at.pills[0].set_value(["NA"])
    at.run(timeout=90)
    assert _bin_count(at) < 100
    # Click reset
    at.button(key="btn_reset_filters").click()
    at.run(timeout=90)
    assert not at.exception
    assert _bin_count(at) == 100


def test_app_position_range():
    """Set pos_slider to (1, 10) then run; summary contains '10 positions'."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    at.slider(key="pos_slider").set_value((1, 10))
    at.run(timeout=90)
    assert not at.exception
    assert re.search(r"<b>10</b> position", _get_summary(at))


def test_app_rank_range():
    """Set rank_slider to (1, 50) then run; bin count <= 50."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)
    at.slider(key="rank_slider").set_value((1, 50))
    at.run(timeout=90)
    assert not at.exception
    assert _bin_count(at) <= 50
