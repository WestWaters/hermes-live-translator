import pytest

from live_translate.state import IllegalTransition, State


def test_a_session_runs_through_its_states():
    s = State("idle")
    for step in ("starting", "live", "stopping", "idle"):
        s.to(step)
    assert s.snapshot()["session"] == "idle"
    assert [h[2] for h in s.history] == ["starting", "live", "stopping", "idle"]


def test_illegal_moves_are_refused():
    s = State("idle")
    s.to("starting")
    s.to("live")
    with pytest.raises(IllegalTransition):
        s.to("setting_up")


def test_live_facets_reset_on_a_new_session():
    s = State("idle")
    s.to("starting"), s.to("live")
    s.facet(activity="hearing", language="zh", voice="cloned")
    assert s.snapshot()["live"]["language"] == "zh"
    s.to("stopping"), s.to("idle"), s.to("starting")
    assert s.facets == {"activity": "waiting", "language": "detecting", "voice": "stock", "pace": "on_time"}


def test_errors_carry_a_reason_and_can_be_retried():
    s = State("idle")
    s.to("starting")
    s.to("error", "no microphone found")
    snap = s.snapshot()
    assert "no microphone found" in snap["message"] and "starting" in snap["can"]
