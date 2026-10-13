"""Tests for the REPL live view: answers, status, and shell approvals."""

import io
import os
import threading

from ares.live import LiveView, visible_answer
from ares.runtime import tools
from ares.runtime.events import EventBus


def _pipe():
    r, w = os.pipe()
    return os.fdopen(r, "r"), os.fdopen(w, "w")


def test_visible_answer_waits_for_final_answer_in_react():
    assert visible_answer("coder", "Thought: hmm\nAction: x") == ""
    assert visible_answer("coder", "Thought: ok\nFinal Answer: 42") == "42"
    assert visible_answer("thinker", "<think>secret</think>answer") == "answer"
    assert visible_answer("thinker", "answer <thi") == "answer "


def test_view_installs_and_removes_the_approver():
    inp, _ = _pipe()
    with LiveView(EventBus(), "coder", out=io.StringIO(), inp=inp) as view:
        assert tools._confirm == view.confirm
    assert tools._confirm is None


def test_confirm_reads_the_answer():
    inp, w = _pipe()
    out = io.StringIO()
    with LiveView(EventBus(), "coder", out=out, inp=inp) as view:
        w.write("y\n")
        w.flush()
        assert view.confirm("coder wants to run: $ ls") is True
        w.write("\n")
        w.flush()
        assert view.confirm("coder wants to run: $ rm -rf /") is False
    assert "$ rm -rf /" in out.getvalue() and "Run it? [y/N]" in out.getvalue()


def test_cancelling_abandons_a_pending_question():
    inp, w = _pipe()
    view = LiveView(EventBus(), "orchestrator", out=io.StringIO(), inp=inp)
    view.__enter__()
    answers = []
    asker = threading.Thread(target=lambda: answers.append(view.confirm("runner wants to run: $ ls")))
    asker.start()
    view.__exit__(KeyboardInterrupt, None, None)  # Ctrl-C on the REPL thread
    asker.join(timeout=2)
    assert answers == [False]
    # The line typed next still reaches the REPL.
    w.write("next prompt\n")
    w.flush()
    assert inp.readline() == "next prompt\n"
