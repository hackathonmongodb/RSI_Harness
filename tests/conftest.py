from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.mocks import reset_store  # noqa: E402

B_TESTS = {"test_harness.py", "test_llm.py", "test_task_agent.py"}


@pytest.fixture(autouse=True)
def _isolate_c_tests(request, monkeypatch):
    name = Path(str(request.fspath)).name
    if name not in B_TESTS:
        monkeypatch.setenv("HX_USE_MOCKS", "1")
    reset_store()
