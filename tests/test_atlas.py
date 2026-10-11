"""Integration test against the real cluster. Skipped unless MONGODB_URI is set."""
import pytest

from hx.config import load_settings
from hx.memory import cases as case_ops
from hx.memory import harness as rule_ops
from hx.memory.models import TurnTrace

pytestmark = pytest.mark.atlas


@pytest.fixture
def mongo():
    s = load_settings()
    if not s.mongodb_uri:
        pytest.skip("MONGODB_URI not set")
    from hx.memory.store import MongoStore

    st = MongoStore(s.mongodb_uri, "hx_memory_pytest")
    yield st
    st.client.drop_database("hx_memory_pytest")


def test_atlas_roundtrip(mongo):
    assert mongo.current_harness().version == 0
    rule_ops.add_rule(mongo, "r", "Do X.")
    rule_ops.judge_rules(mongo, {"r": "helped"})
    hv = mongo.current_harness()
    assert hv.version == 2 and hv.get("r").status == "active"
    assert [h.version for h in mongo.harness_history()] == [2, 1, 0]
    case_ops.file_case(mongo, "t", "desc", "**Symptom:** x", case_id="c")
    assert mongo.get_case("c").description == "desc"
    t = TurnTrace(task="probe")
    mongo.save_trace(t)
    assert mongo.get_trace(t.trace_id).task == "probe"
