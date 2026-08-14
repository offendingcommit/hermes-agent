import copy

import pytest

from agent.context_engine import ContextEngine, StaleSessionBindingError
from hermes_state import SessionDB


class Engine(ContextEngine):
    name = "test"
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, **kwargs): return messages


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    for sid in ("one", "two"):
        db.create_session(sid, source="test")
    db.append_message("one", "user", "private one")
    db.append_message("one", "assistant", "answer one")
    db.append_message("two", "user", "private two")
    return db


def test_bound_reads_cannot_select_another_session_and_include_archived(tmp_path):
    db = _db(tmp_path)
    store = Engine().bind_session_state(session_db=db, session_id="one")
    assert [row["content"] for row in store.search("private")] == ["private one"]
    message_id = store.search("private")[0]["id"]
    assert [row["content"] for row in store.around_message(message_id, before=0, after=1)] == [
        "private one", "answer one"
    ]
    db.archive_and_compact("one", [{"role": "user", "content": "summary"}])
    assert store.search("private")[0]["active"] == 0


def test_bounds_and_revocation_are_enforced(tmp_path):
    db = _db(tmp_path)
    engine = Engine()
    old = engine.bind_session_state(
        session_db=db, session_id="one", max_query_chars=4, max_results=2, max_span=2
    )
    with pytest.raises(ValueError, match="query"):
        old.search("12345")
    with pytest.raises(ValueError, match="span"):
        old.around_message(1, before=1, after=1)
    engine.bind_session_state(session_db=db, session_id="two")
    with pytest.raises(StaleSessionBindingError):
        old.search("one")
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(engine)


def test_deletion_tombstone_is_durable_and_acknowledged(tmp_path):
    db = _db(tmp_path)
    store = Engine().bind_session_state(session_db=db, session_id="one")
    assert db.delete_session("two") is True
    tombstones = store.deletion_tombstones()
    assert len(tombstones) == 1
    assert "two" not in tombstones[0]["session_id_hash"]
    assert store.acknowledge_deletion(tombstones[0]["id"]) is True
    assert store.deletion_tombstones() == []


def test_receipt_enumeration_rebuilds_lost_derived_index(tmp_path):
    db = _db(tmp_path)
    db.archive_and_compact(
        "one", [{"role": "assistant", "content": "first summary"}],
        transaction_id="tx-1", active_summary_hash="hash-1",
    )
    db.archive_and_compact(
        "one", [{"role": "assistant", "content": "second summary"}],
        transaction_id="tx-2", active_summary_hash="hash-2",
    )
    db.archive_and_compact(
        "two", [{"role": "assistant", "content": "other summary"}],
        transaction_id="other", active_summary_hash="other-hash",
    )
    store = Engine().bind_session_state(
        session_db=db, session_id="one", max_results=1
    )

    # A plugin with no derived index can discover the newest durable lineage
    # receipt without knowing its transaction id, but cannot escape host bounds.
    receipts = store.receipts(limit=99)
    assert [receipt["transaction_id"] for receipt in receipts] == ["tx-2"]
    assert receipts[0]["archived_message_ids"]
    assert receipts[0]["active_message_ids"]
