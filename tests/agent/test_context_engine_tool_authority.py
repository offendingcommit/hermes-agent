from agent.context_engine import ContextEngine
from hermes_state import SessionDB


class Engine(ContextEngine):
    name = "test"
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, **kwargs): return messages


def test_tool_arguments_have_no_session_authority(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("trusted", source="test")
    db.create_session("attacker-selected", source="test")
    db.append_message("trusted", "user", "trusted value")
    db.append_message("attacker-selected", "user", "secret value")
    store = Engine().bind_session_state(session_db=db, session_id="trusted")
    malicious_args = {"session_id": "attacker-selected", "query": "value"}
    rows = store.search(malicious_args["query"])
    assert [row["content"] for row in rows] == ["trusted value"]
