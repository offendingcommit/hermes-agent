"""Abstract base class for pluggable context engines.

A context engine controls how conversation context is managed when
approaching the model's token limit. The built-in ContextCompressor
is the default implementation. Third-party engines (e.g. LCM) can
replace it via the plugin system or by being placed in the
``plugins/context_engine/<name>/`` directory.

Selection is config-driven: ``context.engine`` in config.yaml.
Default is ``"compressor"`` (the built-in). Only one engine is active.

The engine is responsible for:
  - Deciding when compaction should fire
  - Performing compaction (summarization, DAG construction, etc.)
  - Optionally exposing tools the agent can call (e.g. lcm_grep)
  - Tracking token usage from API responses

Lifecycle:
  1. Engine is instantiated and registered (plugin register() or default)
  2. on_session_start() called when a conversation begins
  3. update_from_response() called after each API response with usage data
  4. should_compress() checked after each turn
  5. compress() called when should_compress() returns True
  6. on_session_end() called at real session boundaries (CLI exit, /reset,
     gateway session expiry) — NOT per-turn
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import secrets
import threading
from typing import Any, Dict, List, Optional, Sequence

from agent.redact import redact_sensitive_text


MEMORY_CONTEXT_MAX_CHARS = 6_000
_MEMORY_CONTEXT_HEAD_CHARS = 4_000
_MEMORY_CONTEXT_TAIL_CHARS = 1_500
_MEMORY_CONTEXT_TRUNCATION_MARKER = "\n...[memory provider context truncated]...\n"


class StaleSessionBindingError(RuntimeError):
    """Raised when an engine uses a revoked session capability."""


class ContextSessionBlockedError(RuntimeError):
    """Raised before model execution while a session is durably blocked."""


@dataclass(frozen=True)
class CompactionOutcome:
    """Typed result accepted from context engines; plain message lists remain valid."""

    status: str
    messages: Optional[List[Dict[str, Any]]] = None
    transaction_id: Optional[str] = None
    blocked_reason: Optional[str] = None

    @classmethod
    def compacted(cls, messages, transaction_id: str):
        return cls("compacted", list(messages), transaction_id=transaction_id)

    @classmethod
    def noop(cls):
        return cls("noop")

    @classmethod
    def blocked(cls, reason: str):
        return cls("blocked", blocked_reason=reason)


class SessionContextStore:
    """Bounded, current-session-only view of canonical Hermes history.

    The session id and an unguessable incarnation are captured by the host and
    are deliberately absent from every public read method.
    """

    def __init__(self, session_db, session_id: str, *, max_query_chars=512,
                 max_results=25, max_span=50):
        self._db = session_db
        self._session_id = session_id
        self._incarnation = secrets.token_urlsafe(32)
        self._max_query_chars = max(1, int(max_query_chars))
        self._max_results = max(1, int(max_results))
        self._max_span = max(1, int(max_span))
        self._active = True
        self._lock = threading.RLock()

    @property
    def incarnation(self) -> str:
        return self._incarnation

    def revoke(self) -> None:
        with self._lock:
            self._active = False

    def __deepcopy__(self, memo):
        raise TypeError("live session capabilities cannot be copied")

    def _check(self) -> None:
        with self._lock:
            if not self._active:
                raise StaleSessionBindingError("context-engine session binding is stale")

    def search(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        self._check()
        query = str(query or "")
        if len(query) > self._max_query_chars:
            raise ValueError("query exceeds bound")
        limit = max(1, min(int(limit), self._max_results))
        return self._db.search_session_messages(self._session_id, query, limit=limit)

    def around_message(self, message_id: int, *, before: int = 2,
                       after: int = 2) -> List[Dict[str, Any]]:
        self._check()
        before, after = max(0, int(before)), max(0, int(after))
        if before + after + 1 > self._max_span:
            raise ValueError("span exceeds bound")
        return self._db.get_session_messages_around(
            self._session_id, int(message_id), before=before, after=after
        )

    def receipt(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        self._check()
        return self._db.get_compaction_receipt(self._session_id, transaction_id)

    def receipts(self, *, limit: int = 25) -> List[Dict[str, Any]]:
        """Return the bound session's newest durable receipts in stable order."""
        self._check()
        return self._db.list_compaction_receipts(
            self._session_id, limit=min(max(1, int(limit)), self._max_results)
        )

    def deletion_tombstones(self, *, limit: int = 25) -> List[Dict[str, Any]]:
        self._check()
        return self._db.get_session_deletion_tombstones(limit=min(int(limit), self._max_results))

    def acknowledge_deletion(self, tombstone_id: int) -> bool:
        self._check()
        return self._db.acknowledge_session_deletion_tombstone(int(tombstone_id))

    def clear_blocked_state(self) -> None:
        self._check()
        self._db.clear_context_session_block(self._session_id)


def sanitize_memory_context(memory_context: str) -> str:
    """Prepare provider context for a context-engine/LLM egress boundary."""
    sanitized = redact_sensitive_text(
        memory_context.strip(),
        force=True,
        redact_url_credentials=True,
    )
    if len(sanitized) <= MEMORY_CONTEXT_MAX_CHARS:
        return sanitized
    return (
        sanitized[:_MEMORY_CONTEXT_HEAD_CHARS]
        + _MEMORY_CONTEXT_TRUNCATION_MARKER
        + sanitized[-_MEMORY_CONTEXT_TAIL_CHARS:]
    )


class ContextEngine(ABC):
    """Base class all context engines must implement."""

    # -- Identity ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    # -- Token state (read by run_agent.py for display/logging) ------------
    #
    # Engines MUST maintain these. run_agent.py reads them directly.

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0
    _session_store: Optional[SessionContextStore] = None

    # -- Compaction parameters (read by run_agent.py for preflight) --------
    #
    # These control the preflight compression check.  Subclasses may
    # override via __init__ or property; defaults are sensible for most
    # engines.
    #
    # protect_first_n semantics (since PR #13754): count of non-system head
    # messages always preserved verbatim, IN ADDITION to the system prompt
    # which is always implicitly protected.  Default 3 keeps the
    # historical "system + first 3 non-system messages" head shape.

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- Core interface ----------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Called after every LLM call with a normalized usage dict. The legacy
        keys ``prompt_tokens``, ``completion_tokens``, and ``total_tokens``
        are always present. Newer hosts also include canonical buckets:
        ``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
        ``cache_write_tokens``, and ``reasoning_tokens``. Engines should
        treat those fields as optional for compatibility with older hosts.
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Compact the message list and return the new message list.

        This is the main entry point. The engine receives the full message
        list and returns a (possibly shorter) list that fits within the
        context budget. The implementation is free to summarize, build a
        DAG, or do anything else — as long as the returned list is a valid
        OpenAI-format message sequence.

        Args:
            focus_topic: Optional topic string from manual ``/compress <focus>``.
                Engines that support guided compression should prioritise
                preserving information related to this topic.  Engines that
                don't support it may simply ignore this argument.
            force: Whether a user-requested compression should bypass an
                engine-owned cooldown. Engines without cooldowns may ignore it.
            memory_context: Text returned by memory providers immediately before
                compaction. Summarizing engines should include non-empty text in
                their handoff prompt. Older engines may omit this parameter; the
                host filters unsupported optional arguments by signature.
        """

    # -- Optional: pre-flight check ----------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before the API call (no real token count yet).

        Default returns False (skip pre-flight). Override if your engine
        can do a cheap estimate.
        """
        return False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """Return True when preflight should trust recent real usage instead.

        Built-in compression uses this to avoid re-compacting from known-noisy
        rough estimates after a compressed request has already fit. Third-party
        engines can ignore it safely.
        """
        return False

    def is_compaction_eligible(self, messages: Sequence[Dict[str, Any]], *,
                               prompt_tokens: int, emergency: bool = False) -> bool:
        """Message-aware host gate before policy-driven compaction.

        The base contract stays permissive for existing engines. Engines with
        reclaimability rules override this method.
        """
        return True

    # -- Optional: manual /compress preflight ------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick check: is there anything in ``messages`` that can be compacted?

        Used by the gateway ``/compress`` command as a preflight guard —
        returning False lets the gateway report "nothing to compress yet"
        without making an LLM call.

        Default returns True (always attempt).  Engines with a cheap way
        to introspect their own head/tail boundaries should override this
        to return False when the transcript is still entirely protected.
        """
        return True

    # -- Optional: session lifecycle ---------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins.

        Use this to load persisted state (DAG, store) for the session.
        kwargs may include hermes_home, platform, model, etc.
        """

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Called at real session boundaries (CLI exit, /reset, gateway expiry).

        Use this to flush state, close DB connections, etc.
        NOT called per-turn — only when the session truly ends.
        """

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Reset per-session state.

        Default resets compression_count and token tracking.
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.revoke_session_state()

    def bind_session_state(self, *, session_db, session_id: str, **limits) -> SessionContextStore:
        """Bind and return a fresh trusted capability, revoking its predecessor."""
        self.revoke_session_state()
        self._session_store = SessionContextStore(session_db, session_id, **limits)
        return self._session_store

    def revoke_session_state(self) -> None:
        store = getattr(self, "_session_store", None)
        if store is not None:
            store.revoke()
        self._session_store = None

    # -- Optional: tools ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this engine provides to the agent.

        Default returns empty list (no tools). LCM would return schemas
        for lcm_grep, lcm_describe, lcm_expand here.
        """
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent.

        Only called for tool names returned by get_tool_schemas().
        Must return a JSON string.

        kwargs may include:
          messages: the current in-memory message list (for live ingestion)
        """
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Optional: status / display ----------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging.

        Default returns the standard fields run_agent.py expects.
        """
        # Clamp the -1 "compression just ran, awaiting real usage" sentinel
        # (set by conversation_compression) to 0 so status readers don't see a
        # raw -1 or a negative usage_percent on the transitional turn. Mirrors
        # the CLI/gateway status-bar paths (cli.py, tui_gateway/server.py).
        last_prompt = self.last_prompt_tokens if self.last_prompt_tokens > 0 else 0
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, last_prompt / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- Optional: model switch support ------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
        output_token_budget: Optional[int] = None,
    ) -> None:
        """Called when the user switches models or on fallback activation.

        Default updates context_length and recalculates threshold_tokens
        from threshold_percent. Override if your engine needs more
        (e.g. recalculate DAG budgets, switch summary models).
        """
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
        if output_token_budget is not None:
            self.output_token_budget = int(output_token_budget)
