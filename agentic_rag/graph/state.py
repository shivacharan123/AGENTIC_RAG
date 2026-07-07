"""
graph/state.py — GraphState definition and custom reducers.

This file is shown as a PATCH.  Only merge_documents is changed.
Replace your existing merge_documents with the version below.

Key change (Issue 2 & 3 fix):
    When query_rewriter returns `retrieved_documents = []` to signal "start
    fresh", the old reducer blindly did `left + right = left + [] = left`,
    silently keeping all stale documents.

    The new reducer treats an *explicit* empty-list write as a RESET:
      • right is None        → keep left unchanged (branch produced nothing)
      • right is []          → clear: discard left, return []
      • right is non-empty   → append: return left + right  (original behaviour)

    This means query_rewriter can clear the accumulator by returning
    {"retrieved_documents": []} while retrieval nodes that legitimately
    find nothing (also returning []) are distinguished from a reset by
    the node that calls them — the reset always comes from query_rewriter
    BEFORE the retrieval fan-out, so by the time a retrieval node returns
    [] the accumulator is already empty and left + [] = [] is still correct.

    In practice this is safe because:
      1. query_rewriter emits [] BEFORE the next fan-out round.
      2. Each retrieval node then appends its (possibly empty) results.
      3. context_aggregator sees only the current round's documents.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional, Union

from langchain_core.documents import Document

from agentic_rag.prompts.grader import GradeHallucinations


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------

def merge_documents(
    left: Optional[List[Document]],
    right: Optional[Union[Document, List[Document]]],
) -> List[Document]:
    """
    LangGraph reducer for retrieved_documents.

    Behaviour
    ---------
    right is None          → no-op: return left unchanged
    right is []            → RESET: discard left, return []   ← NEW
    right is a Document    → append single doc to left
    right is a non-empty   → append list to left
      list of Documents

    The RESET semantic is what lets query_rewriter clear stale documents
    before a new retrieval round without needing a separate state key.
    """
    # Normalise left
    if left is None:
        left = []

    # No update from this branch — keep accumulator as-is
    if right is None:
        return left

    # Normalise right to a list
    if isinstance(right, Document):
        right_list: List[Document] = [right]
    else:
        right_list = list(right)  # copy so we don't mutate the caller's list

    # ── RESET signal ────────────────────────────────────────────────────────
    # An explicit empty list written to retrieved_documents means "start over".
    # This is emitted by query_rewriter before each retry round and by
    # query_analyzer at the start of each new request.
    if len(right_list) == 0:
        return []

    # ── Normal append ────────────────────────────────────────────────────────
    return left + right_list


# ---------------------------------------------------------------------------
# Example GraphState (add / adjust fields to match your actual state.py)
# ---------------------------------------------------------------------------
# Shown here for completeness so you can see exactly how to annotate the
# field.  If your state.py already has these annotations, just replace the
# merge_documents function above.

from typing import TypedDict

class GraphState(TypedDict, total=False):
    # Input
    user_query:             str
    chat_history:           List[Any]

    # Routing
    current_query:          str
    query_classification:   str
    sub_queries:            List[str]
    route_decision:         str

    # Retrieval — uses merge_documents reducer
    retrieved_documents:    Annotated[List[Document], merge_documents]

    # Post-retrieval
    aggregated_context:     List[Document]
    relevance_status:       str

    # Generation
    generation:             str
    hallucination_status:   str
    hallucination_grade:    GradeHallucinations | None

    # Counters
    retry_count:            int
    generation_retry_count: int