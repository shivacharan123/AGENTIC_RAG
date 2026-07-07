"""
prompts/grader.py — Grading Prompt Templates for Agentic RAG

Responsibilities:
    • Relevance grader: score (query, document) pairs as relevant or not.
    • Hallucination checker: verify the generated answer is grounded in context.
    • Answer quality grader: check if the answer actually addresses the question.

Position in the graph:
    retrieved_documents → relevance_grader node  → grade_relevance edge
    generated answer    → hallucination_checker  → check_hallucination edge
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas — structured output so graders never return free text
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""
    binary_score: str = Field(
        description="Document is relevant to the question: 'yes' or 'no'."
    )


class GradeHallucinations(BaseModel):
    """Binary hallucination check for a generated answer."""
    binary_score: str = Field(
        description=(
            "Answer is grounded in and supported by the provided facts: "
            "'yes' (grounded) or 'no' (contains hallucinations)."
        )
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific claims in the answer that are NOT supported "
            "by the provided context. Empty list if binary_score is 'yes'."
        ),
    )


class GradeAnswer(BaseModel):
    """Binary answer quality check."""
    binary_score: str = Field(
        description=(
            "Answer actually addresses / resolves the question: "
            "'yes' or 'no'."
        )
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the answer does or does not address the question.",
    )


# ---------------------------------------------------------------------------
# Relevance grader prompt
# ---------------------------------------------------------------------------

relevance_grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a strict relevance grader inside a self-corrective Agentic RAG system.\n\n"
            "Your job: decide if a retrieved document contains information that is "
            "useful for answering the user's question.\n\n"
            "Grading rules:\n"
            "  • Score 'yes' if the document contains ANY relevant facts, entities, "
            "    keywords, or context that could help answer the question — even partially.\n"
            "  • Score 'no' if the document is completely unrelated to the question.\n"
            "  • Do NOT require the document to fully answer the question on its own.\n"
            "  • Do NOT consider writing quality — only information relevance.\n\n"
            "Return ONLY the binary_score field: 'yes' or 'no'."
        ),
    ),
    (
        "human",
        "Retrieved document:\n\n{document}\n\nUser question: {question}",
    ),
])

# ---------------------------------------------------------------------------
# Hallucination checker prompt
# ---------------------------------------------------------------------------

hallucination_checker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a strict hallucination detector inside a self-corrective Agentic RAG system.\n\n"
            "Your job: verify that EVERY factual claim in the generated answer is directly "
            "supported by the provided context documents.\n\n"
            "Grading rules:\n"
            "  • Score 'yes' (grounded) ONLY if every claim in the answer can be traced "
            "    back to a specific statement in the context.\n"
            "  • Score 'no' (hallucination) if the answer contains ANY fact, number, name, "
            "    date, or claim that is NOT present in or directly inferable from the context.\n"
            "  • Populate 'unsupported_claims' with the exact phrases that are not grounded.\n"
            "  • Statements like 'the context does not provide...' are acceptable and not hallucinations.\n"
            "  • Do NOT penalise the answer for being incomplete — only for being fabricated."
        ),
    ),
    (
        "human",
        "Context documents:\n\n{context}\n\nGenerated answer:\n\n{generation}",
    ),
])

# ---------------------------------------------------------------------------
# Answer quality grader prompt
# ---------------------------------------------------------------------------

answer_quality_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an answer quality evaluator.\n\n"
            "Your job: decide if the generated answer actually resolves the user's question.\n\n"
            "Score 'yes' if the answer:\n"
            "  • Directly addresses what was asked.\n"
            "  • Provides a useful, actionable response.\n\n"
            "Score 'no' if the answer:\n"
            "  • Says 'I don't know' or 'the context doesn't mention' without attempting an answer.\n"
            "  • Answers a different question than what was asked.\n"
            "  • Is empty or a non-answer."
        ),
    ),
    (
        "human",
        "User question: {question}\n\nGenerated answer: {generation}",
    ),
])
