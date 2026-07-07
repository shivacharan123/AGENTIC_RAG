"""
Relevance Grader Agent - Filters irrelevant retrieved documents.
Core part of the self-corrective retrieval loop.
"""

from typing import List
from langchain_core.documents import Document

# Use the actual prompt template and schema from prompts/grader.py
from ..prompts.grader import relevance_grader_prompt, GradeDocuments


class RelevanceGrader:
    """Grades relevance of retrieved documents against a user query."""

    def __init__(self, llm):
        self.llm = llm
        structured_llm = llm.with_structured_output(GradeDocuments)
        # Wire the real prompt template to the structured LLM
        self.grader_chain = relevance_grader_prompt | structured_llm

    def grade(self, documents, query: str):
        """Grade all documents in a single LLM call instead of one call per doc."""
        if not documents:
            return []

        docs_text = "\n\n".join([
            f"[Doc {i+1}]: {doc.page_content[:600]}"
            for i, doc in enumerate(documents)
        ])

        prompt = f"""You are a relevance grader. Given a user query and a list of documents, 
    determine which documents are relevant to answering the query.

    Query: {query}

    Documents:
    {docs_text}

    Reply with ONLY a JSON array of booleans, one per document, in order.
    true = relevant, false = not relevant.
    Example for 3 documents: [true, false, true]
    No explanation, no markdown, just the JSON array."""

        try:
            result = self.llm.invoke(prompt)
            import json, re
            text = result.content.strip()
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if not match:
                return documents  # pass all if parsing fails
            grades = json.loads(match.group())
            if len(grades) != len(documents):
                return documents  # length mismatch — pass all
            filtered = [doc for doc, g in zip(documents, grades) if g]
            return filtered if filtered else documents  # never return empty
        except Exception as e:
            print(f"[RelevanceGrader] Batch grading failed ({e}), passing all docs.")
            return documents