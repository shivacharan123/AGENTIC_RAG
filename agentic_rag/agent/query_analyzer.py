"""
Query Analyzer Agent - Routes queries and decomposes complex questions.
Part of the Planning layer in Agentic RAG.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Use the actual chain factories from query_transform.py
from agentic_rag.prompts.query_transform import (
    get_query_rewriter_chain,
    get_decomposition_chain,
    get_stepback_chain,
    QueryTransformConfig,
)
from agentic_rag.prompts.config import load_config


class RouteQuery(BaseModel):
    """Structured routing decision."""
    datasource: str = Field(
        description="Route to 'vectorstore', 'websearch', 'sql', or 'parallel_all'."
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision."
    )


class QueryAnalyzer:
    """Analyzes and routes user queries, applies query transformations."""

    def __init__(self, llm):
        self.llm = llm
        self.config = load_config()
        self._transform_config = QueryTransformConfig()

        # Structured output for routing
        structured_llm = llm.with_structured_output(RouteQuery)

        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert query router for an Agentic RAG system.\n"
                "The system has three retrieval sources:\n"
                "  • 'vectorstore'  — internal documents, reports, PDFs\n"
                "  • 'websearch'    — current events, real-time info\n"
                "  • 'sql'          — structured/tabular data, statistics, metrics\n"
                "  • 'parallel_all' — complex queries needing all three sources\n\n"
                "Return ONLY a JSON with 'datasource' and 'reasoning'."
            )),
            ("human", "Question: {question}"),
        ])
        self.router_chain = self.router_prompt | structured_llm

        # Query rewriter chain (used when retrieval fails and query needs improvement)
        self.rewriter_chain = get_query_rewriter_chain(llm, self._transform_config)

    def analyze(self, question: str) -> Dict[str, Any]:
        """Analyze query: decide routing and apply optional transformations."""
        # 1. Routing decision
        try:
            route = self.router_chain.invoke({"question": question})
        except Exception as e:
            print(f"[QueryAnalyzer] router_chain.invoke failed: {e}")
            route = None

        if route is None or not hasattr(route, "datasource"):
            print("[QueryAnalyzer] Structured routing output invalid — defaulting to parallel_all.")
            datasource = "parallel_all"
            reasoning = "Routing LLM call failed or returned unparseable output; defaulting to parallel_all."
        else:
            datasource = route.datasource
            reasoning = route.reasoning

        # 2. Rewrite the query for better retrieval
        try:
            rewrite_result = self.rewriter_chain.invoke({
                "question": question,
                "retry_count": 0,
                "failure_reason": "Initial query — proactive rewrite for retrieval quality.",
            })
            rewritten = rewrite_result.rewritten_query
        except Exception:
            rewritten = question  # safe fallback: keep original

        return {
            "datasource": datasource,
            "reasoning": reasoning,
            "original_query": question,
            "rewritten_query": rewritten,
        }
