"""
agents/hallucination_checker.py — Hallucination Detection & Grounding Agent
"""

from __future__ import annotations
from pydantic import SecretStr
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, cast
import os
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from agentic_rag.prompts.generator import format_docs


logger = logging.getLogger(__name__)


GRADE_GROUNDED    = "grounded"
GRADE_REGENERATE  = "regenerate"

VALID_HALLUCINATION_GRADES = {GRADE_GROUNDED, GRADE_REGENERATE}

SEVERITY_NONE     = "none"
SEVERITY_MINOR    = "minor"
SEVERITY_MAJOR    = "major"
SEVERITY_CRITICAL = "critical"


@dataclass
class HallucinationCheckerConfig:
    model: str                       = "llama-3.3-70b-versatile"   # ← was "gpt-4o-mini"
    temperature: float               = 0.0
    max_regeneration_attempts: int   = 2
    grounding_score_threshold: float = 0.85
    llm_kwargs: Dict[str, Any]       = field(default_factory=dict)


class GradeHallucinations(BaseModel):
    binary_score: Literal["grounded", "regenerate"] = Field(
        description=(
            "Hallucination verdict:\n"
            "  'grounded'    — every factual claim in the answer is directly "
            "supported by the provided context.\n"
            "  'regenerate'  — the answer contains one or more unsupported facts."
        )
    )

    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims NOT traceable to the context.",
    )

    severity: Literal["none", "minor", "major", "critical"] = Field(
        default="none",
    )

    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)

    supported_claims: List[str] = Field(default_factory=list)

    reasoning: str = Field(default="")

    @field_validator("binary_score")
    @classmethod
    def must_be_grounded_or_regenerate(cls, v: str) -> str:
        normalised = v.strip().lower()
        if normalised not in VALID_HALLUCINATION_GRADES:
            raise ValueError(f"binary_score must be 'grounded' or 'regenerate', got {v!r}")
        return normalised

    @field_validator("severity")
    @classmethod
    def must_be_valid_severity(cls, v: str) -> str:
        valid = {SEVERITY_NONE, SEVERITY_MINOR, SEVERITY_MAJOR, SEVERITY_CRITICAL}
        normalised = v.strip().lower()
        if normalised not in valid:
            raise ValueError(f"severity must be one of {valid}, got {v!r}")
        return normalised

    @field_validator("grounding_score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def is_grounded(self) -> bool:
        return self.binary_score == GRADE_GROUNDED

    @property
    def is_critical(self) -> bool:
        return self.severity == SEVERITY_CRITICAL

    @property
    def formatted_unsupported_claims(self) -> str:
        if not self.unsupported_claims:
            return "None"
        return "\n".join(f"  • {c}" for c in self.unsupported_claims)


HALLUCINATION_CHECKER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a meticulous hallucination detector inside an Agentic RAG "
            "self-corrective loop.\n\n"
            "Your task: verify that EVERY factual claim in the generated answer is "
            "directly traceable to the provided context documents.\n\n"
            "OUTPUT RULES:\n"
            "  • Set binary_score = 'grounded' ONLY if ALL claims are supported.\n"
            "  • Set binary_score = 'regenerate' if ANY non-trivial claim is unsupported.\n"
            "  • List every unsupported claim verbatim in unsupported_claims.\n"
            "  • Set grounding_score = (supported claims) / (total claims).\n"
            "  • Be RUTHLESS."
        ),
    ),
    (
        "human",
        (
            "User question: {question}\n\n"
            "Context documents:\n{context}\n\n"
            "Generated answer to verify:\n{generation}\n\n"
            "Regeneration attempt number: {regeneration_attempt}"
        ),
    ),
])


def get_hallucination_checker_chain(
    llm: Optional[BaseChatModel] = None,
    config: Optional[HallucinationCheckerConfig] = None,
):
    config = config or HallucinationCheckerConfig()
    llm    = llm    or _build_llm(config)
    return HALLUCINATION_CHECKER_PROMPT | llm.with_structured_output(GradeHallucinations)


class HallucinationCheckerAgent:
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[HallucinationCheckerConfig] = None,
    ) -> None:
        self.config = config or HallucinationCheckerConfig()
        self._llm   = llm or _build_llm(self.config)
        self._chain = get_hallucination_checker_chain(self._llm, self.config)

        # Warning 3 fix: when a shared LLM is injected (nodes.py path), self.config.model
        # retains its default "gpt-4o-mini" even though a completely different model is
        # actually being used.  Resolve the true model name from the LLM object so the
        # log line is always accurate regardless of how the agent is constructed.
        _actual_model: str = (
            getattr(self._llm, "model_name", None)      # ChatOpenAI
            or getattr(self._llm, "model", None)         # some other LangChain wrappers
            or self.config.model                         # fallback: at least log the config value
        )
        logger.info(
            "HallucinationCheckerAgent initialised "
            "(actual_model=%s, config_model=%s, max_regen=%d, threshold=%.2f).",
            _actual_model,
            self.config.model,
            self.config.max_regeneration_attempts,
            self.config.grounding_score_threshold,
        )

    def check(
        self,
        question: str,
        documents: List[Document],
        generation: str,
        regeneration_attempt: int = 0,
    ) -> GradeHallucinations:
        if not generation or not generation.strip():
            logger.warning("HallucinationChecker: empty generation → 'regenerate'.")
            return _regenerate_fallback(
                reason="Empty generation string.",
                unsupported=["[entire answer is empty]"],
            )

        if not documents:
            logger.warning(
                "HallucinationChecker: no context documents → cannot verify → 'regenerate'."
            )
            return _regenerate_fallback(
                reason="No context documents available to verify against.",
                unsupported=["All claims are unverifiable — no context provided."],
            )

        if regeneration_attempt >= self.config.max_regeneration_attempts:
            logger.warning(
                "HallucinationChecker: regeneration_attempt=%d >= max=%d. "
                "Accepting current answer as 'grounded' to prevent infinite loop.",
                regeneration_attempt, self.config.max_regeneration_attempts,
            )
            return GradeHallucinations(
                binary_score=GRADE_GROUNDED,
                severity=SEVERITY_MINOR,
                grounding_score=0.5,
                reasoning=(
                    f"[MAX REGENERATION LIMIT REACHED: {regeneration_attempt}/"
                    f"{self.config.max_regeneration_attempts}]"
                ),
            )

        context = format_docs(documents)
        logger.info(
            "HallucinationChecker.check | attempt=%d | docs=%d | q=%r | answer_len=%d chars",
            regeneration_attempt, len(documents), question[:80], len(generation),
        )

        try:
            grade = cast(
                GradeHallucinations,
                self._chain.invoke({
                    "question":             question,
                    "context":              context,
                    "generation":           generation,
                    "regeneration_attempt": regeneration_attempt,
                }),
            )
        except Exception as exc:
            logger.error(
                "HallucinationChecker: LLM call failed (%s) → safe 'grounded' fallback.", exc,
            )
            return GradeHallucinations(
                binary_score=GRADE_GROUNDED,
                severity=SEVERITY_MINOR,
                grounding_score=0.5,
                reasoning=f"[LLM error fallback] {exc}",
            )

        original_verdict = grade.binary_score

        if grade.severity == SEVERITY_CRITICAL and grade.binary_score == GRADE_GROUNDED:
            logger.warning(
                "HallucinationChecker: severity=critical but binary_score=grounded "
                "→ overriding to 'regenerate'."
            )
            grade = GradeHallucinations(
                binary_score=GRADE_REGENERATE,
                unsupported_claims=grade.unsupported_claims,
                severity=SEVERITY_CRITICAL,
                grounding_score=grade.grounding_score,
                supported_claims=grade.supported_claims,
                reasoning=f"[severity override: critical] " + grade.reasoning,
            )

        elif (
            grade.binary_score == GRADE_GROUNDED
            and grade.grounding_score < self.config.grounding_score_threshold
        ):
            logger.info(
                "HallucinationChecker: binary_score=grounded but "
                "grounding_score=%.2f < threshold=%.2f → overriding to 'regenerate'.",
                grade.grounding_score, self.config.grounding_score_threshold,
            )
            grade = GradeHallucinations(
                binary_score=GRADE_REGENERATE,
                unsupported_claims=grade.unsupported_claims,
                severity=grade.severity or SEVERITY_MAJOR,
                grounding_score=grade.grounding_score,
                supported_claims=grade.supported_claims,
                reasoning=(
                    f"[threshold override: score {grade.grounding_score:.2f} "
                    f"< {self.config.grounding_score_threshold}] "
                    + grade.reasoning
                ),
            )

        logger.info(
            "HallucinationChecker: %s → %s | severity=%s | score=%.2f | unsupported_claims=%d",
            original_verdict.upper(),
            grade.binary_score.upper(),
            grade.severity,
            grade.grounding_score,
            len(grade.unsupported_claims),
        )
        if grade.unsupported_claims:
            logger.debug("Unsupported claims:\n%s", grade.formatted_unsupported_claims)

        return grade

    @staticmethod
    def route(grade: GradeHallucinations) -> str:
        """
        Translate a GradeHallucinations verdict into a LangGraph edge string.

        Warning 4 fix: previously returned "finalize" for the grounded path, but
        edges.py reads state["hallucination_status"] and branches on "grounded" vs
        "regenerate".  The mismatched token meant this method could never be wired
        as a conditional-edge function without an extra translation shim — making it
        effectively dead code.  Both return values now match the vocabulary that
        edges.py and nodes.py use throughout.

        Returns:
            "grounded"   -> answer is fully supported; deliver to user.
            "regenerate" -> hallucination detected; re-run generator.

        Wiring example in edges.py:
            graph.add_conditional_edges(
                "hallucination_checker",
                lambda s: HallucinationCheckerAgent.route(s["hallucination_grade"]),
                {"grounded": "finalize_node", "regenerate": "answer_generator"},
            )
        """
        if grade.binary_score == GRADE_GROUNDED:
            logger.info("HallucinationChecker.route -> 'grounded'.")
            return GRADE_GROUNDED
        logger.info(
            "HallucinationChecker.route -> 'regenerate' (severity=%s).", grade.severity,
        )
        return GRADE_REGENERATE


def _build_llm(config: HallucinationCheckerConfig) -> BaseChatModel:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found.")
    return ChatGroq(
        model=config.model,
        temperature=config.temperature,
        max_tokens=2048,
        api_key=SecretStr(api_key),
        **config.llm_kwargs,
    )


def _regenerate_fallback(
    reason: str,
    unsupported: Optional[List[str]] = None,
) -> GradeHallucinations:
    return GradeHallucinations(
        binary_score=GRADE_REGENERATE,
        unsupported_claims=unsupported or [],
        severity=SEVERITY_MAJOR,
        grounding_score=0.0,
        reasoning=f"[pre-check fallback] {reason}",
    )


def build_hallucination_checker_from_config(
    config_dict: Dict[str, Any],
    llm: Optional[BaseChatModel] = None,
) -> HallucinationCheckerAgent:
    cfg = HallucinationCheckerConfig(
        model=config_dict.get("model", "llama-3.3-70b-versatile"),  # ← was "gpt-4o-mini"
        temperature=float(config_dict.get("temperature", 0.0)),
        max_regeneration_attempts=int(config_dict.get("max_regeneration_attempts", 2)),
        grounding_score_threshold=float(config_dict.get("grounding_score_threshold", 0.85)),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
    )
    return HallucinationCheckerAgent(llm=llm, config=cfg)