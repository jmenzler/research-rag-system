"""Pydantic models for the four metrics' LLM output shapes.

These models are lifted verbatim from RAGAS' source files, with names tweaked
to match our convention (``StatementsOutput`` vs RAGAS'
``StatementGeneratorOutput``). The field names match exactly so that prompt
output JSON is interchangeable.

Each model is used in two places:
1. As the ``schema=`` argument to ``call_text()`` (provider-side enforcement)
2. As the type to ``model_validate_json()`` for parsing the response

Both happen via the ``call_with_schema`` helper in ``src.eval.metrics``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Faithfulness — two LLM calls
# ---------------------------------------------------------------------------


class StatementGeneratorInput(BaseModel):
    """Input for the statement-extraction call."""

    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementsOutput(BaseModel):
    """List of atomic statements lifted from the answer.

    Equivalent to RAGAS' ``StatementGeneratorOutput``.
    """

    statements: list[str] = Field(description="The generated statements")


class NLIStatementInput(BaseModel):
    """Input for the NLI verdict call: context + the statements to judge."""

    context: str = Field(description="The context of the question")
    statements: list[str] = Field(description="The statements to judge")


class StatementFaithfulnessAnswer(BaseModel):
    """One verdict for one statement."""

    statement: str = Field(description="the original statement, word-by-word")
    reason: str = Field(description="the reason of the verdict")
    verdict: int = Field(description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    """List of per-statement verdicts."""

    statements: list[StatementFaithfulnessAnswer]


# ---------------------------------------------------------------------------
# Answer relevancy — N=3 question-generation calls
# ---------------------------------------------------------------------------


class ResponseRelevanceInput(BaseModel):
    """Input for the question-generation call: just the answer."""

    response: str


class ResponseRelevanceOutput(BaseModel):
    """Generated question + noncommittal flag."""

    question: str
    noncommittal: int


# ---------------------------------------------------------------------------
# Context precision — K verification calls (one per retrieved chunk)
# ---------------------------------------------------------------------------


class QAC(BaseModel):
    """Question / Answer / Context bundle for the precision verifier."""

    question: str = Field(description="Question")
    context: str = Field(description="Context")
    answer: str = Field(description="Answer")


class Verification(BaseModel):
    """Verifier output: useful/not + reason."""

    reason: str = Field(description="Reason for verification")
    verdict: int = Field(description="Binary (0/1) verdict of verification")


# ---------------------------------------------------------------------------
# Context recall — single classification call
# ---------------------------------------------------------------------------


class QCA(BaseModel):
    """Question / Context / Answer bundle for the recall classifier."""

    question: str
    context: str
    answer: str


class ContextRecallClassification(BaseModel):
    """One classification: is this statement attributable to the context?"""

    statement: str
    reason: str
    attributed: int


class ContextRecallClassifications(BaseModel):
    """Wrapper holding the list of classifications."""

    classifications: list[ContextRecallClassification]


# ---------------------------------------------------------------------------
# Aspect critics — citation accuracy + concept disambiguation
# ---------------------------------------------------------------------------


class CitationCheckInput(BaseModel):
    """Input for the citation accuracy judge: answer + citations + contexts."""

    question: str = Field(description="The original question")
    answer: str = Field(description="The synthesized answer with [N] citation markers")
    citations_formatted: str = Field(
        description="List of citation references in '[N] source_file:page' format"
    )
    contexts: str = Field(description="All retrieved context texts, joined with newlines")


class CitationCheckResult(BaseModel):
    """One citation check: does the cited source support the claim?"""

    citation_index: int = Field(description="The [N] index of the citation being checked")
    claim: str = Field(description="The specific claim that the citation is supposed to support")
    verdict: int = Field(
        description="1 if the cited context supports the claim, 0 otherwise"
    )
    reason: str = Field(description="Brief explanation of the verdict")


class CitationAccuracyOutput(BaseModel):
    """List of per-citation checks."""

    checks: list[CitationCheckResult]


class ConceptDisambiguationInput(BaseModel):
    """Input for the concept disambiguation judge."""

    question: str = Field(description="The original question")
    answer: str = Field(description="The synthesized answer")
    contexts: str = Field(description="All retrieved context texts, joined with newlines")


class ConceptDisambiguationOutput(BaseModel):
    """Binary verdict: does the answer correctly distinguish closely-related concepts?"""

    verdict: int = Field(
        description="1 if concepts are correctly distinguished, 0 if conflated or ambiguous"
    )
    reason: str = Field(description="Brief explanation of the verdict")
