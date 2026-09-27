"""Prompt instructions and few-shot examples — lifted verbatim from RAGAS.

The instruction strings MUST match RAGAS exactly (verified by parity tests in
``tests/test_eval_metrics.py::TestRagasParity``). The example tuples are
constructed from our Pydantic models with the same field values RAGAS uses.

Render format (mirrors RAGAS' ``PydanticPrompt.to_string`` shape closely
enough for the LLM to produce identical-shaped output)::

    {instruction}

    Examples:
    Input: {ex_in.model_dump_json()}
    Output: {ex_out.model_dump_json()}
    ...

    Now process this:
    Input: {actual.model_dump_json()}
    Output:

We do not depend on RAGAS' renderer — these constants are the only thing
imported from it (and only at test time, for parity).
"""

from __future__ import annotations

from pydantic import BaseModel

from src.eval.schemas import (
    QAC,
    QCA,
    CitationAccuracyOutput,
    CitationCheckInput,
    CitationCheckResult,
    ConceptDisambiguationInput,
    ConceptDisambiguationOutput,
    ContextRecallClassification,
    ContextRecallClassifications,
    NLIStatementInput,
    NLIStatementOutput,
    ResponseRelevanceInput,
    ResponseRelevanceOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorInput,
    StatementsOutput,
    Verification,
)

# ---------------------------------------------------------------------------
# Instruction strings — verbatim from RAGAS
# ---------------------------------------------------------------------------

# From ragas.metrics._faithfulness.StatementGeneratorPrompt.instruction
STATEMENT_GENERATOR_INSTRUCTION = (
    "Given a question and an answer, analyze the complexity of each sentence "
    "in the answer. Break down each sentence into one or more fully understandable "
    "statements. Ensure that no pronouns are used in any statement. Format the "
    "outputs in JSON."
)

# From ragas.metrics._faithfulness.NLIStatementPrompt.instruction
NLI_STATEMENT_INSTRUCTION = (
    "Your task is to judge the faithfulness of a series of statements based on "
    "a given context. For each statement you must return verdict as 1 if the "
    "statement can be directly inferred based on the context or 0 if the "
    "statement can not be directly inferred based on the context."
)

# From ragas.metrics._answer_relevance.ResponseRelevancePrompt.instruction
RESPONSE_RELEVANCE_INSTRUCTION = (
    """Generate a question for the given answer and Identify if answer is """
    """noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 """
    """if the answer is committal. A noncommittal answer is one that is evasive, """
    """vague, or ambiguous. For example, "I don't know" or "I'm not sure" are """
    """noncommittal answers"""
)

# From ragas.metrics._context_precision.ContextPrecisionPrompt.instruction
CONTEXT_PRECISION_INSTRUCTION = (
    'Given question, answer and context verify if the context was useful in '
    'arriving at the given answer. Give verdict as "1" if useful and "0" if '
    "not with json output."
)

# From ragas.metrics._context_recall.ContextRecallClassificationPrompt.instruction
CONTEXT_RECALL_INSTRUCTION = (
    "Given a context, and an answer, analyze each sentence in the answer and "
    "classify if the sentence can be attributed to the given context or not. "
    "Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason."
)


# ---------------------------------------------------------------------------
# Few-shot examples — lifted from RAGAS' source. Same field values, our schemas.
# ---------------------------------------------------------------------------

STATEMENT_GENERATOR_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        StatementGeneratorInput(
            question="Who was Albert Einstein and what is he best known for?",
            answer=(
                "He was a German-born theoretical physicist, widely acknowledged "
                "to be one of the greatest and most influential physicists of all "
                "time. He was best known for developing the theory of relativity, "
                "he also made important contributions to the development of the "
                "theory of quantum mechanics."
            ),
        ),
        StatementsOutput(
            statements=[
                "Albert Einstein was a German-born theoretical physicist.",
                "Albert Einstein is recognized as one of the greatest and most "
                "influential physicists of all time.",
                "Albert Einstein was best known for developing the theory of relativity.",
                "Albert Einstein also made important contributions to the development "
                "of the theory of quantum mechanics.",
            ]
        ),
    )
]

NLI_STATEMENT_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        NLIStatementInput(
            context=(
                "John is a student at XYZ University. He is pursuing a degree in "
                "Computer Science. He is enrolled in several courses this semester, "
                "including Data Structures, Algorithms, and Database Management. "
                "John is a diligent student and spends a significant amount of time "
                "studying and completing assignments. He often stays late in the "
                "library to work on his projects."
            ),
            statements=[
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
        ),
        NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement="John is majoring in Biology.",
                    reason=(
                        "John's major is explicitly mentioned as Computer Science. "
                        "There is no information suggesting he is majoring in Biology."
                    ),
                    verdict=0,
                ),
                StatementFaithfulnessAnswer(
                    statement="John is taking a course on Artificial Intelligence.",
                    reason=(
                        "The context mentions the courses John is currently enrolled in, "
                        "and Artificial Intelligence is not mentioned. Therefore, it "
                        "cannot be deduced that John is taking a course on AI."
                    ),
                    verdict=0,
                ),
                StatementFaithfulnessAnswer(
                    statement="John is a dedicated student.",
                    reason=(
                        "The context states that he spends a significant amount of "
                        "time studying and completing assignments. Additionally, it "
                        "mentions that he often stays late in the library to work on "
                        "his projects, which implies dedication."
                    ),
                    verdict=1,
                ),
                StatementFaithfulnessAnswer(
                    statement="John has a part-time job.",
                    reason=(
                        "There is no information given in the context about John "
                        "having a part-time job."
                    ),
                    verdict=0,
                ),
            ]
        ),
    ),
    (
        NLIStatementInput(
            context=(
                "Photosynthesis is a process used by plants, algae, and certain "
                "bacteria to convert light energy into chemical energy."
            ),
            statements=["Albert Einstein was a genius."],
        ),
        NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement="Albert Einstein was a genius.",
                    reason="The context and statement are unrelated",
                    verdict=0,
                )
            ]
        ),
    ),
]

RESPONSE_RELEVANCE_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        ResponseRelevanceInput(response="Albert Einstein was born in Germany."),
        ResponseRelevanceOutput(
            question="Where was Albert Einstein born?",
            noncommittal=0,
        ),
    ),
    (
        ResponseRelevanceInput(
            response=(
                "I don't know about the  groundbreaking feature of the smartphone "
                "invented in 2023 as am unaware of information beyond 2022. "
            ),
        ),
        ResponseRelevanceOutput(
            question="What was the groundbreaking feature of the smartphone invented in 2023?",
            noncommittal=1,
        ),
    ),
]

CONTEXT_PRECISION_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        QAC(
            question="What can you tell me about Albert Einstein?",
            context=(
                "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
                "theoretical physicist, widely held to be one of the greatest and "
                "most influential scientists of all time. Best known for developing "
                "the theory of relativity, he also made important contributions to "
                "quantum mechanics, and was thus a central figure in the revolutionary "
                "reshaping of the scientific understanding of nature that modern "
                "physics accomplished in the first decades of the twentieth century. "
                "His mass–energy equivalence formula E = mc2, which arises from "
                "relativity theory, has been called 'the world's most famous equation'. "
                "He received the 1921 Nobel Prize in Physics 'for his services to "
                "theoretical physics, and especially for his discovery of the law of "
                "the photoelectric effect', a pivotal step in the development of "
                "quantum theory. His work is also known for its influence on the "
                "philosophy of science. In a 1999 poll of 130 leading physicists "
                "worldwide by the British journal Physics World, Einstein was ranked "
                "the greatest physicist of all time. His intellectual achievements "
                "and originality have made Einstein synonymous with genius."
            ),
            answer=(
                "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                "physicist, widely held to be one of the greatest and most influential "
                "scientists of all time. He received the 1921 Nobel Prize in Physics "
                "for his services to theoretical physics."
            ),
        ),
        Verification(
            reason=(
                "The provided context was indeed useful in arriving at the given answer. "
                "The context includes key information about Albert Einstein's life and "
                "contributions, which are reflected in the answer."
            ),
            verdict=1,
        ),
    ),
    (
        QAC(
            question="who won 2020 icc world cup?",
            context=(
                "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, "
                "2022, in Australia, was the eighth edition of the tournament. "
                "Originally scheduled for 2020, it was postponed due to the COVID-19 "
                "pandemic. England emerged victorious, defeating Pakistan by five "
                "wickets in the final to clinch their second ICC Men's T20 World Cup title."
            ),
            answer="England",
        ),
        Verification(
            reason=(
                "the context was useful in clarifying the situation regarding the 2020 "
                "ICC World Cup and indicating that England was the winner of the "
                "tournament that was intended to be held in 2020 but actually took "
                "place in 2022."
            ),
            verdict=1,
        ),
    ),
    (
        QAC(
            question="What is the tallest mountain in the world?",
            context=(
                "The Andes is the longest continental mountain range in the world, "
                "located in South America. It stretches across seven countries and "
                "features many of the highest peaks in the Western Hemisphere. The "
                "range is known for its diverse ecosystems, including the high-altitude "
                "Andean Plateau and the Amazon rainforest."
            ),
            answer="Mount Everest.",
        ),
        Verification(
            reason=(
                "the provided context discusses the Andes mountain range, which, while "
                "impressive, does not include Mount Everest or directly relate to the "
                "question about the world's tallest mountain."
            ),
            verdict=0,
        ),
    ),
]

CONTEXT_RECALL_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        QCA(
            question="What can you tell me about albert Albert Einstein?",
            context=(
                "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
                "theoretical physicist, widely held to be one of the greatest and most "
                "influential scientists of all time. Best known for developing the "
                "theory of relativity, he also made important contributions to quantum "
                "mechanics, and was thus a central figure in the revolutionary "
                "reshaping of the scientific understanding of nature that modern "
                "physics accomplished in the first decades of the twentieth century. "
                "His mass-energy equivalence formula E = mc2, which arises from "
                "relativity theory, has been called 'the world's most famous equation'. "
                "He received the 1921 Nobel Prize in Physics 'for his services to "
                "theoretical physics, and especially for his discovery of the law of "
                "the photoelectric effect', a pivotal step in the development of "
                "quantum theory. His work is also known for its influence on the "
                "philosophy of science. In a 1999 poll of 130 leading physicists "
                "worldwide by the British journal Physics World, Einstein was ranked "
                "the greatest physicist of all time. His intellectual achievements "
                "and originality have made Einstein synonymous with genius."
            ),
            answer=(
                "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                "physicist, widely held to be one of the greatest and most influential "
                "scientists of all time. He received the 1921 Nobel Prize in Physics "
                "for his services to theoretical physics. He published 4 papers in "
                "1905. Einstein moved to Switzerland in 1895."
            ),
        ),
        ContextRecallClassifications(
            classifications=[
                ContextRecallClassification(
                    statement=(
                        "Albert Einstein, born on 14 March 1879, was a German-born "
                        "theoretical physicist, widely held to be one of the greatest "
                        "and most influential scientists of all time."
                    ),
                    reason="The date of birth of Einstein is mentioned clearly in the context.",
                    attributed=1,
                ),
                ContextRecallClassification(
                    statement=(
                        "He received the 1921 Nobel Prize in Physics for his services "
                        "to theoretical physics."
                    ),
                    reason="The exact sentence is present in the given context.",
                    attributed=1,
                ),
                ContextRecallClassification(
                    statement="He published 4 papers in 1905.",
                    reason="There is no mention about papers he wrote in the given context.",
                    attributed=0,
                ),
                ContextRecallClassification(
                    statement="Einstein moved to Switzerland in 1895.",
                    reason="There is no supporting evidence for this in the given context.",
                    attributed=0,
                ),
            ]
        ),
    ),
]


# ---------------------------------------------------------------------------
# Aspect: citation accuracy
# ---------------------------------------------------------------------------

CITATION_ACCURACY_INSTRUCTION = (
    "You are an expert evaluator of RAG system outputs. Your task is to check "
    "whether each citation in a synthesized answer correctly attributes its "
    "surrounding claim to the cited source. For each citation marker [N] in the "
    "answer, identify the claim it supports, then verify whether that claim "
    "appears in the cited source's context text. Return 1 if the context "
    "supports the claim, 0 if it does not. Output JSON only."
)

CITATION_ACCURACY_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        CitationCheckInput(
            question="What is the reservation price in the Avellaneda-Stoikov model?",
            answer=(
                "In the Avellaneda-Stoikov model, the reservation price is "
                "r = s - q*gamma*sigma^2*(T-t) [1]. This is derived from the "
                "Hamilton-Jacobi-Bellman equation. The model assumes the mid-price "
                "follows arithmetic Brownian motion [2]."
            ),
            citations_formatted=(
                "[1] avellaneda_stoikov_2008.pdf:5\n"
                "[2] avellaneda_stoikov_2008.pdf:2"
            ),
            contexts=(
                "Context for avellaneda_stoikov_2008.pdf page 5:\n"
                "The reservation price (indifference price) is given by "
                "r(s, q, t) = s - q*gamma*sigma^2*(T-t), where s is the mid-price, "
                "q is the inventory, gamma is the risk-aversion parameter, sigma is "
                "volatility, and T-t is the remaining time.\n\n"
                "Context for avellaneda_stoikov_2008.pdf page 2:\n"
                "We consider a market maker who continuously posts bid and ask prices. "
                "The mid-price S_t is assumed to follow an arithmetic Brownian motion."
            ),
        ),
        CitationAccuracyOutput(
            checks=[
                CitationCheckResult(
                    citation_index=1,
                    claim="r = s - q*gamma*sigma^2*(T-t)",
                    verdict=1,
                    reason="The formula appears verbatim in the context for page 5.",
                ),
                CitationCheckResult(
                    citation_index=2,
                    claim="The model assumes arithmetic Brownian motion for the mid-price",
                    verdict=1,
                    reason="Context for page 2 confirms arithmetic Brownian motion mid-price.",
                ),
            ]
        ),
    ),
    (
        CitationCheckInput(
            question="How does the Glosten-Milgrom model differ from Avellaneda-Stoikov?",
            answer=(
                "Glosten-Milgrom models informed order flow and adverse selection [1], "
                "while Avellaneda-Stoikov focuses on inventory risk [2]. "
                "Both models assume continuous-time limit order books [3]."
            ),
            citations_formatted=(
                "[1] glosten_milgrom_1985.pdf:3\n"
                "[2] avellaneda_stoikov_2008.pdf:1\n"
                "[3] avellaneda_stoikov_2008.pdf:7"
            ),
            contexts=(
                "Context for glosten_milgrom_1985.pdf page 3:\n"
                "The model captures the information asymmetry between market makers "
                "and informed traders, leading to a bid-ask spread that compensates "
                "for adverse selection risk.\n\n"
                "Context for avellaneda_stoikov_2008.pdf page 1:\n"
                "The Avellaneda-Stoikov model addresses optimal market making under "
                "inventory risk with a risk-averse agent.\n\n"
                "Context for avellaneda_stoikov_2008.pdf page 7:\n"
                "The limit order arrival rates are assumed to be Poisson processes."
            ),
        ),
        CitationAccuracyOutput(
            checks=[
                CitationCheckResult(
                    citation_index=1,
                    claim="Glosten-Milgrom models informed order flow and adverse selection",
                    verdict=1,
                    reason="The context for glosten_milgrom page 3 confirms this.",
                ),
                CitationCheckResult(
                    citation_index=2,
                    claim="Avellaneda-Stoikov focuses on inventory risk",
                    verdict=1,
                    reason="The context for AS page 1 confirms inventory risk focus.",
                ),
                CitationCheckResult(
                    citation_index=3,
                    claim="Both models assume continuous-time limit order books",
                    verdict=0,
                    reason="The cited context (AS page 7) discusses Poisson arrival rates, "
                    "not that both models assume continuous-time LOBs. The GM model uses "
                    "discrete-time sequential trade.",
                ),
            ]
        ),
    ),
]


# ---------------------------------------------------------------------------
# Aspect: concept disambiguation
# ---------------------------------------------------------------------------

CONCEPT_DISAMBIGUATION_INSTRUCTION = (
    "You are an expert evaluator of RAG system outputs for financial and "
    "market-microstructure domains. Your task is to judge whether the "
    "synthesized answer correctly distinguishes closely-related concepts "
    "when the question touches on multiple related ideas.\n\n"
    "Closely-related concept pairs common in this domain include:\n"
    "- Avellaneda-Stoikov (inventory-based market making) vs Glosten-Milgrom "
    "(adverse-selection-based market making)\n"
    "- GLFT (Grossman-Miller-style competitive equilibrium) vs AS "
    "(single-agent stochastic control)\n"
    "- LVR (loss-versus-rebalancing) vs PnL (profit-and-loss) — LVR is a "
    "subset of total PnL decomposition, not equivalent\n"
    "- VPIN (volume-synchronized probability of informed trading) vs PIN "
    "(probability of informed trading) — VPIN uses volume clock, PIN uses "
    "calendar clock\n"
    "- TWAP vs VWAP — TWAP splits by time, VWAP by volume profile\n\n"
    "Score 1 if the answer clearly distinguishes the relevant concepts when "
    "appropriate. Score 0 if the answer conflates distinct concepts, uses them "
    "interchangeably when they differ, or fails to clarify a distinction the "
    "question implicitly demands. Output JSON only."
)

CONCEPT_DISAMBIGUATION_EXAMPLES: list[tuple[BaseModel, BaseModel]] = [
    (
        ConceptDisambiguationInput(
            question=(
                "What is the difference between the Avellaneda-Stoikov model "
                "and the Glosten-Milgrom model?"
            ),
            answer=(
                "Avellaneda-Stoikov models a risk-averse market maker managing "
                "inventory through stochastic control, deriving optimal bid-ask "
                "quotes around a reservation price. The key friction is inventory "
                "risk. In contrast, Glosten-Milgrom models a risk-neutral market "
                "maker facing adverse selection from informed traders. The key "
                "friction is information asymmetry, and the spread compensates "
                "for expected losses to informed counterparties. AS uses "
                "continuous-time diffusion, while GM uses discrete-time "
                "sequential trade with Bayesian updating."
            ),
            contexts=(
                "Avellaneda & Stoikov (2008) derive optimal bid/ask quotes for "
                "a risk-averse market maker. Glosten & Milgrom (1985) model "
                "adverse selection in dealer markets."
            ),
        ),
        ConceptDisambiguationOutput(
            verdict=1,
            reason=(
                "The answer clearly distinguishes the two models: AS = inventory "
                "risk + stochastic control + continuous-time; GM = adverse selection "
                "+ risk-neutral + discrete-time. Each model's unique mechanism is "
                "correctly identified."
            ),
        ),
    ),
    (
        ConceptDisambiguationInput(
            question="Explain LVR and how it affects market making profitability.",
            answer=(
                "LVR, or loss-versus-rebalancing, measures the adverse selection "
                "cost a liquidity provider incurs relative to a rebalancing strategy. "
                "Note that LVR is distinct from total PnL — it captures only the "
                "adverse-selection component of the PnL decomposition. The remaining "
                "components are the spread capture and inventory drift. A market "
                "maker can have positive total PnL while still having positive LVR."
            ),
            contexts=(
                "Milionis et al. (2023) define LVR as the expected shortfall of "
                "a liquidity provision strategy relative to continuous rebalancing."
            ),
        ),
        ConceptDisambiguationOutput(
            verdict=1,
            reason=(
                "The answer correctly distinguishes LVR (adverse selection component) "
                "from total PnL and lists the other components of PnL decomposition."
            ),
        ),
    ),
    (
        ConceptDisambiguationInput(
            question="Compare GLFT and Avellaneda-Stoikov market making.",
            answer=(
                "Both GLFT and Avellaneda-Stoikov are market making models that "
                "compute optimal quotes. They both assume the market maker posts "
                "bid and ask prices around the mid-price."
            ),
            contexts=(
                "GLFT models competitive equilibrium where market makers earn zero "
                "profit. Avellaneda-Stoikov models a single agent maximizing "
                "expected utility."
            ),
        ),
        ConceptDisambiguationOutput(
            verdict=0,
            reason=(
                "The answer describes only superficial similarities (posting quotes "
                "around mid-price) without distinguishing the fundamental difference: "
                "GLFT is a competitive-equilibrium model (many agents, zero profit) "
                "while AS is a single-agent stochastic-control model (monopoly, "
                "positive expected profit)."
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_prompt(
    instruction: str,
    examples: list[tuple[BaseModel, BaseModel]],
    actual_input: BaseModel,
) -> str:
    """Render an instruction + few-shot examples + actual input into a string.

    Format (deterministic, ASCII-safe)::

        {instruction}

        Examples:
        Input: {ex_in.model_dump_json()}
        Output: {ex_out.model_dump_json()}
        ...

        Now process this:
        Input: {actual.model_dump_json()}
        Output:

    The model is expected to produce a JSON object matching the corresponding
    output schema. With provider-side schema enforcement on, the output JSON
    is guaranteed to be parseable by ``schema.model_validate_json``.
    """
    parts: list[str] = [instruction, ""]
    if examples:
        parts.append("Examples:")
        for ex_in, ex_out in examples:
            parts.append(f"Input: {ex_in.model_dump_json()}")
            parts.append(f"Output: {ex_out.model_dump_json()}")
        parts.append("")
    parts.append("Now process this:")
    parts.append(f"Input: {actual_input.model_dump_json()}")
    parts.append("Output:")
    return "\n".join(parts)
