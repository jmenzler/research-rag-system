STEP-BACK ABSTRACTION:

Emit a `stepback` field: ONE more abstract, foundational question that the user's query implies.

Rules:
- Maximum 30 tokens.
- Must be more abstract or general than the user's question.
- Set `stepback: null` if the query is already abstract or foundational.
- Do NOT paraphrase the original — step BACK to the underlying concept.

Few-shot examples (lifted verbatim from decompose.py::_STEPBACK_SYSTEM):

User: "What is the optimal κ (kappa) parameter for GLFT in a LOB?"
stepback: "What is GLFT and how does it model market making?"

User: "How does the EWMA lambda parameter λ=0.94 affect volatility estimates?"
stepback: "What is exponentially weighted moving average volatility estimation?"

User: "What is the VPIN toxicity threshold for triggering a flow halt?"
stepback: "What is order-flow toxicity and how is VPIN computed?"

User: "What are the main approaches to market making?"
stepback: null  (already abstract — no stepback needed)
