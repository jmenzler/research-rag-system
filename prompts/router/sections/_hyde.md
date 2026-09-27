HYPOTHETICAL DOCUMENT EMBEDDING (HyDE):

Emit a `hyde_doc` field: write one paragraph that a real answering chunk would contain.

Rules:
- Use specific technical vocabulary the answering chunk would use (model names, formulas, paper authors).
- The paragraph MAY be factually incorrect — semantic shape is what matters, not factual accuracy.
- No hedging, no disclaimers, no "as an AI" or "I think". Write as if it is a real chunk.
- Length: aim for the `hyde_max_tokens` token budget (default 80 tokens).
- If you cannot generate a meaningful hypothetical, set `hyde_doc: null`.

Example for "What is the Avellaneda-Stoikov reservation price?":
hyde_doc: "The reservation price in the Avellaneda-Stoikov model is given by r = s - q·γ·σ²·(T−t), where s is the mid-price, q is the current inventory, γ is the risk-aversion coefficient, σ² is the variance of the asset price, and (T−t) is the remaining time horizon. This formula captures the dealer's inventory risk: positive inventory lowers the reservation price to incentivize selling."

Per-intent guidance:
- multi_hop / comparative: focus on the bridging relation between entities.
- computational: include the formula structure, parameter names, and units.
- literature_synthesis: focus on the synthesis question's central claim, not one paper.
