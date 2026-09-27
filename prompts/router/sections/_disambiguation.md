DISAMBIGUATION (for `unknown` intent only):

Emit a `disambiguation` field: 1-3 candidate interpretations of the query.

Format: array of objects, each with:
- `label`: short label for the interpretation (≤10 words)
- `clarified_query`: the query rewritten for this interpretation (≤25 words)

Rules:
- Only populate when the query is genuinely ambiguous (unknown intent).
- Empty array `[]` when the query is unambiguous.
- Do NOT invent disambiguation for clear queries just to fill the field.

Example:
User: "Tell me about spreads"
disambiguation: [
  {"label": "bid-ask spread in market making", "clarified_query": "What is the bid-ask spread in limit order book market making?"},
  {"label": "credit spread in fixed income", "clarified_query": "What is the credit spread between corporate and government bonds?"},
  {"label": "option spread strategy", "clarified_query": "What are spread option strategies such as bull spread or calendar spread?"}
]

Per Tree of Clarifications (Kim et al. 2023): surface the most likely interpretations, not exhaustive enumeration.
