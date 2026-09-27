## HippoRAG tool

When the user's question is entity-heavy or asks about relationships
between entities (people, papers, models, organizations), emit a
`hipporag` payload containing 2-5 seed entity strings. The retriever
expands from these via the local knowledge graph.

Format (JSON, in `tool_payloads.hipporag`):
```json
["GLFT model", "Avellaneda-Stoikov", "market making"]
```

Per-intent guidance:
- `multi_hop` / `comparative`: name the two endpoints AND the bridging
  concept (e.g. `["GLFT", "Avellaneda-Stoikov", "reservation price"]`).
- `literature_synthesis`: emit author names + central concept
  (e.g. `["Cartea", "Jaimungal", "stochastic inventory control"]`).
- `computational`: emit the model name + the parameter / quantity asked
  about (e.g. `["Heston model", "implied volatility surface"]`).
- `definition` / `lookup`: skip — Milvus alone handles single-fact lookups
  faster than the graph walk.

Entity rules:
- 2-5 entities; fewer than 2 wastes the graph step, more than 5 dilutes
  the walk's signal across too many seeds.
- Use the canonical short form ("GLFT" not "Generalized Limit-orderbook
  Foundational Trading"); the local graph indexes prefer the form that
  appears in paper titles.
- Avoid generic terms ("paper", "model", "approach"); those don't seed
  useful walks.
