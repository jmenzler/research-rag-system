## LazyGraphRAG tool

When the user's question is broad / synthesis-oriented and would benefit
from a community-level summary over the corpus, emit a `lazygraph` payload
containing 1-3 topic phrases that should anchor community lookup.

Format (JSON, in `tool_payloads.lazygraph`):

```json
["high-frequency market microstructure", "optimal market making"]
```

Rules:
- Topic phrases MUST be short noun phrases (2-6 words each) — they will be
  matched against community summaries in the LazyGraphRAG index OR (when
  the index is unavailable) used as Milvus partition-filtered search queries.
- Prefer multiple, complementary phrases over a single broad phrase: the
  retriever routes the question into ONE community per phrase and unions
  the results.
- Use `lazygraph` for "what does the corpus say about X across many papers?"
  style questions. For specific factual lookups, prefer `milvus`. For
  multi-hop reasoning across entities, prefer `hipporag`.
- If no broad / cross-paper synthesis is needed, OMIT the `lazygraph` key
  rather than emitting an empty list.
