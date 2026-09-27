SELF-GATE (emission rules):

The system has activated one or more optional output fields via the sections
above. Follow these rules to control what you emit:

1. For each optional field whose instructions appear in a section above
   (HyDE, Stepback, Disambiguation, Filters): emit that field with non-empty
   content per its section's spec.

2. For any optional field whose instructions are NOT in the prompt above
   (no section provided): set the field to null (or [] for list-typed fields).

3. Always include all 4 optional fields in your output JSON, even when null/[]:
   hyde_doc, stepback, disambiguation, filters.

The activation set is determined globally by the run's RewriterConfig
(REWRITER_EMIT_* env vars), NOT per query. If a section is in this prompt,
emit the corresponding field; if absent, null it.
