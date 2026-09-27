# Security and data boundaries

This snapshot is for code inspection and local development. It has no application authentication boundary suitable for public exposure. Keep HTTP, MCP network transports, databases, and model services on loopback or behind access controls you provide. Changing a bind address does not add authentication or TLS.

The application can fetch URLs, launch local parsing processes, write files, ingest documents, and call paid external providers. Only process trusted inputs in an isolated environment with limited filesystem and network access. Parser and browser dependencies are not a sandbox for hostile documents.

Queries and retrieved text can be sent to configured providers. Chat records, source content, URLs, retrieval traces, and provider outputs can be persisted locally. Review provider terms and retention rules before processing confidential material. Provider cost figures in code are estimates, not billing guarantees.

Runtime files belong outside version control: `.env` files, source documents, model weights, SQLite/Milvus/graph databases, caches, chat/query logs, browser state, and generated research outputs. This repository includes none of the original application's corpus or deployment credentials. The PDF fixture is an original one-page synthetic document; frontend fixtures and mocked test inputs are not real operational records.

Dependency installation and passing tests are not a security audit. Optional adapters, native parsers, model downloads, and live provider paths require their own review. Do not assume an optional integration is safe or supported merely because its source is present.

This export has independent history. Updates are manually reviewed; there is no automatic synchronization from a private repository.
