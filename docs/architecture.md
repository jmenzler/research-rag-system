# Architecture

The application is organized around a source checkout rather than a standalone Python wheel. Runtime prompts, command entry points, and selected scripts live alongside `src/`.

| Area | Responsibility |
| --- | --- |
| `src/fetch/`, `src/pdf_parsers/`, `src/cleanup/` | Fetch documents, extract content, and reject or quarantine unsuitable inputs. |
| `src/pipeline/`, `bin/ragctl` | Plan and run ingestion stages, track progress, and supervise parser processes. |
| `src/ingest/`, `prompts/contextualize/` | Split and contextualize content, produce embeddings, and populate retrieval storage. |
| `src/query/`, `bin/rag` | Rewrite queries, retrieve candidates, rerank evidence, and optionally synthesize an answer. |
| `src/citations/` | Import and traverse externally supplied citation and paper metadata. |
| `src/server/` | HTTP routes, persistent work queues, chat storage, graph views, streaming, and cancellation. |
| `src/mcp_server.py` | Forward MCP tool requests to the HTTP API. |
| `frontend/src/` | React chat, corpus, query-history, and graph interfaces. |

## Data flow

Documents enter a staged ingestion pipeline. Parsed content is split into parent and child chunks. Child embeddings and retrieval fields are stored in Milvus; local SQLite data retains parent text and application state. Contextualization and embedding can call configured providers.

A query selects a collection and optional notebook scope. Retrieval can combine dense and lexical evidence, rewriting, and additional retriever implementations. Candidate evidence is reranked before returning source chunks or passing them to an LLM for synthesis. Audit records capture the actual path taken; optional adapters are not equivalent to independently verified retrieval systems.

Citation exploration uses separate metadata and graph stores. Those datasets are not included, and an empty source checkout cannot reproduce a populated graph.

The browser uses FastAPI endpoints. The MCP process is a client of the same service, not a separate corpus owner. SQLite migrations run at API startup. The server can serve a built frontend bundle, while local development uses Vite's proxy.

## External boundaries

Milvus, embedding/generation providers, reranking services or model weights, citation datasets, and optional PDF tooling are runtime dependencies. Default tests replace these boundaries with synthetic fixtures and mocks. Optional retriever dependencies and GPU-daemon integration are not required for the default test suite.

Collection names in this snapshot are generic examples, not evidence of an included corpus. Static frontend fixtures are test/development material, not research results or a record of real usage.
