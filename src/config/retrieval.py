"""Retrieval, reranking, and query-pipeline feature flags."""
from __future__ import annotations

RETRIEVE_TOP_K: int = 100
RERANK_TOP_K: int = 20
RERANK_SCORE_THRESHOLD: float = 0.20
# Backend-calibrated drop threshold. Score scales differ per reranker model, so
# one fixed value is wrong when the backend changes. The NVIDIA NIM
# (rerank-qa-mistral-4b) is bimodal — ~0.99 confident, 0.02-0.18 for valid-but-
# subtle matches — so the Qwen3-calibrated 0.20 dropped every valid hit. The
# resolver in src/query/policies.py picks this for remote_nvidia.
RERANK_SCORE_THRESHOLD_NVIDIA: float = 0.01
# Floor: when the threshold drops every candidate, keep this many top hits so a
# valid query never returns empty (low reranker scores / dead fallback). 0=off.
RERANK_MIN_KEEP: int = 5
MAX_CHILDREN_PER_PARENT: int = 2
DECOMPOSE_MAX_SUBQUERIES: int = 4

USE_STEPBACK: bool = False
USE_CRAG_LITE: bool = True
CRAG_THRESHOLD: float = 0.7

# MMR diversity filter (post-rerank). When USE_MMR=True, an MMR_POOL_MULT× pool
# is collected from the reranker, then MMR picks MMR_TOP_K from it by trading
# relevance against intra-set diversity using cosine similarity over Milvus
# dense_embedding vectors. Lambda=1.0 → pure relevance (same as no MMR);
# 0.0 → pure diversity. 0.6 favors relevance with a meaningful diversity push.
# Validated 2026-05-08 against V4-Flash judge on N=10: F=0.97 (+0.02),
# CR=0.51 (+0.10) — MMR closes ~1/3 of the recall gap to the 0.70 target with
# no precision regression.
USE_MMR: bool = True
MMR_LAMBDA: float = 0.6
MMR_TOP_K: int = 20
MMR_POOL_MULT: int = 3
