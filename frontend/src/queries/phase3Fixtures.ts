// Synthetic UI fixtures; citations and usage figures are not research evidence.

export type Collection = "trading" | "ecology" | "notes" | "system";
export type Retriever =
  | "milvus"
  | "paperqa"
  | "lightrag"
  | "hipporag"
  | "lazygraph"
  | "graphrag"
  | "fused";
export type Outcome = "ok" | "retry" | "failed" | "skipped";

export interface Paper {
  id: string;
  title: string;
  authors: string;
  year: number;
  collection: Collection;
  chunks: number;
  arxiv: string | null;
  doi: string | null;
  ingested: string;
  shortCite: string;
  /** When true, Milvus is unreachable → chunk count renders as a degraded "—". */
  degradedChunks?: boolean;
}

export interface QueryLogRow {
  id: string;
  ts: string;
  query: string;
  retriever: Retriever;
  model: string;
  tokens: number;
  cost: number;
  latencyMs: number;
  outcome: Outcome;
  chat: string | null;
  turn: number | null;
}

export interface ProvenanceEntry {
  k: string;
  /** Plain value; rich rendering (degraded, coltag) is decided in the view. */
  v: string | null;
  mono?: boolean;
  /** Set when the value is unavailable from the 3-substrate join. */
  degradedReason?: string;
  kind?: "coltag" | "extraction-quality" | "chunks";
}

export interface QueryForPaper {
  id: string;
  query: string;
  retriever: Retriever;
  cost: number;
  ts: string;
}

export interface ChatForPaper {
  id: string;
  title: string;
  turn: number;
}

export interface PaperDetail extends Paper {
  provenance: ProvenanceEntry[];
  queriesForPaper: QueryForPaper[];
  chatsForPaper: ChatForPaper[];
  extractionQuality: "high" | "medium" | "low";
  parentChunks: number;
  fileHash: string;
}

export type StageState = "run" | "skip" | "fail";

export interface QueryStage {
  n: string;
  name: string;
  state: StageState;
  model?: string;
  tokens?: number;
  cost?: number;
  ms?: number;
  candidates?: number;
  in?: number;
  out?: number;
  note?: string;
  desc: string;
  payload?: Record<string, unknown>;
}

export interface RetrievedChunk {
  f: string;
  cite: string;
  page: number;
  score: number;
  paperId: string;
}

export interface ReportBlock {
  heading: string;
  /** Paragraph parts: plain text segments interleaved with citation markers. */
  parts: Array<{ text: string } | { cite: number; resolved: boolean }>;
}

export interface QueryDetail {
  row: QueryLogRow;
  reportPath: string;
  report: ReportBlock[];
  stages: QueryStage[];
  chunks: RetrievedChunk[];
  kpi: {
    tokensSub: string;
    costSub: string;
    latencySub: string;
  };
}

export interface PaperListResult {
  papers: Paper[];
  totalLabel: string;
  nextCursor: string | null;
}

export interface QueryListResult {
  queries: QueryLogRow[];
  totalLabel: string;
  nextCursor: string | null;
}

const SHORT_CITES: Record<string, string> = {
  "2204.00817": "Avellaneda & Stoikov (2008)",
  "1105.3115": "Guéant et al. (2013)",
  "2208.06046": "Milionis et al. (2022)",
  "doi-rfs053": "Easley et al. (2012)",
  "1011.6402": "Cont, Kukanov, Stoikov (2014)",
  "1412.4839": "Curato, Gatheral, Lillo (2016)",
  "2310.04321": "Mehrotra et al. (2024)",
  "2402.07181": "Yu et al. (2024)",
  "2401.08967": "Edge et al. (2024)",
  "doi-jmlr12": "Andoni & Indyk (2008)",
  "2305.13245": "Malkov et al. (2023)",
  "1306.5036": "Foucault et al. (2013)",
  "2107.10000": "Bacry et al. (2021)",
  "notes-1": "Example author (2026)",
};

export const PAPERS: Paper[] = [
  { id: "2204.00817", title: "High-frequency trading in a limit order book", authors: "Avellaneda, Stoikov", year: 2008, collection: "trading", chunks: 142, arxiv: "2204.00817", doi: null, ingested: "2026-05-12", shortCite: SHORT_CITES["2204.00817"]! },
  { id: "1105.3115", title: "Dealing with the inventory risk: a solution to the market making problem", authors: "Guéant, Lehalle, Fernandez-Tapia", year: 2013, collection: "trading", chunks: 98, arxiv: "1105.3115", doi: null, ingested: "2026-05-12", shortCite: SHORT_CITES["1105.3115"]! },
  { id: "2208.06046", title: "Automated market making and loss-versus-rebalancing", authors: "Milionis, Moallemi, Roughgarden, Zhang", year: 2022, collection: "trading", chunks: 76, arxiv: "2208.06046", doi: null, ingested: "2026-05-13", shortCite: SHORT_CITES["2208.06046"]! },
  { id: "doi-rfs053", title: "Flow toxicity and liquidity in a high-frequency world (VPIN)", authors: "Easley, López de Prado, O'Hara", year: 2012, collection: "trading", chunks: 61, arxiv: null, doi: "10.1093/rfs/hhs053", ingested: "2026-05-14", shortCite: SHORT_CITES["doi-rfs053"]! },
  { id: "1011.6402", title: "Order flow imbalance and the price formation process", authors: "Cont, Kukanov, Stoikov", year: 2014, collection: "trading", chunks: 54, arxiv: "1011.6402", doi: null, ingested: "2026-05-15", shortCite: SHORT_CITES["1011.6402"]! },
  { id: "1412.4839", title: "Optimal execution with nonlinear transient market impact", authors: "Curato, Gatheral, Lillo", year: 2016, collection: "trading", chunks: 38, arxiv: "1412.4839", doi: null, ingested: "2026-05-16", shortCite: SHORT_CITES["1412.4839"]! },
  { id: "2310.04321", title: "Reranker fusion: empirical bounds on CRAG retry", authors: "Mehrotra, Lin, Khatib", year: 2024, collection: "ecology", chunks: 44, arxiv: "2310.04321", doi: null, ingested: "2026-05-17", shortCite: SHORT_CITES["2310.04321"]! },
  { id: "2402.07181", title: "Dense retrieval for scientific corpora: a survey", authors: "Yu, Reichart, Tatariya, Glavaš", year: 2024, collection: "ecology", chunks: 71, arxiv: "2402.07181", doi: null, ingested: "2026-05-17", shortCite: SHORT_CITES["2402.07181"]! },
  { id: "2401.08967", title: "GraphRAG: hierarchical community summaries for global QA", authors: "Edge, Trinh, Cheng, Bradley, Larson", year: 2024, collection: "ecology", chunks: 113, arxiv: "2401.08967", doi: null, ingested: "2026-05-18", shortCite: SHORT_CITES["2401.08967"]! },
  { id: "doi-jmlr12", title: "Locality-sensitive hashing and approximate nearest neighbors", authors: "Andoni, Indyk", year: 2008, collection: "ecology", chunks: 22, arxiv: null, doi: "10.1145/1327452.1327494", ingested: "2026-05-18", shortCite: SHORT_CITES["doi-jmlr12"]!, degradedChunks: true },
  { id: "2305.13245", title: "HNSW revisited: dimensional scaling beyond M=64", authors: "Malkov, Yashunin, Gao", year: 2023, collection: "ecology", chunks: 31, arxiv: "2305.13245", doi: null, ingested: "2026-05-18", shortCite: SHORT_CITES["2305.13245"]! },
  { id: "1306.5036", title: "Kyle's lambda and adverse selection in continuous markets", authors: "Foucault, Pagano, Röell", year: 2013, collection: "trading", chunks: 47, arxiv: "1306.5036", doi: null, ingested: "2026-05-19", shortCite: SHORT_CITES["1306.5036"]! },
  { id: "2107.10000", title: "Hawkes processes for limit-order arrivals at sub-second resolution", authors: "Bacry, Mastromatteo, Muzy", year: 2021, collection: "trading", chunks: 56, arxiv: "2107.10000", doi: null, ingested: "2026-05-19", shortCite: SHORT_CITES["2107.10000"]! },
  { id: "notes-1", title: "Synthetic literature review notes", authors: "Example author", year: 2026, collection: "notes", chunks: 4, arxiv: null, doi: null, ingested: "2026-05-20", shortCite: SHORT_CITES["notes-1"]! },
];

export const QUERIES: QueryLogRow[] = [
  { id: "20260519T211408Z_a1b2c3d4", ts: "2026-05-19 21:14:08", query: "GLFT vs Avellaneda-Stoikov spread derivation", retriever: "milvus", model: "gemini-3.1-flash", tokens: 18204, cost: 0.011, latencyMs: 3214, outcome: "ok", chat: "c01", turn: 3 },
  { id: "20260519T180214Z_b2c3d4e5", ts: "2026-05-19 18:02:14", query: "Reservation price vs mid under inventory skew", retriever: "milvus", model: "gemini-3.1-flash", tokens: 12880, cost: 0.008, latencyMs: 2611, outcome: "ok", chat: "c01", turn: 4 },
  { id: "20260518T114703Z_c3d4e5f6", ts: "2026-05-18 11:47:03", query: "Optimal quoting with terminal inventory penalty", retriever: "fused", model: "gemini-3.1-pro", tokens: 41330, cost: 0.031, latencyMs: 8112, outcome: "retry", chat: "c02", turn: 7 },
  { id: "20260518T092011Z_d4e5f6g7", ts: "2026-05-18 09:20:11", query: "Inventory-risk reservation price intuition", retriever: "paperqa", model: "gemini-3.1-pro", tokens: 33012, cost: 0.024, latencyMs: 6402, outcome: "ok", chat: "c02", turn: 2 },
  { id: "20260517T225502Z_e5f6g7h8", ts: "2026-05-17 22:55:02", query: "Avellaneda closed-form vs numerical HJB solution", retriever: "milvus", model: "gemini-3.1-flash", tokens: 9140, cost: 0.006, latencyMs: 2114, outcome: "failed", chat: null, turn: null },
  { id: "20260517T143309Z_f6g7h8i9", ts: "2026-05-17 14:33:09", query: "LVR pricing without informed-flow assumption", retriever: "lightrag", model: "gemini-3.1-flash", tokens: 22408, cost: 0.014, latencyMs: 4502, outcome: "ok", chat: "c03", turn: 5 },
  { id: "20260517T091802Z_g7h8i9j0", ts: "2026-05-17 09:18:02", query: "Hawkes intensity calibration under sparse arrivals", retriever: "milvus", model: "gemini-3.1-flash", tokens: 11200, cost: 0.007, latencyMs: 2840, outcome: "ok", chat: "c11", turn: 1 },
  { id: "20260516T230011Z_h8i9j0k1", ts: "2026-05-16 23:00:11", query: "GraphRAG global-query communities depth limit", retriever: "graphrag", model: "gemini-3.1-pro", tokens: 54002, cost: 0.041, latencyMs: 9818, outcome: "skipped", chat: "c06", turn: 3 },
  { id: "20260516T184405Z_i9j0k1l2", ts: "2026-05-16 18:44:05", query: "HNSW efSearch tradeoff for 1024-dim qwen embeddings", retriever: "milvus", model: "gemini-3.1-flash", tokens: 8004, cost: 0.005, latencyMs: 1980, outcome: "ok", chat: "c08", turn: 2 },
  { id: "20260516T112201Z_j0k1l2m3", ts: "2026-05-16 11:22:01", query: "Reranker pool sizing — Qwen3-reranker vs bge", retriever: "fused", model: "gemini-3.1-pro", tokens: 28110, cost: 0.022, latencyMs: 5912, outcome: "ok", chat: "c06", turn: 1 },
  { id: "20260515T215008Z_k1l2m3n4", ts: "2026-05-15 21:50:08", query: "OFI windowing — Cont-Kukanov vs minute-bar baseline", retriever: "paperqa", model: "gemini-3.1-flash", tokens: 14550, cost: 0.010, latencyMs: 3320, outcome: "ok", chat: "c05", turn: 4 },
];

const CHATS_FOR_PAPER: ChatForPaper[] = [
  { id: "c01", title: "GLFT vs Avellaneda-Stoikov spread", turn: 4 },
  { id: "c02", title: "Reservation price vs mid", turn: 2 },
  { id: "c05", title: "OFI as a short-horizon signal", turn: 1 },
];

export function buildPaperDetail(paper: Paper): PaperDetail {
  const extractionQuality: "high" | "low" =
    paper.collection === "notes" ? "low" : "high";
  const provenance: ProvenanceEntry[] = [
    { k: "paper_id", v: paper.id, mono: true },
    { k: "short cite", v: paper.shortCite },
    {
      k: "fetch source",
      v: paper.arxiv
        ? "arXiv PDF → MinerU 2.1.0"
        : paper.doi
          ? "Crossref → MinerU 2.1.0"
          : "manual upload",
    },
    { k: "arXiv", v: paper.arxiv, mono: true, ...(paper.arxiv ? {} : { degradedReason: "no arXiv id" }) },
    { k: "DOI", v: paper.doi, mono: true, ...(paper.doi ? {} : { degradedReason: "no DOI" }) },
    { k: "extraction quality", v: extractionQuality, kind: "extraction-quality" },
    { k: "parser version", v: "mineru@2.1.0", mono: true },
    { k: "collection", v: paper.collection, kind: "coltag" },
    {
      k: "chunks in milvus",
      v: paper.degradedChunks ? null : String(paper.chunks),
      kind: "chunks",
      ...(paper.degradedChunks ? { degradedReason: "Milvus unreachable" } : {}),
    },
    { k: "ingested", v: paper.ingested, mono: true },
    { k: "file hash", v: "sha256:9c4f1e2a…d3b8a72f", mono: true },
  ];
  return {
    ...paper,
    provenance,
    queriesForPaper: QUERIES.slice(0, 5).map((q) => ({
      id: q.id,
      query: q.query,
      retriever: q.retriever,
      cost: q.cost,
      ts: q.ts,
    })),
    chatsForPaper: CHATS_FOR_PAPER,
    extractionQuality,
    parentChunks: 38,
    fileHash: "sha256:9c4f1e2a…d3b8a72f",
  };
}

const STAGES: QueryStage[] = [
  {
    n: "01", name: "decompose", state: "run", model: "gemini-3.1-flash",
    tokens: 2104, cost: 0.001, ms: 410,
    desc: "3 sub-questions · least-to-most",
    payload: {
      sub_questions: [
        "What is the Avellaneda–Stoikov optimal spread?",
        "How does GLFT derive the asymptotic quote?",
        "Where do the two frameworks diverge on inventory?",
      ],
      decomposition_strategy: "least-to-most",
      max_subqs: 5,
    },
  },
  {
    n: "02", name: "retrieve", state: "run", model: "milvus hybrid",
    candidates: 50, ms: 880,
    desc: "50 candidates · qwen3-1024 + bm25, rrf-fused",
    payload: {
      retriever: "milvus",
      mode: "hybrid",
      dense: { model: "qwen3-embed-1024", k: 50, ef_search: 96 },
      sparse: { model: "bm25", k: 50, k1: 1.2, b: 0.75 },
      fusion: "rrf",
      candidates_returned: 50,
      sub_query_calls: 3,
    },
  },
  {
    n: "03", name: "rerank", state: "run", model: "Qwen3-reranker-4B",
    in: 50, out: 8, ms: 620,
    desc: "top 50 → 8 (threshold 0.72)",
    payload: {
      model: "qwen3-reranker-4b",
      top_n: 8,
      score_threshold: 0.72,
      kept: [
        { chunk_id: "01_a3f1", score: 0.91 },
        { chunk_id: "02_b8c2", score: 0.88 },
        { chunk_id: "03_d4e9", score: 0.84 },
        { chunk_id: "04_f102", score: 0.81 },
        { chunk_id: "05_e7c4", score: 0.79 },
      ],
    },
  },
  {
    n: "04", name: "synthesize", state: "run", model: "gemini-3.1-flash",
    tokens: 15200, cost: 0.009, ms: 1180,
    desc: "8 chunks → 4 citations · gemini-3.1-flash",
    payload: {
      model: "gemini-3.1-flash",
      temperature: 0.2,
      max_tokens: 1024,
      input_chunks: 8,
      citations_emitted: 4,
      report_path: "logs/queries/2026-05-19/a1b2c3d4/report.md",
    },
  },
  { n: "05", name: "verify", state: "skip", desc: "", note: "not configured for milvus retriever" },
  { n: "06", name: "critique", state: "skip", desc: "", note: "not configured for milvus retriever" },
  { n: "07", name: "refine", state: "skip", desc: "", note: "not configured for milvus retriever" },
  { n: "08", name: "crag_retry", state: "skip", desc: "", note: "not triggered (synthesis confidence 0.82 > threshold 0.65)" },
];

const CHUNKS: RetrievedChunk[] = [
  { f: "01_a3f1.txt", cite: "Guéant et al. 2013", page: 7, score: 0.91, paperId: "1105.3115" },
  { f: "02_b8c2.txt", cite: "Avellaneda & Stoikov 2008", page: 4, score: 0.88, paperId: "2204.00817" },
  { f: "03_d4e9.txt", cite: "Avellaneda & Stoikov 2008", page: 6, score: 0.84, paperId: "2204.00817" },
  { f: "04_f102.txt", cite: "Guéant et al. 2013", page: 11, score: 0.81, paperId: "1105.3115" },
  { f: "05_e7c4.txt", cite: "Guéant et al. 2013", page: 14, score: 0.79, paperId: "1105.3115" },
  { f: "06_b9d1.txt", cite: "Cont, Kukanov, Stoikov 2014", page: 3, score: 0.76, paperId: "1011.6402" },
  { f: "07_c204.txt", cite: "Avellaneda & Stoikov 2008", page: 9, score: 0.74, paperId: "2204.00817" },
  { f: "08_a771.txt", cite: "Curato, Gatheral, Lillo 2016", page: 5, score: 0.72, paperId: "1412.4839" },
];

const REPORT: ReportBlock[] = [
  {
    heading: "Answer",
    parts: [
      { text: "The Guéant–Lehalle–Fernandez-Tapia (GLFT) framework recovers the Avellaneda–Stoikov optimal spread as a limiting case, but replaces the exponential-utility HJB with a closed-form asymptotic quote that is linear in inventory." },
      { cite: 1, resolved: true },
      { text: " The reservation price shift is identical to first order" },
      { cite: 2, resolved: true },
      { text: ", while GLFT's spread is symmetric around it and inventory-clamped." },
      { cite: 1, resolved: true },
      { cite: 3, resolved: true },
    ],
  },
  {
    heading: "Key contrast",
    parts: [
      { text: "A&S solves a finite-horizon stochastic-control problem yielding time-dependent quotes; GLFT assumes a stationary regime so quotes depend only on current inventory — cheaper to compute and with no terminal-time blow-up." },
      { cite: 3, resolved: true },
    ],
  },
  {
    heading: "Limits",
    parts: [
      { text: "Both frameworks assume symmetric arrival intensities and a Brownian mid-price; under regime-shifted volatility the GLFT closed form drifts." },
      { cite: 4, resolved: false },
    ],
  },
];

export function buildQueryDetail(row: QueryLogRow): QueryDetail {
  return {
    row,
    reportPath: "logs/queries/2026-05-19/a1b2c3d4/report.md",
    report: REPORT,
    stages: STAGES,
    chunks: CHUNKS,
    kpi: {
      tokensSub: "3 sub-queries · 1 synth call",
      costSub: "flash $0.001 · synth $0.009 · rerank free",
      latencySub: "retrieve 880ms · rerank 620ms · synth 1.18s",
    },
  };
}
