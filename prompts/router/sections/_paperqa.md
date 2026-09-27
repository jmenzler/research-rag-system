PaperQA tool (paperqa):

When the user's question would benefit from a multi-paper agentic read
(e.g. "compare the assumptions of GLFT vs Avellaneda-Stoikov", "what does
each paper claim about adverse selection"), emit a ``paperqa`` payload of
1-3 query rewrites that should be issued against the paper-qa retriever.

Each rewrite should reframe the question in a way that improves passage-
level extraction:
  - specify named entities ("GLFT", "Avellaneda-Stoikov 2008")
  - expand acronyms ("HJB" → "Hamilton-Jacobi-Bellman equation")
  - narrow the question to a paper class ("optimal market making PDE
    formulations", not "trading strategies")

Payload format (JSON, in ``tool_payloads.paperqa``):

  [
    "how does the GLFT model handle adverse selection?",
    "Avellaneda Stoikov GLFT optimal spread inventory penalty"
  ]

Stay disciplined: avoid generic rewrites that just paraphrase the original
query — paper-qa is most useful when the rewrites add discriminating
vocabulary the original question lacks.
