## Title hint (auto-name first turn)

Also emit a `title_hint` field: a concise topic label (≤ 60 characters, hard cap at 80) summarizing what this user question is about. The title will be used as the chat name in the sidebar.

Rules:
- 60 characters or fewer; never above 80.
- Plain text. No surrounding quotes. No trailing punctuation. No emoji.
- Sentence case. Capitalize the first letter; leave acronyms as-is (e.g., "GLFT model behavior", not "Glft model behavior").
- Describe the topic, not the question form. "GLFT model spread" not "How does GLFT work?".
- If the question is too vague to characterize, emit `null`.
