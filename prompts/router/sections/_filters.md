METADATA FILTERS:

Emit a `filters` field with year range or paper title constraints.

Supported filter keys:
- `year_min`: integer (inclusive lower bound on publication year)
- `year_max`: integer (inclusive upper bound on publication year)
- `paper_titles_required`: list of paper title strings the user explicitly named

Rules:
- Only populate when the user EXPLICITLY mentions a year range or paper names.
- Empty object `{}` otherwise — do NOT infer filters from the query content.

Examples:
User: "What does the 2023 Milionis paper say about LVR?"
filters: {"year_min": 2023, "year_max": 2023, "paper_titles_required": ["Milionis LVR"]}

User: "Papers on AMM impermanent loss from 2021 to 2023"
filters: {"year_min": 2021, "year_max": 2023}

User: "What is LVR?" (no year/paper mention)
filters: {}
