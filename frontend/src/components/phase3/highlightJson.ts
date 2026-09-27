/**
 * Lightweight JSON syntax highlighter (no deps) for the query-detail stage
 * payload viewer. Returns sanitized HTML wrapping each token in a
 * `.tok-*` span (keys blue / strings amber / numbers green / bools violet).
 *
 * Input is `JSON.stringify`'d locally (never raw user HTML) and every literal
 * `& < >` is escaped before any markup is injected, so the resulting string is
 * safe to pass to `dangerouslySetInnerHTML`.
 */
export function highlightJson(obj: unknown): string {
  const json = JSON.stringify(obj, null, 2);
  const safe = json
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return safe.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    (m) => {
      let cls = "tok-num";
      if (/^"/.test(m)) cls = /:$/.test(m) ? "tok-key" : "tok-str";
      else if (/true|false/.test(m)) cls = "tok-bool";
      else if (/null/.test(m)) cls = "tok-null";
      return `<span class="${cls}">${m}</span>`;
    },
  );
}

/** Plain (un-highlighted) escaped JSON, for the highlight-off path. */
export function plainJson(obj: unknown): string {
  return JSON.stringify(obj, null, 2)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
