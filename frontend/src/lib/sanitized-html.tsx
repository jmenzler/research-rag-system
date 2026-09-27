/**
 * The ONE wrapper file allowed to use `dangerouslySetInnerHTML` (D-24, CRIT-9).
 *
 * Phase 1 likely never uses this — react-markdown + rehype-sanitize handles
 * the LLM-answer rendering path safely without raw HTML injection. This file
 * exists to anchor the ESLint `react/no-danger` allowlist so additions in
 * later phases either go through this wrapper or trigger a CI failure.
 *
 * Pass HTML that has ALREADY been sanitized by rehype-sanitize or an
 * equivalent allowlist sanitizer. Never pass raw LLM output.
 */
import type { CSSProperties, ReactElement } from "react";

export interface SanitizedHtmlProps {
  html: string;
  className?: string;
  style?: CSSProperties;
}

export function SanitizedHtml(props: SanitizedHtmlProps): ReactElement {
  return (
    <div
      className={props.className}
      style={props.style}
      // eslint-disable-next-line react/no-danger -- D-24 single allowlist
      dangerouslySetInnerHTML={{ __html: props.html }}
    />
  );
}
