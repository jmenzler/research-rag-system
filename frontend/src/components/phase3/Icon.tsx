/**
 * Phase 3 Icon wrapper — maps the design bundle's icon names to lucide-react
 * components (Lucide-only, no emoji, no hand-rolled SVG). Stroke 1.75 matches
 * the design's line weight. Keeps call-sites terse (`<Icon name="library" />`)
 * while staying in the project's existing lucide-react dependency.
 */
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpDown,
  Check,
  ChevronRight,
  Coins,
  Copy,
  ExternalLink,
  FileJson,
  FileText,
  Hash,
  Layers,
  Library,
  Link2,
  MessageSquare,
  Plus,
  Search,
  X,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { CSSProperties, ReactElement } from "react";

const MAP: Record<string, LucideIcon> = {
  "message-square": MessageSquare,
  library: Library,
  activity: Activity,
  search: Search,
  "chevron-right": ChevronRight,
  "arrow-right": ArrowRight,
  "arrow-left": ArrowLeft,
  "arrow-up-down": ArrowUpDown,
  "arrow-up": ArrowUp,
  "arrow-down": ArrowDown,
  x: X,
  plus: Plus,
  "external-link": ExternalLink,
  link: Link2,
  "file-text": FileText,
  "file-json": FileJson,
  hash: Hash,
  "alert-triangle": AlertTriangle,
  "alert-circle": AlertCircle,
  check: Check,
  coins: Coins,
  zap: Zap,
  layers: Layers,
  copy: Copy,
};

export interface IconProps {
  name: keyof typeof MAP | string;
  size?: number;
  className?: string;
  style?: CSSProperties;
}

export function Icon({
  name,
  size = 16,
  className = "",
  style,
}: IconProps): ReactElement {
  const Cmp = MAP[name];
  if (Cmp === undefined) {
    return (
      <span
        className={`icon ${className}`}
        style={{ width: size, height: size, ...style }}
      />
    );
  }
  return (
    <Cmp
      size={size}
      strokeWidth={1.75}
      className={`icon ${className}`}
      {...(style ? { style } : {})}
    />
  );
}
