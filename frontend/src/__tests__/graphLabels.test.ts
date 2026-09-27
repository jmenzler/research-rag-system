/**
 * Wave-0 RED test for deriveShortLabel (D-03).
 *
 * Implementation lands in Plan 04.1-01 (frontend/src/lib/graphLabels.ts).
 *
 * Behavioural contract:
 *   - shortCite wins when present (pass-through verbatim)
 *   - null shortCite + title + year → first token + year (first token > 2 chars = surname)
 *   - null shortCite + short first token → title.slice(0,12) + year
 *   - null shortCite + no title + year → String(year)
 *   - null shortCite + no title + no year → "?"
 */
import { describe, expect, it } from "vitest";

type DeriveShortLabel = (
  shortCite: string | null | undefined,
  title: string | null | undefined,
  year: number | null | undefined,
) => string;

describe("deriveShortLabel (D-03) — Wave 0 pure-function tests", () => {
  it("returns shortCite verbatim when present", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(mod.deriveShortLabel("Avellaneda 2008", null, null)).toBe("Avellaneda 2008");
  });

  it("extracts first token + year when shortCite is null and first token length > 2", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(
      mod.deriveShortLabel(null, "High-frequency trading in a limit order book", 2008),
    ).toBe("High-frequency 2008");
  });

  it("returns String(year) when shortCite and title are null", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(mod.deriveShortLabel(null, null, 2008)).toBe("2008");
  });

  it("returns '?' when shortCite, title, and year are all null", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(mod.deriveShortLabel(null, null, null)).toBe("?");
  });

  it("uses title.slice(0,12) as surname proxy when first token length <= 2", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(mod.deriveShortLabel(null, "A short title", 2020)).toBe("A short titl 2020");
  });

  it("returns surname only (no year) when title present but year is null", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(mod.deriveShortLabel(null, "Reinforcement learning", null)).toBe("Reinforcement");
  });

  it("handles undefined inputs identically to null", async () => {
    const mod = (await import("@/lib/graphLabels")) as unknown as {
      deriveShortLabel: DeriveShortLabel;
    };
    expect(mod.deriveShortLabel(undefined, undefined, undefined)).toBe("?");
    expect(mod.deriveShortLabel("Cont 2014", undefined, undefined)).toBe("Cont 2014");
  });
});
