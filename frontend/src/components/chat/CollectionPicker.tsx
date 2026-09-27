/**
 * <CollectionPicker> (Plan 02-13).
 *
 * Four shadcn <Checkbox> rows — one per Milvus partition (trading, ecology,
 * notes, system). Default = all four selected (D-03). Controlled
 * component: the parent NewChatDialog owns the selected-set state.
 *
 * Empty selection is meaningful: the server treats an empty collections
 * list as "all four" (matches D-03 ergonomics for filter chips), so the
 * UI never forces the user to keep at least one box checked.
 */
import { type ReactElement } from "react";

import { Checkbox } from "@/components/ui/checkbox";
import { COLLECTION_ORDER, type CollectionId } from "@/lib/retrievers";

export interface CollectionPickerProps {
  /** Currently checked collection ids. */
  value: CollectionId[];
  /** Called with the next full set of selected ids on toggle. */
  onChange: (next: CollectionId[]) => void;
  /** When true, all checkboxes render in a disabled state (used while the
   * dialog is in the submitting state — prevents racing the POST). */
  disabled?: boolean;
}

export function CollectionPicker(props: CollectionPickerProps): ReactElement {
  const { value, onChange, disabled = false } = props;
  const selected = new Set(value);

  function toggle(id: CollectionId, checked: boolean): void {
    const next = new Set(selected);
    if (checked) next.add(id);
    else next.delete(id);
    // Preserve canonical order regardless of click sequence so server
    // PATCHes are deterministic.
    onChange(COLLECTION_ORDER.filter((c) => next.has(c)));
  }

  return (
    <div data-slot="collection-picker" className="flex flex-col gap-2">
      {COLLECTION_ORDER.map((id) => {
        const checked = selected.has(id);
        return (
          <label
            key={id}
            data-collection={id}
            className="flex cursor-pointer items-center gap-2 text-sm"
          >
            <Checkbox
              name={id}
              aria-label={id}
              checked={checked}
              disabled={disabled}
              onCheckedChange={(c) => toggle(id, c === true)}
            />
            <span>{id}</span>
          </label>
        );
      })}
    </div>
  );
}
