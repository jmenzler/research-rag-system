import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { groupKeymap, formatBinding } from "@/lib/keymap";
import type { KeyBinding } from "@/lib/keymap";

interface KeyboardHelpSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bindings: KeyBinding[];
}

export function KeyboardHelpSheet({
  open,
  onOpenChange,
  bindings,
}: KeyboardHelpSheetProps) {
  const groups = groupKeymap(bindings);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader className="sr-only">
          <DialogTitle>Keyboard shortcuts</DialogTitle>
          <DialogDescription>
            All available keyboard shortcuts, grouped by section.
          </DialogDescription>
        </DialogHeader>
        <div className="py-2">
          <h2 className="mb-4 text-base font-semibold">Keyboard shortcuts</h2>
          {Object.entries(groups).map(([group, items]) => (
            <section key={group} className="mb-4">
              <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                {group}
              </h3>
              <ul className="space-y-1">
                {items.map((binding) => (
                  <li
                    key={`${binding.group}-${binding.label}`}
                    className="flex items-center justify-between text-sm"
                  >
                    <span>{binding.label}</span>
                    <kbd className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">
                      {formatBinding(binding)}
                    </kbd>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
