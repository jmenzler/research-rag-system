import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { groupKeymap, formatBinding } from "@/lib/keymap";
import type { KeyBinding } from "@/lib/keymap";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bindings: KeyBinding[];
}

export function CommandPalette({
  open,
  onOpenChange,
  bindings,
}: CommandPaletteProps) {
  const groups = groupKeymap(bindings);

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput placeholder="Type a command..." />
      <CommandList>
        <CommandEmpty>No commands found.</CommandEmpty>
        {Object.entries(groups).map(([group, items]) => (
          <CommandGroup key={group} heading={group}>
            {items.map((binding) => (
              <CommandItem
                key={`${binding.group}-${binding.label}`}
                onSelect={() => {
                  binding.action();
                  onOpenChange(false);
                }}
              >
                {binding.label}
                <CommandShortcut>{formatBinding(binding)}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  );
}
