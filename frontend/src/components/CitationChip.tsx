interface Props {
  label: string;
  onOpen: (label: string) => void;
}

export function CitationChip({ label, onOpen }: Props) {
  return (
    <button
      type="button"
      onClick={() => onOpen(label)}
      aria-label={`Open source ${label}`}
      className="mx-0.5 inline rounded bg-teal-50 px-1.5 py-0.5 font-mono text-xs text-teal-800 ring-1 ring-teal-700/30 hover:bg-teal-100"
    >
      {label.slice(1, -1)}
    </button>
  );
}
