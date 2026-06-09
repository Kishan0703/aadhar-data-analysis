interface Props {
  metadata: { all_states: string[]; date_min: string; date_max: string; dupes_removed: number; total_enrollments: number };
  selectedStates: string[];
  onStatesChange: (states: string[]) => void;
  dateRange: [string, string];
  onDateRangeChange: (range: [string, string]) => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
}

export default function Sidebar({
  metadata,
  selectedStates,
  onStatesChange,
  dateRange,
  onDateRangeChange,
  searchQuery,
  onSearchChange,
}: Props) {
  const filtered = metadata.all_states.filter((s) =>
    s.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const opts = Array.from(e.target.options)
      .filter((o) => o.selected)
      .map((o) => o.value);
    onStatesChange(opts);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">🇮🇳</div>
        <div className="sidebar-brand-text">
          <h2>Aadhaar Dashboard</h2>
          <p>UIDAI Enrollment Analytics</p>
        </div>
      </div>

      <div className="sidebar-section">
        <label>
          States
          <span
            onClick={() =>
              onStatesChange(
                selectedStates.length === metadata.all_states.length
                  ? []
                  : [...metadata.all_states]
              )
            }
          >
            {selectedStates.length === metadata.all_states.length
              ? "Deselect all"
              : "Select all"}
          </span>
        </label>
        <input
          type="text"
          className="state-search"
          placeholder="Search states..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
        <select
          className="state-select"
          multiple
          value={filtered.filter((s) => selectedStates.includes(s))}
          onChange={handleSelect}
        >
          {filtered.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="sidebar-section">
        <label>Date Range</label>
        <div className="date-inputs">
          <input
            type="date"
            value={dateRange[0]}
            min={metadata.date_min}
            max={metadata.date_max}
            onChange={(e) => onDateRangeChange([e.target.value, dateRange[1]])}
          />
          <span>→</span>
          <input
            type="date"
            value={dateRange[1]}
            min={metadata.date_min}
            max={metadata.date_max}
            onChange={(e) => onDateRangeChange([dateRange[0], e.target.value])}
          />
        </div>
      </div>

      <button
        className="sidebar-btn"
        onClick={() => {
          onStatesChange([...metadata.all_states]);
          onDateRangeChange([metadata.date_min, metadata.date_max]);
        }}
      >
        ↺ Reset Filters
      </button>

      <div className="sidebar-notes">
        <p><strong>⚠ Known Issues</strong></p>
        <p>• August 2025 data missing</p>
        <p>• 9 variants of "West Bengal" in source</p>
        <p>• CV = 220% (extreme daily volatility)</p>
      </div>
    </aside>
  );
}
