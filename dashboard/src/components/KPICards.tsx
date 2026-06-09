interface Props {
  filteredDaily: { total: number; children: number; date: string; state: string }[];
  metadata: { total_enrollments: number; total_children: number; child_pct_national: number; states: number; districts: number; date_min: string; date_max: string };
}

export default function KPICards({ filteredDaily, metadata }: Props) {
  const total = filteredDaily.reduce((s, r) => s + r.total, 0);
  const children = filteredDaily.reduce((s, r) => s + r.children, 0);
  const states = new Set(filteredDaily.map((r) => r.state)).size;
  const dates = filteredDaily.length
    ? [filteredDaily[0].date, filteredDaily[filteredDaily.length - 1].date]
    : [metadata.date_min, metadata.date_max];

  const fmt = (n: number) => (n / 1e6).toFixed(2) + "M";

  const childPct = total > 0 ? (children / total) * 100 : 0;
  const isFiltered = states < metadata.states;

  return (
    <div className="kpi-grid">
      <div className="kpi-card">
        <div className="kpi-top">
          <span className="kpi-label">Total Enrollments</span>
        </div>
        <div className="kpi-value">{fmt(total)}</div>
        <div className="kpi-sub">
          {isFiltered
            ? `${((total / metadata.total_enrollments) * 100).toFixed(1)}% of national`
            : "National total"}
        </div>
      </div>

      <div className="kpi-card">
        <div className="kpi-top">
          <span className="kpi-label">Children (5-17)</span>
          <span className={`kpi-trend ${childPct > metadata.child_pct_national ? "up" : "down"}`}>
            {childPct.toFixed(1)}%
          </span>
        </div>
        <div className="kpi-value">{fmt(children)}</div>
        <div className="kpi-sub">
          {childPct.toFixed(1)}% of total · National avg: {metadata.child_pct_national}%
        </div>
      </div>

      <div className="kpi-card">
        <div className="kpi-top">
          <span className="kpi-label">States / UTs</span>
        </div>
        <div className="kpi-value">
          {states}
          {isFiltered && (
            <span style={{ fontSize: "0.8rem", color: "var(--text-dim)", fontWeight: 400 }}>
              {" "}/ {metadata.states}
            </span>
          )}
        </div>
        <div className="kpi-sub">
          {isFiltered ? "Filtered selection" : "All states"}
        </div>
      </div>

      <div className="kpi-card">
        <div className="kpi-top">
          <span className="kpi-label">Districts</span>
        </div>
        <div className="kpi-value">{metadata.districts}</div>
        <div className="kpi-sub">Across all states</div>
      </div>

      <div className="kpi-card">
        <div className="kpi-top">
          <span className="kpi-label">Date Range</span>
        </div>
        <div className="kpi-value" style={{ fontSize: "0.95rem" }}>
          {dates[0]} – {dates[1]}
        </div>
        <div className="kpi-sub">
          {metadata.date_min} – {metadata.date_max} (full)
        </div>
      </div>
    </div>
  );
}
