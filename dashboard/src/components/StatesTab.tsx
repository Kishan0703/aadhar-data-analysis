import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ZAxis,
} from "recharts";
import type { StateSummary } from "../App";

interface Props {
  stateSummary: StateSummary[];
  stateMonthly: any[];
}

const COLORS = [
  "#1d9e75", "#58a6ff", "#d29922", "#f85149", "#bc8cff",
  "#3fb950", "#db6d28", "#79c0ff", "#ff7b72", "#a5d6ff",
];

export default function StatesTab({ stateSummary }: Props) {
  const [sortBy, setSortBy] = useState<"total" | "child_pct" | "per_capita">("total");
  const [topN, setTopN] = useState(20);

  const sorted = useMemo(
    () =>
      [...stateSummary]
        .sort((a, b) => b[sortBy] - a[sortBy])
        .slice(0, topN)
        .reverse(),
    [stateSummary, sortBy, topN]
  );

  const dataKey = sortBy === "total" ? "total" : sortBy === "child_pct" ? "child_pct" : "per_capita";
  const xLabel =
    sortBy === "total"
      ? "Total Enrollments"
      : sortBy === "child_pct"
        ? "Child Enrollment %"
        : "Per-Capita Rate";

  const fmt =
    sortBy === "total"
      ? (v: number) => (v / 1e6).toFixed(1) + "M"
      : sortBy === "child_pct"
        ? (v: number) => v.toFixed(1) + "%"
        : (v: number) => v.toFixed(0);

  const scatterData = stateSummary.filter((s) => s.total >= 1000);

  return (
    <div>
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>State Rankings</h3>
          <span className="badge">{sortBy === "per_capita" ? "Per capita" : sortBy === "child_pct" ? "Child %" : "Volume"}</span>
        </div>

        <div className="controls-row">
          <div className="controls-group">
            <label>Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
            >
              <option value="total">Total enrollment</option>
              <option value="child_pct">Child enrollment %</option>
              <option value="per_capita">Per-capita rate</option>
            </select>
          </div>
          <div className="controls-group">
            <label>Top N:</label>
            <input
              type="range"
              min={5}
              max={stateSummary.length}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
            />
            <span className="range-value">{topN}</span>
          </div>
        </div>

        <div className="insight-box info">
          💡 {sortBy === "total"
            ? "Top states by absolute enrollment volume. Uttar Pradesh alone accounts for ~18%."
            : sortBy === "child_pct"
              ? "States ranked by child enrollment share. Higher % indicates better youth coverage."
              : "Per-capita enrollment rate — smaller states like Chandigarh and Delhi lead."}
        </div>

        <ResponsiveContainer width="100%" height={Math.max(300, topN * 26)}>
          <BarChart data={sorted} layout="vertical" margin={{ left: 120, right: 20 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e2d3d"
              horizontal={false}
            />
            <XAxis
              type="number"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              tickFormatter={fmt}
            />
            <YAxis
              type="category"
              dataKey="state"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              width={110}
            />
            <Tooltip
              contentStyle={{
                background: "#1c2333",
                border: "1px solid #2d4055",
                borderRadius: 6,
                fontSize: "0.78rem",
              }}
              formatter={(value: number, _name: string) => [
                sortBy === "total"
                  ? value.toLocaleString()
                  : sortBy === "child_pct"
                    ? value.toFixed(1) + "%"
                    : value.toFixed(0),
                xLabel,
              ]}
              labelFormatter={(label) => `State: ${label}`}
            />
            <Bar dataKey={dataKey} radius={[0, 4, 4, 0]}>
              {sorted.map((_, i) => (
                <Cell
                  key={i}
                  fill={COLORS[i % COLORS.length]}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Scatter: Volume vs Child % with Per-Capita as size */}
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>Enrollment Volume vs Child %</h3>
          <span className="badge">Bubble size = per-capita rate</span>
        </div>
        <p className="chart-subtitle">
          Each bubble is a state. X-axis = total enrollment, Y-axis = child %, size = per-capita rate.
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
            <XAxis
              type="number"
              dataKey="total"
              name="Enrollments"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
            />
            <YAxis
              type="number"
              dataKey="child_pct"
              name="Child %"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              domain={[0, "auto"]}
              tickFormatter={(v: number) => v.toFixed(0) + "%"}
            />
            <ZAxis
              type="number"
              dataKey="per_capita"
              range={[40, 600]}
              name="Per Capita"
            />
            <Tooltip
              contentStyle={{
                background: "#1c2333",
                border: "1px solid #2d4055",
                borderRadius: 6,
                fontSize: "0.78rem",
              }}
              formatter={(value: number, name: string) => [
                name === "Enrollments"
                  ? value.toLocaleString()
                  : name === "Child %"
                    ? value.toFixed(1) + "%"
                    : value.toFixed(0),
                name,
              ]}
              labelFormatter={(label) => `State: ${label}`}
            />
            <Scatter
              data={scatterData}
              fill="#1d9e75"
              fillOpacity={0.6}
              shape="circle"
            >
              {scatterData.map((d, i) => (
                <Cell
                  key={d.state}
                  fill={
                    d.child_pct > 15
                      ? "#1d9e75"
                      : d.child_pct > 10
                        ? "#d29922"
                        : "#f85149"
                  }
                  fillOpacity={0.7}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Quick rankings table */}
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>State Rankings Summary</h3>
          <span className="badge">Top 10 by volume</span>
        </div>
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>State</th>
                <th className="num">Enrollments</th>
                <th className="num">Child %</th>
                <th className="num">Per Capita</th>
                <th className="num">Share</th>
                <th className="num">Districts</th>
              </tr>
            </thead>
            <tbody>
              {stateSummary.slice(0, 15).map((s, i) => (
                <tr key={s.state}>
                  <td style={{ color: "var(--text-dim)" }}>{i + 1}</td>
                  <td style={{ fontWeight: 500 }}>{s.state}</td>
                  <td className="num">{s.total.toLocaleString()}</td>
                  <td className="num">{s.child_pct.toFixed(1)}%</td>
                  <td className="num">{s.per_capita.toFixed(0)}</td>
                  <td className="num">{s.share_pct.toFixed(1)}%</td>
                  <td className="num">{s.districts}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
