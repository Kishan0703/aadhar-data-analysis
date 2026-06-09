import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { DistrictSummary } from "../App";

interface Props {
  districts: DistrictSummary[];
  selectedStates: string[];
}

const STATE_COLORS = [
  "#1d9e75", "#58a6ff", "#d29922", "#f85149", "#bc8cff",
  "#3fb950", "#db6d28", "#79c0ff", "#ff7b72", "#a5d6ff",
  "#7ee787", "#c9d1d9",
];

export default function DistrictsTab({ districts, selectedStates }: Props) {
  const [topN, setTopN] = useState(20);
  const [stateFilter, setStateFilter] = useState<string[]>([]);
  const [tableSearch, setTableSearch] = useState("");
  const [sortCol, setSortCol] = useState<string>("total");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const allStates = useMemo(
    () => [...new Set(districts.map((d) => d.state))].sort(),
    [districts]
  );

  const filtered = useMemo(() => {
    let data = districts;
    if (stateFilter.length) {
      data = data.filter((d) => stateFilter.includes(d.state));
    } else if (selectedStates.length < allStates.length) {
      data = data.filter((d) => selectedStates.includes(d.state));
    }
    return data.slice(0, topN).reverse();
  }, [districts, topN, stateFilter, selectedStates, allStates.length]);

  // Table data: searchable
  const tableData = useMemo(() => {
    let data = districts;
    if (stateFilter.length) {
      data = data.filter((d) => stateFilter.includes(d.state));
    } else if (selectedStates.length < allStates.length) {
      data = data.filter((d) => selectedStates.includes(d.state));
    }
    if (tableSearch) {
      const q = tableSearch.toLowerCase();
      data = data.filter(
        (d) =>
          d.district.toLowerCase().includes(q) ||
          d.state.toLowerCase().includes(q)
      );
    }
    return [...data]
      .sort((a, b) => {
        const aVal = (a as any)[sortCol] ?? 0;
        const bVal = (b as any)[sortCol] ?? 0;
        return sortDir === "desc" ? bVal - aVal : aVal - bVal;
      })
      .slice(0, 50);
  }, [districts, stateFilter, selectedStates, allStates.length, tableSearch, sortCol, sortDir]);

  const toggleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortCol(col);
      setSortDir("desc");
    }
  };

  const handleStateFilter = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStateFilter(
      Array.from(e.target.options)
        .filter((o) => o.selected)
        .map((o) => o.value)
    );
  };

  return (
    <div>
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>Top Districts by Enrollment</h3>
          <span className="badge">
            {stateFilter.length
              ? `${stateFilter.length} states selected`
              : "All states"}
          </span>
        </div>

        <div className="controls-row">
          <div className="controls-group">
            <label>Top N:</label>
            <input
              type="range"
              min={10}
              max={100}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
            />
            <span className="range-value">{topN}</span>
          </div>
          <div className="controls-group">
            <label>Filter by State:</label>
            <select
              multiple
              value={stateFilter}
              onChange={handleStateFilter}
              style={{ minHeight: 60, width: 160 }}
            >
              {allStates.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>

        <ResponsiveContainer
          width="100%"
          height={Math.max(350, topN * 24)}
        >
          <BarChart
            data={filtered}
            layout="vertical"
            margin={{ left: 140, right: 20 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e2d3d"
              horizontal={false}
            />
            <XAxis
              type="number"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
            />
            <YAxis
              type="category"
              dataKey="district"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              width={130}
            />
            <Tooltip
              contentStyle={{
                background: "#1c2333",
                border: "1px solid #2d4055",
                borderRadius: 6,
                fontSize: "0.78rem",
              }}
              formatter={(value: number) => [value.toLocaleString(), "Enrollments"]}
              labelFormatter={(label, payload) =>
                `${label} (${payload?.[0]?.payload?.state ?? ""})`
              }
            />
            <Bar dataKey="total" radius={[0, 4, 4, 0]}>
              {filtered.map((d) => {
                const idx = allStates.indexOf(d.state);
                return (
                  <Cell
                    key={d.district}
                    fill={STATE_COLORS[idx % STATE_COLORS.length]}
                    fillOpacity={0.85}
                  />
                );
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Data table */}
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>District Explorer</h3>
          <span className="badge">{tableData.length} of {districts.length.toLocaleString()}</span>
        </div>
        <input
          type="text"
          className="table-search"
          placeholder="Search district or state..."
          value={tableSearch}
          onChange={(e) => setTableSearch(e.target.value)}
        />
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ cursor: "default" }}>#</th>
                <th onClick={() => toggleSort("district")}>
                  District {sortCol === "district" ? (sortDir === "desc" ? "↓" : "↑") : ""}
                </th>
                <th onClick={() => toggleSort("state")}>
                  State {sortCol === "state" ? (sortDir === "desc" ? "↓" : "↑") : ""}
                </th>
                <th className="num" onClick={() => toggleSort("total")}>
                  Enrollments {sortCol === "total" ? (sortDir === "desc" ? "↓" : "↑") : ""}
                </th>
                <th className="num" onClick={() => toggleSort("child_pct")}>
                  Child % {sortCol === "child_pct" ? (sortDir === "desc" ? "↓" : "↑") : ""}
                </th>
                <th className="num" onClick={() => toggleSort("state_rank")}>
                  State Rank {sortCol === "state_rank" ? (sortDir === "desc" ? "↓" : "↑") : ""}
                </th>
                <th className="num" onClick={() => toggleSort("pincodes")}>
                  Pincodes {sortCol === "pincodes" ? (sortDir === "desc" ? "↓" : "↑") : ""}
                </th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((d, i) => (
                <tr key={`${d.state}-${d.district}`}>
                  <td style={{ color: "var(--text-dim)" }}>{i + 1}</td>
                  <td style={{ fontWeight: 500 }}>{d.district}</td>
                  <td>{d.state}</td>
                  <td className="num">{d.total.toLocaleString()}</td>
                  <td className="num">{d.child_pct.toFixed(1)}%</td>
                  <td className="num">#{d.state_rank}</td>
                  <td className="num">{d.pincodes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
