import { useMemo } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Area,
} from "recharts";
import type { WeekdayRow, MonthlyRow, WeeklyRow, HeatmapRow } from "../App";

const DAY_ORDER = [
  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
];
const MONTH_ORDER = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

interface Props {
  weekday: WeekdayRow[];
  monthly: MonthlyRow[];
  weekly: WeeklyRow[];
  quarterly: any[];
  heatmap: HeatmapRow[];
}

export default function SeasonalityTab({
  weekday,
  monthly,
  weekly,
  quarterly,
  heatmap,
}: Props) {
  const sortedWeekday = DAY_ORDER.map(
    (d) => weekday.find((w) => w.day_name === d)!
  ).filter(Boolean);

  const sortedMonthly = MONTH_ORDER.map(
    (m) => monthly.find((mo) => mo.month_name === m)!
  ).filter(Boolean);

  const heatValues = heatmap.reduce(
    (acc, r) => {
      acc[`${r.month_name}|${r.day_name}`] = r.total;
      return acc;
    },
    {} as Record<string, number>
  );

  const maxHeat = Math.max(...heatmap.map((r) => r.total), 1);

  const heatColor = (val: number) => {
    if (!val) return "#0d1117";
    const pct = val / maxHeat;
    const g = Math.round(30 + pct * 70);
    const b = Math.round(20 + pct * 50);
    return `rgb(29, ${g}, ${b})`;
  };

  const avgWeekday = sortedWeekday.reduce((s, r) => s + r.avg, 0) / sortedWeekday.length;

  return (
    <div>
      {/* First row: Weekday + Monthly */}
      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Avg Enrollments by Day of Week</h3>
            <span className="badge">
              Weekend is {avgWeekday > 0
                ? ((sortedWeekday.find((d) => d.day_name === "Sunday")?.avg ?? 0) / avgWeekday * 100).toFixed(0)
                : "—"}% of weekday
            </span>
          </div>
          <p className="chart-subtitle">
            Saturday sees a significant spike — 2.5× higher than typical weekdays.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={sortedWeekday}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis
                dataKey="day_name"
                stroke="#546377"
                tick={{ fontSize: 11 }}
              />
              <YAxis
                stroke="#546377"
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => (v / 1e3).toFixed(0) + "K"}
              />
              <Tooltip
                contentStyle={{
                  background: "#1c2333",
                  border: "1px solid #2d4055",
                  borderRadius: 6,
                  fontSize: "0.78rem",
                }}
                formatter={(value: number) => [value.toLocaleString(), "Avg daily"]}
              />
              <Bar dataKey="avg" radius={[4, 4, 0, 0]}>
                {sortedWeekday.map((r) => (
                  <Cell
                    key={r.day_name}
                    fill={
                      r.day_name === "Saturday" || r.day_name === "Sunday"
                        ? "#d29922"
                        : "#58a6ff"
                    }
                    fillOpacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Monthly Enrollment & Child %</h3>
            <span className="badge">Dual-axis</span>
          </div>
          <p className="chart-subtitle">
            Bars = total enrollment · Red line = child enrollment percentage.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={sortedMonthly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis
                dataKey="month_name"
                stroke="#546377"
                tick={{ fontSize: 10 }}
                angle={-25}
                textAnchor="end"
                height={50}
              />
              <YAxis
                yAxisId="left"
                stroke="#58a6ff"
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#f85149"
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => v.toFixed(0) + "%"}
              />
              <Tooltip
                contentStyle={{
                  background: "#1c2333",
                  border: "1px solid #2d4055",
                  borderRadius: 6,
                  fontSize: "0.78rem",
                }}
              />
              <Bar
                yAxisId="left"
                dataKey="total"
                fill="#58a6ff"
                fillOpacity={0.6}
                radius={[3, 3, 0, 0]}
                name="Enrollments"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="child_pct"
                stroke="#f85149"
                strokeWidth={2.5}
                dot={{ r: 4, fill: "#f85149" }}
                name="Child %"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Second row: Weekly trend + Quarterly */}
      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Weekly Enrollment Trend</h3>
            <span className="badge">Week-by-week</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={weekly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis
                dataKey="label"
                stroke="#546377"
                tick={{ fontSize: 9 }}
                interval={3}
              />
              <YAxis
                stroke="#546377"
                tick={{ fontSize: 10 }}
                tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
              />
              <Tooltip
                contentStyle={{
                  background: "#1c2333",
                  border: "1px solid #2d4055",
                  borderRadius: 6,
                  fontSize: "0.78rem",
                }}
                formatter={(value: number) => [value.toLocaleString(), "Weekly total"]}
              />
              <Area
                type="monotone"
                dataKey="total"
                stroke="#1d9e75"
                fill="#1d9e75"
                fillOpacity={0.1}
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Quarterly Breakdown</h3>
            <span className="badge">2025</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={quarterly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis dataKey="quarter" stroke="#546377" tick={{ fontSize: 12 }} />
              <YAxis
                stroke="#546377"
                tick={{ fontSize: 10 }}
                tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
              />
              <Tooltip
                contentStyle={{
                  background: "#1c2333",
                  border: "1px solid #2d4055",
                  borderRadius: 6,
                  fontSize: "0.78rem",
                }}
                formatter={(value: number) => [value.toLocaleString(), "Total"]}
              />
              <Bar dataKey="total" radius={[4, 4, 0, 0]} fill="#58a6ff" fillOpacity={0.8}>
                {quarterly.map((_: any, i: number) => (
                  <Cell
                    key={i}
                    fill={["#1d9e75", "#58a6ff", "#d29922", "#bc8cff"][i % 4]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
            {quarterly.map((q: any) => (
              <span key={q.quarter} className="state-tag">
                {q.quarter}: {(q.total / 1e6).toFixed(1)}M · {q.child_pct.toFixed(1)}% child
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>Enrollment Density: Month × Weekday</h3>
          <span className="badge">Values in millions</span>
        </div>
        <p className="chart-subtitle">
          Darker cells = higher enrollment. Rows are months, columns are weekdays.
        </p>
        <div className="heatmap-wrapper">
          <table className="heatmap-table">
            <thead>
              <tr>
                <th>Month</th>
                {DAY_ORDER.map((d) => (
                  <th key={d}>{d.slice(0, 3)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MONTH_ORDER.map((month) => (
                <tr key={month}>
                  <td
                    style={{
                      color: "var(--text-secondary)",
                      fontWeight: 500,
                      textAlign: "left",
                    }}
                  >
                    {month}
                  </td>
                  {DAY_ORDER.map((day) => {
                    const val = heatValues[`${month}|${day}`] || 0;
                    const bright = val > maxHeat * 0.35;
                    return (
                      <td
                        key={day}
                        style={{
                          background: heatColor(val),
                          color: bright ? "#fff" : "var(--text-dim)",
                        }}
                      >
                        {(val / 1e6).toFixed(1)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
