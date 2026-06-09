import { useState, useMemo } from "react";
import {
  ComposedChart,
  LineChart,
  Line,
  Area,
  Bar,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { DailyNational, Metadata } from "../App";

interface Props {
  dailyNational: DailyNational[];
  metadata: Metadata;
}

export default function TrendsTab({ dailyNational, metadata }: Props) {
  const [showAnomalies, setShowAnomalies] = useState(true);
  const [metric, setMetric] = useState("Total");

  const data = useMemo(() => {
    return dailyNational.map((d) => ({
      ...d,
      dateLabel: d.date.slice(5),
      adults: d.total - d.children,
      rolling7_adults: 0,
    })).map((d, i, arr) => {
      if (i >= 6) {
        let sum = 0;
        for (let j = i - 6; j <= i; j++) sum += arr[j].adults;
        return { ...d, rolling7_adults: Math.round(sum / 7) };
      }
      return d;
    });
  }, [dailyNational]);

  const anomalies = data.filter((d) => d.anomaly);

  const yKey =
    metric === "Children (5-17)"
      ? "children"
      : metric === "Adults (18+)"
        ? "adults"
        : "total";
  const rollKey =
    metric === "Children (5-17)"
      ? "rolling7_children"
      : metric === "Adults (18+)"
        ? "rolling7_adults"
        : "rolling7";

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const row = payload[0]?.payload;
    return (
      <div
        style={{
          background: "#1c2333",
          border: "1px solid #2d4055",
          borderRadius: 6,
          padding: "0.6rem 0.8rem",
          fontSize: "0.78rem",
          maxWidth: 240,
        }}
      >
        <p style={{ color: "#8b949e", marginBottom: 4, fontWeight: 600 }}>
          {row?.date}
        </p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color, margin: "1px 0", display: "flex", justifyContent: "space-between", gap: "1rem" }}>
            <span>{p.name}</span>
            <span style={{ fontWeight: 600 }}>{Number(p.value).toLocaleString()}</span>
          </p>
        ))}
        {row && (
          <>
            <p style={{ color: "#8b949e", marginTop: 4, fontSize: "0.72rem" }}>
              WoW growth: {row.wow_growth > 0 ? "+" : ""}{row.wow_growth.toFixed(1)}%
            </p>
            {row.anomaly && (
              <p style={{ color: "#f85149", marginTop: 2, fontWeight: 600 }}>
                ⚠ Anomaly (z = {row.z.toFixed(2)})
              </p>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <div>
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>Daily Enrollment Trend</h3>
          <span className="badge">7-day avg · Anomaly detection</span>
        </div>
        <div className="controls-row">
          <div className="toggle-group">
            <button
              className={`toggle ${showAnomalies ? "on" : ""}`}
              onClick={() => setShowAnomalies(!showAnomalies)}
            />
            <span className="toggle-label">
              Highlight anomalies ({anomalies.length})
            </span>
          </div>
          <div className="controls-group">
            <label>Show:</label>
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option>Total</option>
              <option>Children (5-17)</option>
              <option>Adults (18+)</option>
            </select>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
            <XAxis
              dataKey="dateLabel"
              stroke="#546377"
              tick={{ fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#546377"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area
              type="monotone"
              dataKey={yKey}
              stroke="#58a6ff"
              strokeWidth={1.5}
              fill="#58a6ff"
              fillOpacity={0.08}
              dot={false}
              name={metric}
            />
            <Line
              type="monotone"
              dataKey={rollKey}
              stroke="#1d9e75"
              strokeWidth={2.5}
              dot={false}
              name="7-day rolling avg"
            />
            {showAnomalies && anomalies.length > 0 && (
              <Scatter
                data={anomalies}
                dataKey={yKey}
                fill="#f85149"
                shape="cross"
                name={`Anomaly (${anomalies.length})`}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>

        {anomalies.length > 0 && (
          <details style={{ marginTop: "0.75rem" }}>
            <summary style={{ cursor: "pointer", color: "#8b949e", fontSize: "0.8rem" }}>
              📋 {anomalies.length} anomalous days (|z| &gt; 2.5)
            </summary>
            <div className="anomaly-grid" style={{ marginTop: "0.5rem" }}>
              {anomalies.map((a) => (
                <div key={a.date} className="anomaly-item">
                  {a.date} — {a.total.toLocaleString()} (z={a.z.toFixed(2)})
                </div>
              ))}
            </div>
          </details>
        )}
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Cumulative Enrollment</h3>
            <span className="badge">Total to date</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={data} />
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Week-over-Week Growth</h3>
            <span className="badge">% change</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={data.slice(7)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis
                dataKey="dateLabel"
                stroke="#546377"
                tick={{ fontSize: 10 }}
                interval="preserveStartEnd"
              />
              <YAxis
                stroke="#546377"
                tick={{ fontSize: 10 }}
                tickFormatter={(v: number) => v.toFixed(0) + "%"}
              />
              <Tooltip
                contentStyle={{
                  background: "#1c2333",
                  border: "1px solid #2d4055",
                  borderRadius: 6,
                  fontSize: "0.78rem",
                }}
                formatter={(v: number) => [v.toFixed(1) + "%", "WoW Growth"]}
              />
              <Bar
                dataKey="wow_growth"
                fill="#d29922"
                fillOpacity={0.6}
                radius={[2, 2, 0, 0]}
              />
              <Line
                type="monotone"
                dataKey="wow_growth"
                stroke="#d29922"
                strokeWidth={1.5}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-box">
          <div className="value">{(metadata.daily_mean / 1e6).toFixed(2)}M</div>
          <div className="label">Daily Mean</div>
        </div>
        <div className="stat-box">
          <div className="value">{(metadata.daily_median / 1e6).toFixed(2)}M</div>
          <div className="label">Daily Median</div>
        </div>
        <div className="stat-box">
          <div className="value">{metadata.daily_cv}%</div>
          <div className="label">Coefficient of Variation</div>
        </div>
        <div className="stat-box">
          <div className="value">{(metadata.daily_max / 1e6).toFixed(2)}M</div>
          <div className="label">Peak Day</div>
        </div>
        <div className="stat-box">
          <div className="value">{(metadata.daily_min / 1e6).toFixed(2)}M</div>
          <div className="label">Lowest Day</div>
        </div>
        <div className="stat-box">
          <div className="value">{metadata.peak_day || "—"}</div>
          <div className="label">Peak Date</div>
        </div>
      </div>
    </div>
  );
}

/* Mini cumulative chart — inline component */
function AreaChart({ data }: { data: any[] }) {
  return (
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
      <XAxis
        dataKey="dateLabel"
        stroke="#546377"
        tick={{ fontSize: 10 }}
        interval="preserveStartEnd"
      />
      <YAxis
        stroke="#546377"
        tick={{ fontSize: 10 }}
        tickFormatter={(v: number) => (v / 1e6).toFixed(0) + "M"}
      />
      <Tooltip
        contentStyle={{
          background: "#1c2333",
          border: "1px solid #2d4055",
          borderRadius: 6,
          fontSize: "0.78rem",
        }}
        formatter={(v: number) => [v.toLocaleString(), "Cumulative"]}
      />
      <Area
        type="monotone"
        dataKey="cumulative"
        stroke="#1d9e75"
        fill="#1d9e75"
        fillOpacity={0.12}
        strokeWidth={2}
        dot={false}
      />
    </LineChart>
  );
}
