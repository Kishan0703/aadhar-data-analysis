import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { DailyNational } from "../App";

interface Props {
  dailyNational: DailyNational[];
}

export default function ForecastTab({ dailyNational }: Props) {
  const forecast = useMemo(() => {
    if (!dailyNational.length) return [];

    const recent = dailyNational.slice(-30).map((d, i) => ({ x: i, y: d.total }));
    const n = recent.length;
    const sumX = recent.reduce((s, r) => s + r.x, 0);
    const sumY = recent.reduce((s, r) => s + r.y, 0);
    const sumXY = recent.reduce((s, r) => s + r.x * r.y, 0);
    const sumX2 = recent.reduce((s, r) => s + r.x * r.x, 0);

    const denom = n * sumX2 - sumX * sumX;
    const slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
    const intercept = sumY / n - (slope * sumX) / n;

    const lastDate = new Date(dailyNational[dailyNational.length - 1].date);
    const result: any[] = [];

    const tail = dailyNational.slice(-14);
    tail.forEach((d) => {
      result.push({
        dateLabel: d.date.slice(5),
        actual: d.total,
        forecast: null,
      });
    });

    for (let i = 1; i <= 14; i++) {
      const dt = new Date(lastDate);
      dt.setDate(dt.getDate() + i);
      const label = `${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`;
      const pred = slope * (n - 1 + i) + intercept;
      result.push({
        dateLabel: label,
        actual: null,
        forecast: Math.round(Math.max(pred, 0)),
      });
    }

    return result;
  }, [dailyNational]);

  const dailyMean = dailyNational.length
    ? dailyNational.reduce((s, d) => s + d.total, 0) / dailyNational.length
    : 0;
  const recentMean = dailyNational.length
    ? dailyNational.slice(-30).reduce((s, d) => s + d.total, 0) / 30
    : 0;

  return (
    <div>
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>Enrollment Forecast (14-Day Projection)</h3>
          <span className="badge">Linear regression</span>
        </div>

        <div className="stats-grid" style={{ marginBottom: "1rem" }}>
          <div className="stat-box">
            <div className="value">{(dailyMean / 1e6).toFixed(2)}M</div>
            <div className="label">Historical Daily Avg</div>
          </div>
          <div className="stat-box">
            <div className="value">{(recentMean / 1e6).toFixed(2)}M</div>
            <div className="label">Last 30 Days Avg</div>
          </div>
          <div className="stat-box">
            <div className="value">
              {recentMean > 0
                ? `${((recentMean / dailyMean - 1) * 100).toFixed(1)}%`
                : "—"}
            </div>
            <div className="label">Recent vs Overall</div>
          </div>
          <div className="stat-box">
            <div className="value">{forecast.length - 14} days</div>
            <div className="label">Forecast Horizon</div>
          </div>
        </div>

        <div className="insight-box warn">
          ⚠ Simple projection based on last 30 days. Does not account for seasonality, policy changes,
          or known data gaps (missing August 2025).
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={forecast}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
            <XAxis
              dataKey="dateLabel"
              stroke="#546377"
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#546377"
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"}
            />
            <Tooltip
              contentStyle={{
                background: "#1c2333",
                border: "1px solid #2d4055",
                borderRadius: 6,
                fontSize: "0.78rem",
              }}
              formatter={(value: number) => [value ? value.toLocaleString() : "—", "Enrollments"]}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={false}
              name="Historical"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#1d9e75"
              strokeWidth={2.5}
              strokeDasharray="6 3"
              dot={{ r: 3, fill: "#1d9e75" }}
              name="Forecast (14-day)"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
