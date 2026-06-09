import { useState, useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { kmeans } from "../utils/kmeans";
import type { StateSummary } from "../App";

interface Props {
  stateSummary: StateSummary[];
}

const CLUSTER_COLORS = ["#1d9e75", "#58a6ff", "#d29922", "#bc8cff", "#f85149", "#3fb950"];

export default function ClustersTab({ stateSummary }: Props) {
  const [nClusters, setNClusters] = useState(4);

  const clustered = useMemo(() => {
    const features = ["total", "children", "child_pct", "districts"];
    const vals = stateSummary.map((s) =>
      features.map((f) => s[f as keyof typeof s] as number)
    );

    const means = features.map((_, i) =>
      vals.reduce((sum, r) => sum + r[i], 0) / vals.length
    );
    const stds = features.map((_, i) =>
      Math.sqrt(vals.reduce((sum, r) => sum + (r[i] - means[i]) ** 2, 0) / vals.length)
    );
    const standardized = vals.map((r) =>
      r.map((v, i) => (v - means[i]) / (stds[i] || 1))
    );

    const { labels } = kmeans(standardized, nClusters);
    return stateSummary.map((s, i) => ({ ...s, cluster: labels[i] }));
  }, [stateSummary, nClusters]);

  const clusters = useMemo(
    () =>
      Array.from({ length: nClusters }, (_, ci) => {
        const members = clustered.filter((s) => s.cluster === ci);
        return {
          id: ci,
          states: members.map((s) => s.state),
          count: members.length,
          avgTotal: members.length
            ? members.reduce((s, m) => s + m.total, 0) / members.length
            : 0,
          avgChildPct: members.length
            ? members.reduce((s, m) => s + m.child_pct, 0) / members.length
            : 0,
          avgPerCapita: members.length
            ? members.reduce((s, m) => s + m.per_capita, 0) / members.length
            : 0,
          avgDistricts: members.length
            ? members.reduce((s, m) => s + m.districts, 0) / members.length
            : 0,
          totalEnrollments: members.reduce((s, m) => s + m.total, 0),
        };
      }),
    [clustered, nClusters]
  );

  return (
    <div>
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>State Clustering: Volume vs Child %</h3>
          <span className="badge">K-Means · Bubble size = per-capita</span>
        </div>

        <div className="controls-row">
          <div className="controls-group">
            <label>Number of clusters:</label>
            <input
              type="range"
              min={2}
              max={6}
              value={nClusters}
              onChange={(e) => setNClusters(Number(e.target.value))}
            />
            <span className="range-value">{nClusters}</span>
          </div>
        </div>

        <div className="insight-box info">
          💡 Each cluster groups states with similar enrollment patterns. Hover over a point to see details.
        </div>

        <div className="cluster-legend">
          {clusters.map((c) => (
            <div key={c.id} className="cluster-item">
              <span
                className="cluster-dot"
                style={{
                  background: CLUSTER_COLORS[c.id % CLUSTER_COLORS.length],
                }}
              />
              <span>
                Cluster {c.id} ({c.count} states)
              </span>
            </div>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={450}>
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
              range={[50, 500]}
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
            {Array.from({ length: nClusters }, (_, ci) => (
              <Scatter
                key={ci}
                name={`Cluster ${ci}`}
                data={clustered.filter((s) => s.cluster === ci)}
                fill={CLUSTER_COLORS[ci % CLUSTER_COLORS.length]}
                fillOpacity={0.65}
                shape="circle"
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster profiles */}
      <div className="chart-card">
        <div className="chart-card-header">
          <h3>Cluster Profiles</h3>
          <span className="badge">{nClusters} segments</span>
        </div>
        <div className="insight-box info">
          💡 Click each cluster to see which states belong and their key metrics.
        </div>
        {clusters.map((c) => (
          <details key={c.id} className="cluster-detail">
            <summary>
              Cluster {c.id} · {c.count} states ·{" "}
              {(c.avgTotal / 1e6).toFixed(1)}M avg enrollment ·{" "}
              {c.avgChildPct.toFixed(1)}% avg child ·{" "}
              {c.avgPerCapita.toFixed(0)} avg per-capita
            </summary>
            <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <div className="stat-box" style={{ flex: 1, minWidth: 100 }}>
                <div className="value">{(c.totalEnrollments / 1e6).toFixed(1)}M</div>
                <div className="label">Total Enrollment</div>
              </div>
              <div className="stat-box" style={{ flex: 1, minWidth: 100 }}>
                <div className="value">{c.avgChildPct.toFixed(1)}%</div>
                <div className="label">Avg Child %</div>
              </div>
              <div className="stat-box" style={{ flex: 1, minWidth: 100 }}>
                <div className="value">{c.avgPerCapita.toFixed(0)}</div>
                <div className="label">Avg Per-Capita</div>
              </div>
              <div className="stat-box" style={{ flex: 1, minWidth: 100 }}>
                <div className="value">{c.avgDistricts.toFixed(0)}</div>
                <div className="label">Avg Districts</div>
              </div>
            </div>
            <div className="states-list">
              {c.states.map((s) => (
                <span key={s} className="state-tag">{s}</span>
              ))}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
