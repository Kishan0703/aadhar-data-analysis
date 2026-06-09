import { useState, useMemo, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import KPICards from "./components/KPICards";
import TrendsTab from "./components/TrendsTab";
import StatesTab from "./components/StatesTab";
import SeasonalityTab from "./components/SeasonalityTab";
import DistrictsTab from "./components/DistrictsTab";
import ClustersTab from "./components/ClustersTab";
import ForecastTab from "./components/ForecastTab";

export interface DailyState {
  date: string;
  state: string;
  total: number;
  children: number;
  adults: number;
  districts: number;
  pincodes: number;
  child_pct: number;
  population: number;
  per_capita: number;
}

export interface DailyNational {
  date: string;
  total: number;
  children: number;
  adults: number;
  child_pct: number;
  rolling7: number;
  rolling30: number;
  rolling7_children: number;
  cumulative: number;
  z: number;
  anomaly: boolean;
  dow: string;
  wow_growth: number;
  mom_growth: number;
}

export interface StateSummary {
  state: string;
  total: number;
  children: number;
  adults: number;
  districts: number;
  pincodes: number;
  records: number;
  child_pct: number;
  adult_pct: number;
  population: number;
  per_capita: number;
  avg_per_district: number;
  share_pct: number;
  child_adult_ratio: number;
}

export interface DistrictSummary {
  state: string;
  district: string;
  total: number;
  children: number;
  adults: number;
  records: number;
  pincodes: number;
  child_pct: number;
  state_rank: number;
}

export interface WeekdayRow {
  day_name: string;
  total: number;
  children: number;
  transactions: number;
  avg: number;
  child_pct: number;
}
export interface MonthlyRow {
  month_name: string;
  total: number;
  children: number;
  adults: number;
  records: number;
  districts: number;
  child_pct: number;
  share_pct: number;
  cumulative: number;
}
export interface HeatmapRow {
  month_name: string;
  day_name: string;
  total: number;
}
export interface WeeklyRow {
  label: string;
  total: number;
  children: number;
  child_pct: number;
}

export interface Metadata {
  total_records: number;
  total_enrollments: number;
  total_children: number;
  total_adults: number;
  states: number;
  districts: number;
  pincodes: number;
  date_min: string;
  date_max: string;
  num_days: number;
  dupes_removed: number;
  all_states: string[];
  child_pct_national: number;
  daily_mean: number;
  daily_median: number;
  daily_std: number;
  daily_min: number;
  daily_max: number;
  daily_cv: number;
  peak_day: string;
  peak_day_total: number;
}

const TABS = [
  "📈 Trends",
  "🗺️ States",
  "📅 Seasonality",
  "🏘️ Districts",
  "🔵 Clusters",
  "📊 Forecast",
] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("📈 Trends");
  const [selectedStates, setSelectedStates] = useState<string[]>([]);
  const [dateRange, setDateRange] = useState<[string, string]>(["", ""]);
  const [searchQuery, setSearchQuery] = useState("");

  const [dailyState, setDailyState] = useState<DailyState[]>([]);
  const [dailyNational, setDailyNational] = useState<DailyNational[]>([]);
  const [stateSummary, setStateSummary] = useState<StateSummary[]>([]);
  const [districts, setDistricts] = useState<DistrictSummary[]>([]);
  const [weekday, setWeekday] = useState<WeekdayRow[]>([]);
  const [monthly, setMonthly] = useState<MonthlyRow[]>([]);
  const [weekly, setWeekly] = useState<WeeklyRow[]>([]);
  const [quarterly, setQuarterly] = useState<any[]>([]);
  const [heatmap, setHeatmap] = useState<HeatmapRow[]>([]);
  const [stateMonthly, setStateMonthly] = useState<any[]>([]);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("data/daily_state.json").then((r) => r.json()),
      fetch("data/daily_national.json").then((r) => r.json()),
      fetch("data/states.json").then((r) => r.json()),
      fetch("data/districts.json").then((r) => r.json()),
      fetch("data/weekday.json").then((r) => r.json()),
      fetch("data/monthly.json").then((r) => r.json()),
      fetch("data/weekly.json").then((r) => r.json()),
      fetch("data/quarterly.json").then((r) => r.json()),
      fetch("data/heatmap.json").then((r) => r.json()),
      fetch("data/state_monthly.json").then((r) => r.json()),
      fetch("data/metadata.json").then((r) => r.json()),
    ]).then(
      ([ds, dn, ss, dd, wd, mo, wk, qr, hm, sm, md]) => {
        setDailyState(ds);
        setDailyNational(dn);
        setStateSummary(ss);
        setDistricts(dd);
        setWeekday(wd);
        setMonthly(mo);
        setWeekly(wk);
        setQuarterly(qr);
        setHeatmap(hm);
        setStateMonthly(sm);
        setMetadata(md);
        setSelectedStates(md.all_states);
        setDateRange([md.date_min, md.date_max]);
        setLoading(false);
      }
    );
  }, []);

  const filteredStates = useMemo(() => {
    if (!searchQuery) return selectedStates;
    return selectedStates.filter((s) =>
      s.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [selectedStates, searchQuery]);

  const filteredDaily = useMemo(() => {
    if (!dailyState.length) return [];
    const [d1, d2] = dateRange;
    return dailyState.filter(
      (r) =>
        selectedStates.includes(r.state) &&
        (!d1 || r.date >= d1) &&
        (!d2 || r.date <= d2)
    );
  }, [dailyState, selectedStates, dateRange]);

  const filteredNational = useMemo(() => {
    if (!dailyNational.length) return [];
    const [d1, d2] = dateRange;
    return dailyNational.filter(
      (r) => (!d1 || r.date >= d1) && (!d2 || r.date <= d2)
    );
  }, [dailyNational, dateRange]);

  const handleExport = useCallback(() => {
    const rows = filteredDaily.length
      ? filteredDaily
      : dailyState;
    const headers = Object.keys(rows[0] || {});
    const csv = [
      headers.join(","),
      ...rows.map((r) =>
        headers.map((h) => (r as any)[h] ?? "").join(",")
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aadhaar_data_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filteredDaily, dailyState]);

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner" />
        <p>Loading dashboard…</p>
      </div>
    );
  }

  return (
    <div className="app-layout">
      <Sidebar
        metadata={metadata!}
        selectedStates={selectedStates}
        onStatesChange={setSelectedStates}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />
      <main className="main-content">
        <header className="header">
          <div className="header-left">
            <h1>🇮🇳 Aadhaar Enrollment Analysis</h1>
            <p>
              {metadata!.num_days} days · {metadata!.total_records.toLocaleString()} records ·{" "}
              {metadata!.states} states · {metadata!.districts} districts
            </p>
          </div>
          <div className="header-actions">
            <button className="btn btn-outline" onClick={handleExport}>
              ⬇ Export CSV
            </button>
          </div>
        </header>

        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab}
              className={`tab ${activeTab === tab ? "active" : ""}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </nav>

        <KPICards filteredDaily={filteredDaily} metadata={metadata!} />

        <div className="tab-content">
          {activeTab === "📈 Trends" && (
            <TrendsTab dailyNational={filteredNational} metadata={metadata!} />
          )}
          {activeTab === "🗺️ States" && (
            <StatesTab
              stateSummary={stateSummary}
              stateMonthly={stateMonthly}
            />
          )}
          {activeTab === "📅 Seasonality" && (
            <SeasonalityTab
              weekday={weekday}
              monthly={monthly}
              weekly={weekly}
              quarterly={quarterly}
              heatmap={heatmap}
            />
          )}
          {activeTab === "🏘️ Districts" && (
            <DistrictsTab
              districts={districts}
              selectedStates={selectedStates}
            />
          )}
          {activeTab === "🔵 Clusters" && (
            <ClustersTab stateSummary={stateSummary} />
          )}
          {activeTab === "📊 Forecast" && (
            <ForecastTab dailyNational={dailyNational} />
          )}
        </div>
      </main>
    </div>
  );
}
