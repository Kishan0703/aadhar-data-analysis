"""
Preprocess Aadhaar CSV into rich aggregated JSON files for the enterprise dashboard.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PUBLIC_DIR = Path(__file__).resolve().parent / "public" / "data"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

STATE_MAP = {
    "West Bangal": "West Bengal", "West  Bengal": "West Bengal", "Westbengal": "West Bengal",
    "WEST BENGAL": "West Bengal", "WESTBENGAL": "West Bengal", "West bengal": "West Bengal",
    "west Bengal": "West Bengal", "West Bengli": "West Bengal", "andhra pradesh": "Andhra Pradesh",
    "ODISHA": "Odisha", "odisha": "Odisha", "Orissa": "Odisha",
    "Jammu & Kashmir": "Jammu and Kashmir", "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "Daman & Diu": "Daman and Diu", "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
}

# 2025 approximate populations (in crores / 10M) — scaled to match enrollment scale
POPULATION = {
    "Andhra Pradesh": 5.5, "Arunachal Pradesh": 0.15, "Assam": 3.6, "Bihar": 12.8,
    "Chhattisgarh": 3.0, "Goa": 0.15, "Gujarat": 7.0, "Haryana": 3.0,
    "Himachal Pradesh": 0.75, "Jharkhand": 3.9, "Karnataka": 7.0, "Kerala": 3.6,
    "Madhya Pradesh": 8.5, "Maharashtra": 12.5, "Manipur": 0.33, "Meghalaya": 0.36,
    "Mizoram": 0.12, "Nagaland": 0.22, "Odisha": 4.7, "Punjab": 3.0,
    "Rajasthan": 8.0, "Sikkim": 0.07, "Tamil Nadu": 7.6, "Telangana": 3.9,
    "Tripura": 0.42, "Uttar Pradesh": 23.5, "Uttarakhand": 1.2, "West Bengal": 9.8,
    "Andaman and Nicobar Islands": 0.04, "Chandigarh": 0.12, "Dadra and Nagar Haveli": 0.06,
    "Daman and Diu": 0.06, "Delhi": 1.9, "Jammu and Kashmir": 1.4,
    "Ladakh": 0.03, "Lakshadweep": 0.01, "Puducherry": 0.14,
}

print("Loading data...")
df = pd.read_csv(DATA_DIR / "data.csv")
print(f"  Loaded {len(df):,} rows")

df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df = df.rename(columns={"demo_age_17_": "demo_age_18_plus"})
df["demo_age_5_17"] = df["demo_age_5_17"].fillna(0).astype(int)
df["demo_age_18_plus"] = df["demo_age_18_plus"].fillna(0).astype(int)

before = len(df)
df = df.drop_duplicates()
df["state"] = df["state"].replace(STATE_MAP)
print(f"  Removed {before - len(df):,} dupes")

df["total"] = df["demo_age_5_17"] + df["demo_age_18_plus"]
df["child_pct"] = np.where(df["total"] > 0, df["demo_age_5_17"] / df["total"] * 100, 0)
df["month"] = df["date"].dt.to_period("M").astype(str)
df["day_name"] = df["date"].dt.day_name()
df["month_name"] = df["date"].dt.month_name()
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
df["year"] = df["date"].dt.year

# ── 1. Daily per-state data (for filtering) ──
print("Aggregating daily per-state data...")
daily_state = df.groupby(["date", "state"]).agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
    adults=("demo_age_18_plus", "sum"),
    districts=("district", "nunique"),
    pincodes=("pincode", "nunique"),
).reset_index()
daily_state["date"] = daily_state["date"].dt.strftime("%Y-%m-%d")
daily_state["child_pct"] = np.where(
    daily_state["total"] > 0,
    daily_state["children"] / daily_state["total"] * 100,
    0,
)
daily_state["population"] = daily_state["state"].map(POPULATION).fillna(0.1)
daily_state["per_capita"] = np.where(
    daily_state["population"] > 0,
    daily_state["total"] / daily_state["population"],
    0,
)
daily_state.to_json(PUBLIC_DIR / "daily_state.json", orient="records", date_format="iso")
print(f"  → daily_state.json ({len(daily_state):,} rows)")

# ── 2. Daily national totals ──
daily_nation = daily_state.groupby("date").agg(
    total=("total", "sum"),
    children=("children", "sum"),
    adults=("adults", "sum"),
).reset_index()
daily_nation["child_pct"] = np.where(
    daily_nation["total"] > 0,
    daily_nation["children"] / daily_nation["total"] * 100,
    0,
)
daily_nation["rolling7"] = daily_nation["total"].rolling(7).mean().fillna(0).round(0)
daily_nation["rolling30"] = daily_nation["total"].rolling(30).mean().fillna(0).round(0)
daily_nation["rolling7_children"] = daily_nation["children"].rolling(7).mean().fillna(0).round(0)
daily_nation["cumulative"] = daily_nation["total"].cumsum()
daily_nation["z"] = (
    (daily_nation["total"] - daily_nation["total"].rolling(30).mean())
    / daily_nation["total"].rolling(30).std()
).fillna(0)
daily_nation["anomaly"] = daily_nation["z"].abs() > 2.5
daily_nation["dow"] = pd.to_datetime(daily_nation["date"]).dt.day_name()
# WoW growth
daily_nation["wow_growth"] = daily_nation["total"].pct_change(periods=7).fillna(0) * 100
# MoM growth
monthly_totals = daily_nation.copy()
monthly_totals["month"] = pd.to_datetime(monthly_totals["date"]).dt.to_period("M")
mom = monthly_totals.groupby("month")["total"].sum().pct_change().fillna(0) * 100
mom_map = mom.to_dict()
daily_nation["mom_label"] = pd.to_datetime(daily_nation["date"]).dt.to_period("M").astype(str)
daily_nation["mom_growth"] = daily_nation["mom_label"].map(mom_map).fillna(0)

daily_nation.to_json(PUBLIC_DIR / "daily_national.json", orient="records", date_format="iso")
print(f"  → daily_national.json ({len(daily_nation):,} rows)")

# ── 3. State summary ──
state_summary = df.groupby("state").agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
    adults=("demo_age_18_plus", "sum"),
    districts=("district", "nunique"),
    pincodes=("pincode", "nunique"),
    records=("total", "count"),
).reset_index()
state_summary["child_pct"] = np.where(
    state_summary["total"] > 0,
    state_summary["children"] / state_summary["total"] * 100,
    0,
)
state_summary["adult_pct"] = np.where(
    state_summary["total"] > 0,
    state_summary["adults"] / state_summary["total"] * 100,
    0,
)
state_summary["population"] = state_summary["state"].map(POPULATION).fillna(0.1)
state_summary["per_capita"] = np.where(
    state_summary["population"] > 0,
    state_summary["total"] / state_summary["population"],
    0,
)
state_summary["avg_per_district"] = np.where(
    state_summary["districts"] > 0,
    state_summary["total"] / state_summary["districts"],
    0,
)
# National share
total_enroll = state_summary["total"].sum()
state_summary["share_pct"] = state_summary["total"] / total_enroll * 100
# Child-to-adult ratio
state_summary["child_adult_ratio"] = np.where(
    state_summary["adults"] > 0,
    state_summary["children"] / state_summary["adults"],
    0,
)

state_summary = state_summary.sort_values("total", ascending=False).reset_index(drop=True)

# Per-state monthly trend
state_monthly = df.groupby(["state", "month_name"]).agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
).reset_index()
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
state_monthly["month_order"] = state_monthly["month_name"].map(
    {m: i for i, m in enumerate(month_order)}
)
state_monthly = state_monthly.sort_values(["state", "month_order"]).reset_index(drop=True)
state_monthly["child_pct"] = np.where(
    state_monthly["total"] > 0,
    state_monthly["children"] / state_monthly["total"] * 100,
    0,
)
state_monthly.to_json(PUBLIC_DIR / "state_monthly.json", orient="records")
state_summary.to_json(PUBLIC_DIR / "states.json", orient="records")
print(f"  → states.json ({len(state_summary)} states)")

# ── 4. District summary ──
district_summary = df.groupby(["state", "district"]).agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
    adults=("demo_age_18_plus", "sum"),
    records=("total", "count"),
    pincodes=("pincode", "nunique"),
).reset_index()
district_summary["child_pct"] = np.where(
    district_summary["total"] > 0,
    district_summary["children"] / district_summary["total"] * 100,
    0,
)

# Rank within state
district_summary["state_rank"] = district_summary.groupby("state")["total"].rank(
    ascending=False, method="dense"
).astype(int)

district_summary = district_summary.sort_values("total", ascending=False).reset_index(drop=True)
district_summary.to_json(PUBLIC_DIR / "districts.json", orient="records")
print(f"  → districts.json ({len(district_summary):,} districts)")

# ── 5. Seasonality ──
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

wkday = df.groupby("day_name").agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
    transactions=("date", "nunique"),
).reindex(DAY_ORDER).reset_index()
day_counts = df.groupby("day_name")["date"].nunique().reindex(DAY_ORDER)
wkday["avg"] = (wkday["total"] / day_counts.values).fillna(0).round(0)
wkday["child_pct"] = np.where(
    wkday["total"] > 0,
    wkday["children"] / wkday["total"] * 100,
    0,
)
wkday.to_json(PUBLIC_DIR / "weekday.json", orient="records")

# Monthly
monthly = df.groupby("month_name").agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
    adults=("demo_age_18_plus", "sum"),
    records=("total", "count"),
    districts=("district", "nunique"),
).reset_index()
monthly["month_name"] = pd.Categorical(monthly["month_name"], MONTH_ORDER, ordered=True)
monthly = monthly.sort_values("month_name").reset_index(drop=True)
monthly["child_pct"] = np.where(monthly["total"] > 0, monthly["children"] / monthly["total"] * 100, 0)
monthly["share_pct"] = monthly["total"] / monthly["total"].sum() * 100
monthly["cumulative"] = monthly["total"].cumsum()
monthly.to_json(PUBLIC_DIR / "monthly.json", orient="records")

# Week-by-week
weekly = df.groupby(["year", "week"]).agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
).reset_index()
weekly["label"] = weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2)
weekly["child_pct"] = np.where(weekly["total"] > 0, weekly["children"] / weekly["total"] * 100, 0)
weekly.to_json(PUBLIC_DIR / "weekly.json", orient="records")

# Quarter
quarterly = df.groupby("quarter").agg(
    total=("total", "sum"),
    children=("demo_age_5_17", "sum"),
).reset_index()
quarterly["child_pct"] = np.where(quarterly["total"] > 0, quarterly["children"] / quarterly["total"] * 100, 0)
quarterly.to_json(PUBLIC_DIR / "quarterly.json", orient="records")

# Heatmap
heat = df.groupby(["month_name", "day_name"]).agg(total=("total", "sum")).reset_index()
heat["month_name"] = pd.Categorical(heat["month_name"], MONTH_ORDER, ordered=True)
heat = heat.sort_values(["month_name", "day_name"])
heat.to_json(PUBLIC_DIR / "heatmap.json", orient="records")
print(f"  → seasonality files (weekday/weekly/monthly/quarterly/heatmap)")

# ── 6. Summary stats for distribution analysis ──
peak_idx = int(daily_nation["total"].idxmax())
dist_stats = {
    "daily_mean": float(round(daily_nation["total"].mean(), 0)),
    "daily_median": float(round(daily_nation["total"].median(), 0)),
    "daily_std": float(round(daily_nation["total"].std(), 0)),
    "daily_min": float(round(daily_nation["total"].min(), 0)),
    "daily_max": float(round(daily_nation["total"].max(), 0)),
    "daily_cv": float(round(daily_nation["total"].std() / daily_nation["total"].mean() * 100, 1)),
    "total_enrollments": int(total_enroll),
    "total_records": int(len(df)),
    "total_children": int(df["demo_age_5_17"].sum()),
    "total_adults": int(df["demo_age_18_plus"].sum()),
    "states": int(df["state"].nunique()),
    "districts": int(df["district"].nunique()),
    "pincodes": int(df["pincode"].nunique()),
    "date_min": str(df["date"].min().strftime("%Y-%m-%d")),
    "date_max": str(df["date"].max().strftime("%Y-%m-%d")),
    "num_days": int(daily_nation["date"].nunique()),
    "dupes_removed": int(before - len(df)),
    "all_states": sorted(df["state"].unique().tolist()),
    "child_pct_national": float(round(df["demo_age_5_17"].sum() / total_enroll * 100, 1)),
    "peak_day": str(daily_nation.iloc[peak_idx]["date"]),
    "peak_day_total": int(daily_nation["total"].max()),
}
with open(PUBLIC_DIR / "metadata.json", "w") as f:
    json.dump(dist_stats, f, indent=2)

# ── 7. Top/bottom state rankings (quick reference) ──
rankings = {
    "top_states": state_summary.head(10)[["state", "total", "child_pct", "per_capita", "share_pct"]].to_dict("records"),
    "bottom_states": state_summary.tail(10)[["state", "total", "child_pct", "per_capita", "share_pct"]].to_dict("records"),
    "top_child_pct": state_summary.nlargest(10, "child_pct")[["state", "child_pct", "total"]].to_dict("records"),
    "top_per_capita": state_summary.nlargest(10, "per_capita")[["state", "per_capita", "total"]].to_dict("records"),
}
with open(PUBLIC_DIR / "rankings.json", "w") as f:
    json.dump(rankings, f, indent=2)

print(f"  → rankings.json + metadata.json")
print("\nDone. All data files written to public/data/")
