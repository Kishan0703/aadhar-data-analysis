import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Aadhaar Analysis", page_icon="🇮🇳", layout="wide")

STATE_MAP = {
    "West Bangal": "West Bengal", "West  Bengal": "West Bengal", "Westbengal": "West Bengal",
    "WEST BENGAL": "West Bengal", "WESTBENGAL": "West Bengal", "West bengal": "West Bengal",
    "west Bengal": "West Bengal", "West Bengli": "West Bengal", "andhra pradesh": "Andhra Pradesh",
    "ODISHA": "Odisha", "odisha": "Odisha", "Orissa": "Odisha",
    "Jammu & Kashmir": "Jammu and Kashmir", "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "Daman & Diu": "Daman and Diu", "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
}

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.rename(columns={"demo_age_17_": "demo_age_18_plus"})
    df["demo_age_5_17"] = df["demo_age_5_17"].fillna(0).astype(int)
    df["demo_age_18_plus"] = df["demo_age_18_plus"].fillna(0).astype(int)
    before = len(df)
    df = df.drop_duplicates()
    df["state"] = df["state"].replace(STATE_MAP)
    df["total"] = df["demo_age_5_17"] + df["demo_age_18_plus"]
    df["child_pct"] = np.where(df["total"] > 0, df["demo_age_5_17"] / df["total"] * 100, 0)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day_name"] = df["date"].dt.day_name()
    return df, before - len(df)

df, dupes = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🇮🇳 Aadhaar Dashboard")
st.sidebar.caption(f"~{len(df)/1e6:.1f}M records · {dupes:,} dupes removed")

all_states = sorted(df["state"].unique())
selected_states = st.sidebar.multiselect("Filter by State", all_states, default=all_states)

date_min, date_max = df["date"].min().date(), df["date"].max().date()
d1, d2 = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

mask = (df["state"].isin(selected_states)) & (df["date"] >= pd.Timestamp(d1)) & (df["date"] <= pd.Timestamp(d2))
fdf = df[mask]

st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Known Issues**")
st.sidebar.markdown("- August 2025 data missing\n- 9 variants of 'West Bengal' in raw data\n- CV = 220% (extreme volatility)")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🇮🇳 Aadhaar Enrollment Analysis")

total = fdf["total"].sum()
children = fdf["demo_age_5_17"].sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Enrollments", f"{total/1e6:.2f}M")
c2.metric("Children (5-17)", f"{children/1e6:.2f}M", f"{children/total*100:.1f}% of total")
c3.metric("States", fdf["state"].nunique())
c4.metric("Districts", fdf["district"].nunique())
c5.metric("Date Range", f"{d1.strftime('%b %Y')} – {d2.strftime('%b %Y')}")

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "🗺️ States", "📅 Seasonality", "🏘️ Districts", "🔵 Clusters"])

# ── Tab 1: Trends ──────────────────────────────────────────────────────────────
with tab1:
    daily = fdf.groupby("date").agg(total=("total", "sum"), children=("demo_age_5_17", "sum")).reset_index()
    daily["rolling7"] = daily["total"].rolling(7).mean()
    daily["rolling7_children"] = daily["children"].rolling(7).mean()

    # Anomaly detection
    daily["z"] = (daily["total"] - daily["total"].rolling(30).mean()) / daily["total"].rolling(30).std()
    anomalies = daily[daily["z"].abs() > 2.5]

    col1, col2 = st.columns([3, 1])
    with col1:
        show_anomalies = st.toggle("Highlight anomalies", value=True)
    with col2:
        metric = st.selectbox("Show", ["Total", "Children (5-17)", "Adults (18+)"], label_visibility="collapsed")

    if metric == "Children (5-17)":
        y_col, roll_col, label = "children", "rolling7_children", "Children (5-17)"
    elif metric == "Adults (18+)":
        daily["adults"] = daily["total"] - daily["children"]
        daily["rolling7_adults"] = daily["adults"].rolling(7).mean()
        y_col, roll_col, label = "adults", "rolling7_adults", "Adults (18+)"
    else:
        y_col, roll_col, label = "total", "rolling7", "Total"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily[y_col], mode="lines",
                             name=label, line=dict(color="#a8c7fa", width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily[roll_col], mode="lines",
                             name="7-day avg", line=dict(color="#4285f4", width=2.5)))
    if show_anomalies and len(anomalies):
        fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies[y_col], mode="markers",
                                 name=f"Anomaly ({len(anomalies)})", marker=dict(color="#ea4335", size=8, symbol="x")))
    fig.update_layout(title="Daily Enrollment Trend", template="plotly_white",
                      legend=dict(orientation="h", y=1.1), margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies):
        with st.expander(f"📋 {len(anomalies)} anomalous days (|z| > 2.5)"):
            st.dataframe(
                anomalies[["date", "total", "z"]].rename(columns={"z": "z_score"})
                .assign(day=anomalies["date"].dt.day_name())
                .sort_values("z_score", key=abs, ascending=False)
                .style.format({"total": "{:,.0f}", "z_score": "{:.2f}"}),
                use_container_width=True
            )

# ── Tab 2: States ──────────────────────────────────────────────────────────────
with tab2:
    state_df = fdf.groupby("state").agg(
        total=("total", "sum"),
        children=("demo_age_5_17", "sum"),
        districts=("district", "nunique"),
    ).reset_index()
    state_df["child_pct"] = state_df["children"] / state_df["total"] * 100
    state_df = state_df.sort_values("total", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.slider("Top N states", 5, len(state_df), 15)
        sort_by = st.radio("Sort by", ["Total enrollment", "Child %"])
    
    subset = state_df.head(top_n) if sort_by == "Total enrollment" else state_df.nlargest(top_n, "child_pct")

    with col2:
        if sort_by == "Total enrollment":
            fig = px.bar(subset.sort_values("total"), x="total", y="state", orientation="h",
                         color="child_pct", color_continuous_scale="RdYlGn",
                         labels={"total": "Enrollments", "child_pct": "Child %"},
                         title=f"Top {top_n} States — colored by Child %")
            fig.update_layout(yaxis_title="", coloraxis_colorbar_title="Child %",
                              template="plotly_white", margin=dict(l=10))
        else:
            fig = px.bar(subset.sort_values("child_pct"), x="child_pct", y="state", orientation="h",
                         color="total", color_continuous_scale="Blues",
                         labels={"child_pct": "Child Enrollment %", "total": "Total"},
                         title=f"Top {top_n} States — by Child Enrollment %")
            fig.update_layout(yaxis_title="", coloraxis_colorbar_title="Total",
                              template="plotly_white", margin=dict(l=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full State Table"):
        st.dataframe(
            state_df.style.format({"total": "{:,.0f}", "children": "{:,.0f}", "child_pct": "{:.1f}%"})
            .background_gradient(subset=["child_pct"], cmap="RdYlGn"),
            use_container_width=True, height=400
        )

# ── Tab 3: Seasonality ─────────────────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        # Weekday pattern
        DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wkday = fdf.groupby("day_name")["total"].sum().reindex(DAY_ORDER).reset_index()
        day_counts = fdf.groupby("day_name")["date"].nunique().reindex(DAY_ORDER)
        wkday["avg"] = wkday["total"] / day_counts.values
        colors = ["#fbbc04" if d in ("Saturday", "Sunday") else "#4285f4" for d in DAY_ORDER]

        fig = go.Figure(go.Bar(x=wkday["day_name"], y=wkday["avg"],
                               marker_color=colors, text=wkday["avg"].apply(lambda x: f"{x/1e3:.0f}K"),
                               textposition="outside"))
        fig.update_layout(title="Avg Enrollments per Day of Week<br><sup>🟡 Saturday is 2.5x higher than weekdays</sup>",
                          template="plotly_white", yaxis_title="Avg daily enrollments", margin=dict(t=70))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Monthly trend
        MONTH_ORDER = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        monthly = fdf.copy()
        monthly["month_name"] = monthly["date"].dt.month_name()
        mon = monthly.groupby("month_name").agg(total=("total","sum"), child_pct=("child_pct","mean")).reset_index()
        mon["month_name"] = pd.Categorical(mon["month_name"], MONTH_ORDER, ordered=True)
        mon = mon.sort_values("month_name")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=mon["month_name"], y=mon["total"], name="Enrollments",
                              marker_color="#4285f4", yaxis="y"))
        fig2.add_trace(go.Scatter(x=mon["month_name"], y=mon["child_pct"], name="Child %",
                                  mode="lines+markers", line=dict(color="#ea4335", width=2),
                                  yaxis="y2"))
        fig2.update_layout(
            title="Monthly Enrollments + Child %",
            template="plotly_white",
            yaxis=dict(title="Enrollments"),
            yaxis2=dict(title="Child %", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: month × weekday
    st.subheader("Enrollment Heatmap: Month × Weekday")
    heat = fdf.copy()
    heat["month_name"] = pd.Categorical(heat["date"].dt.month_name(), MONTH_ORDER, ordered=True)
    pivot = heat.groupby(["month_name", "day_name"])["total"].sum().unstack(fill_value=0).reindex(columns=DAY_ORDER)
    
    fig3 = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                     labels=dict(color="Enrollments"), title="Total Enrollments by Month & Weekday")
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Districts ───────────────────────────────────────────────────────────
with tab4:
    dist = fdf.groupby(["state", "district"]).agg(
        total=("total", "sum"),
        child_pct=("child_pct", "mean"),
    ).reset_index().sort_values("total", ascending=False)

    col1, col2 = st.columns([1, 2])
    with col1:
        top_d = st.slider("Top N districts", 10, 50, 20)
        state_filter = st.multiselect("Filter by State", sorted(dist["state"].unique()),
                                       placeholder="All states")
    
    d_subset = dist[dist["state"].isin(state_filter)] if state_filter else dist
    d_subset = d_subset.head(top_d)

    with col2:
        fig = px.bar(d_subset.sort_values("total"), x="total", y="district",
                     orientation="h", color="state",
                     hover_data={"child_pct": ":.1f"},
                     labels={"total": "Enrollments", "district": ""},
                     title=f"Top {top_d} Districts",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_white", height=max(400, top_d * 25),
                          legend=dict(orientation="v"), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: Clusters ────────────────────────────────────────────────────────────
with tab5:
    state_cl = df.groupby("state").agg(
        total=("total","sum"), children=("demo_age_5_17","sum"),
        districts=("district","nunique"), pincodes=("pincode","nunique"),
    ).reset_index()
    state_cl["child_pct"] = state_cl["children"] / state_cl["total"] * 100

    features = ["total", "children", "child_pct", "districts"]
    scaler = StandardScaler()
    X = scaler.fit_transform(state_cl[features])
    
    n_clusters = st.slider("Number of clusters", 2, 6, 4)
    state_cl["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X).astype(str)

    fig = px.scatter(state_cl, x="total", y="child_pct", color="cluster",
                     size="total", size_max=60, hover_name="state",
                     hover_data={"total": ":,.0f", "child_pct": ":.1f", "districts": True},
                     labels={"total": "Total Enrollments", "child_pct": "Child %"},
                     title="State Clusters: Volume vs Child Enrollment %",
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    for c in sorted(state_cl["cluster"].unique()):
        sub = state_cl[state_cl["cluster"] == c]
        with st.expander(f"Cluster {c} · {len(sub)} states · avg {sub['total'].mean()/1e6:.1f}M enrolments · {sub['child_pct'].mean():.1f}% child"):
            st.write(", ".join(sorted(sub["state"].tolist())))
