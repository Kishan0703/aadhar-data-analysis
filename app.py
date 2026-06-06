import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Aadhaar Demographics Analysis",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stMetric > label {
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    .stMetric > div {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    
    h2, h3 {
        color: #2d3748;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .insight-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
    
    .insight-text {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .danger-box {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f7fafc;
        border-radius: 10px 10px 0 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    .stSidebar .stMultiSelect label,
    .stSidebar .stDateInput label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# State name standardization function
def standardize_state_names(df):
    """Fix inconsistent state names"""
    state_mapping = {
        'West Bangal': 'West Bengal',
        'West  Bengal': 'West Bengal',
        'Westbengal': 'West Bengal',
        'WEST BENGAL': 'West Bengal',
        'WESTBENGAL': 'West Bengal',
        'West bengal': 'West Bengal',
        'west Bengal': 'West Bengal',
        'West Bengli': 'West Bengal',
        'andhra pradesh': 'Andhra Pradesh',
        'ODISHA': 'Odisha',
        'odisha': 'Odisha',
        'Orissa': 'Odisha',
        'Jammu & Kashmir': 'Jammu and Kashmir',
        'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
        'Daman & Diu': 'Daman and Diu',
        'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands'
    }
    df['state'] = df['state'].replace(state_mapping)
    return df

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv', dtype={
        'state': 'category',
        'district': 'category',
        'pincode': 'Int32',
        'demo_age_5_17': 'Int32',
        'demo_age_17_': 'Int32'
    })
    df['pincode'] = df['pincode'].fillna(0).astype(int)
    df['demo_age_5_17'] = df['demo_age_5_17'].fillna(0).astype(int)
    df['demo_age_17_'] = df['demo_age_17_'].fillna(0).astype(int)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df.rename(columns={'demo_age_17_': 'demo_age_18_plus'})
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    
    # Standardize state names
    df = standardize_state_names(df)
    
    df['total_enrolments'] = df['demo_age_5_17'] + df['demo_age_18_plus']
    df['child_pct'] = np.where(df['total_enrolments'] > 0,
                               (df['demo_age_5_17'] / df['total_enrolments']) * 100, 0)
    df['month_name'] = df['date'].dt.month_name()
    df['day_name'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5,6]).astype(int)
    
    return df, before - after

@st.cache_data
def compute_aggregates(df):
    # Daily
    daily = df.groupby('date').agg({
        'demo_age_5_17': 'sum',
        'demo_age_18_plus': 'sum',
        'total_enrolments': 'sum'
    }).reset_index().sort_values('date')
    daily['child_pct'] = (daily['demo_age_5_17'] / daily['total_enrolments']) * 100

    # State summary
    state_summary = df.groupby('state').agg({
        'demo_age_5_17': 'sum',
        'demo_age_18_plus': 'sum',
        'total_enrolments': 'sum',
        'district': 'nunique',
        'pincode': 'nunique'
    }).reset_index()
    state_summary.columns = ['state', 'total_children', 'total_adults', 'total_enrolments', 'num_districts', 'num_pincodes']
    state_summary['child_pct'] = (state_summary['total_children'] / state_summary['total_enrolments']) * 100
    state_summary = state_summary.sort_values('total_enrolments', ascending=False).reset_index(drop=True)

    # Monthly
    monthly = df.groupby(['month_name']).agg({
        'total_enrolments': 'sum',
        'demo_age_5_17': 'sum'
    }).reset_index()
    monthly['child_pct'] = (monthly['demo_age_5_17'] / monthly['total_enrolments']) * 100

    # District
    district_summary = df.groupby(['state','district']).agg({
        'total_enrolments': 'sum',
        'demo_age_5_17': 'sum'
    }).reset_index()
    district_summary['child_pct'] = (district_summary['demo_age_5_17'] / district_summary['total_enrolments']) * 100
    district_summary = district_summary.sort_values('total_enrolments', ascending=False).reset_index(drop=True)

    # Clusters
    cluster_features = state_summary[['total_enrolments','total_children','total_adults','num_districts','num_pincodes','child_pct']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    state_summary['cluster'] = kmeans.fit_predict(scaled)

    return daily, state_summary, monthly, district_summary

# Load data
try:
    df, duplicates_removed = load_data()
    daily, state_summary, monthly, district_summary = compute_aggregates(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### 🎛️ Filters")
all_states = sorted(df['state'].unique().tolist())
selected_states = st.sidebar.multiselect("Select States", all_states, default=all_states[:10])

date_min = df['date'].min().date()
date_max = df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)

if len(date_range) == 2:
    df_filtered = df[(df['state'].isin(selected_states)) & 
                     (df['date'] >= pd.Timestamp(date_range[0])) & 
                     (df['date'] <= pd.Timestamp(date_range[1]))]
else:
    df_filtered = df[df['state'].isin(selected_states)]

# Header
st.title("🇮🇳 Aadhaar Demographics Analysis")
st.markdown("**Deep insights from 36M+ enrollment records across India**")
st.markdown("---")

# KEY INSIGHTS SECTION (Important!)
st.markdown("## 🔍 Key Insights & Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">⚠️ Critical Data Quality Issues</div>
        <div class="insight-text">
        • <b>9 variations</b> of "West Bengal" found in data<br>
        • <b>August 2025</b> data completely missing<br>
        • <b>21.6%</b> duplicate records removed<br>
        • Extreme volatility: CV = <b>220%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">🏆 Geographic Dominance</div>
        <div class="insight-text">
        • <b>Uttar Pradesh</b> leads with 17.7% of all enrolments<br>
        • Top 3 states (UP, Maharashtra, Bihar) = <b>38%</b> of total<br>
        • Thane & Pune are the top districts<br>
        • <b>65 unique states</b> (including variations)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">📊 Counter-Intuitive Weekend Pattern</div>
        <div class="insight-text">
        • Saturdays show <b>2.5x higher</b> enrolments than average<br>
        • Weekend average: <b>542K</b> vs Weekday: <b>318K</b><br>
        • Possible: Special camps or data batching<br>
        • <b>Sunday</b> has lowest enrolments
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">👶 Child Enrollment Concern</div>
        <div class="insight-text">
        • Only <b>9.8%</b> children (5-17 years) enrolled<br>
        • <b>Maharashtra</b> lowest at 5.4% child enrolment<br>
        • <b>Ladakh</b> highest at 23.5%<br>
        • States with high volume have low child %
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# KPIs
st.markdown("## 📊 Key Metrics")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_enrolments = df_filtered['total_enrolments'].sum()
total_children = df_filtered['demo_age_5_17'].sum()
total_adults = df_filtered['demo_age_18_plus'].sum()

with kpi1:
    st.metric("Total Enrollments", f"{total_enrolments:,.0f}")
with kpi2:
    st.metric("Records Analyzed", f"{len(df_filtered):,.0f}")
with kpi3:
    st.metric("States Covered", f"{df_filtered['state'].nunique()}")
with kpi4:
    st.metric("Districts", f"{df_filtered['district'].nunique()}")
with kpi5:
    child_pct = (total_children / total_enrolments * 100) if total_enrolments > 0 else 0
    st.metric("Child Enrollment %", f"{child_pct:.1f}%", f"{child_pct - 9.8:.1f}% vs avg")

st.markdown("---")

# Data Quality Alert
if duplicates_removed > 0:
    st.markdown(f"""
    <div class="warning-box">
        <b>⚠️ Data Cleaning Applied:</b> Removed {duplicates_removed:,} duplicate records ({duplicates_removed/(len(df)+duplicates_removed)*100:.1f}% of data). 
        State names standardized (9 variations of "West Bengal" fixed). August 2025 data is missing from source.
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "🗺️ State Analysis", "📅 Trends & Seasonality", "🏘️ Geography", "🔵 Clusters", "⚠️ Anomalies"
])

with tab1:
    st.subheader("Enrollment Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily trend
        fig = px.line(daily, x='date', y='total_enrolments',
                      title="Daily Enrollment Trend",
                      labels={'total_enrolments': 'Enrollments', 'date': 'Date'},
                      template='plotly_white')
        fig.add_scatter(x=daily['date'], y=daily['total_enrolments'].rolling(7).mean(),
                        mode='lines', name='7-day MA', line=dict(color='#FF6B6B', width=3))
        fig.update_traces(line=dict(width=2), selector=dict(name='Daily Total Enrollment'))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age group breakdown
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Children (5-17)', x=daily['date'], y=daily['demo_age_5_17'],
                              marker_color='#FF6B6B'))
        fig2.add_trace(go.Bar(name='Adults (18+)', x=daily['date'], y=daily['demo_age_18_plus'],
                              marker_color='#4ECDC4'))
        fig2.update_layout(
            barmode='stack',
            title="Daily Enrollment by Age Group",
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font_size=16
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly trend
    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    monthly['month_name'] = pd.Categorical(monthly['month_name'], categories=month_order, ordered=True)
    monthly = monthly.sort_values('month_name')
    
    fig3 = px.bar(monthly, x='month_name', y='total_enrolments',
                  title="Monthly Enrollment Distribution",
                  labels={'total_enrolments': 'Enrollments', 'month_name': 'Month'},
                  template='plotly_white',
                  color='total_enrolments',
                  color_continuous_scale='Viridis')
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        title_font_size=16
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("State-Level Analysis")
    
    top_n = st.slider("Number of States to Display", 5, 30, 15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_states = state_summary.head(top_n)
        fig = px.bar(top_states, x='total_enrolments', y='state', orientation='h',
                     title=f"Top {top_n} States by Enrollment",
                     labels={'total_enrolments': 'Enrollments', 'state': 'State'},
                     template='plotly_white',
                     color='total_enrolments',
                     color_continuous_scale='Blues')
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600,
                         plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(family='Inter', size=12))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.bar(top_states, x='child_pct', y='state', orientation='h',
                      title=f"Child Enrollment % (Top {top_n} States)",
                      labels={'child_pct': 'Child %', 'state': 'State'},
                      template='plotly_white',
                      color='child_pct',
                      color_continuous_scale='RdYlGn')
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=600,
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12))
        st.plotly_chart(fig2, use_container_width=True)
    
    # State table
    st.subheader("Complete State Summary")
    st.dataframe(
        state_summary[['state','total_enrolments','total_children','total_adults','child_pct','num_districts','num_pincodes']].style.format({
            'total_enrolments': '{:,.0f}',
            'total_children': '{:,.0f}',
            'total_adults': '{:,.0f}',
            'child_pct': '{:.2f}%',
            'num_districts': '{:.0f}',
            'num_pincodes': '{:.0f}'
        }).background_gradient(subset=['child_pct'], cmap='RdYlGn'),
        use_container_width=True,
        height=500
    )

with tab3:
    st.subheader("Temporal Patterns & Seasonality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekday pattern
        weekly = df.groupby('day_name').agg({'total_enrolments': 'sum'}).reset_index()
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        weekly['day_name'] = pd.Categorical(weekly['day_name'], categories=day_order, ordered=True)
        weekly = weekly.sort_values('day_name')
        
        # Add day count for average
        day_counts = df.groupby('day_name')['date'].nunique().reset_index()
        day_counts.columns = ['day_name', 'num_days']
        weekly = weekly.merge(day_counts, on='day_name')
        weekly['avg_per_day'] = weekly['total_enrolments'] / weekly['num_days']
        
        colors = ['#4ECDC4' if d not in ['Saturday', 'Sunday'] else '#FF6B6B' for d in day_order]
        
        fig = px.bar(weekly, x='day_name', y='avg_per_day',
                      title="Average Daily Enrollment by Weekday",
                      labels={'avg_per_day': 'Avg Enrollments/Day', 'day_name': 'Day'},
                      template='plotly_white')
        fig.update_traces(marker_color=['#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#FF6B6B', '#FF6B6B'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">⚠️ Weekend Anomaly</div>
            <div class="insight-text">
            Saturday shows <b>2.5x higher</b> average enrolments than weekdays! 
            This contradicts typical expectations of lower weekend activity.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Child % by month
        fig2 = px.line(monthly, x='month_name', y='child_pct',
                       title="Child Enrollment % by Month",
                       labels={'child_pct': 'Child %', 'month_name': 'Month'},
                       template='plotly_white',
                       markers=True)
        fig2.update_traces(line=dict(color='#FF6B6B', width=3), marker=dict(size=10))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font_size=16
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Volatility stats
        cv = daily['total_enrolments'].std() / daily['total_enrolments'].mean() * 100
        st.markdown(f"""
        <div class="danger-box">
            <b>📊 Volatility Analysis</b><br>
            Coefficient of Variation: <b>{cv:.1f}%</b> (Extreme)<br>
            Peak Day: <b>{daily['total_enrolments'].max():,}</b><br>
            Lowest Day: <b>{daily['total_enrolments'].min():,}</b><br>
            Ratio: <b>{daily['total_enrolments'].max()/daily['total_enrolments'].min():.0f}x</b>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.subheader("District & Pincode Analysis")
    
    top_districts = district_summary.head(25)
    
    fig = px.bar(top_districts, x='total_enrolments', y='district', color='state',
                 orientation='h', title="Top 25 Districts by Enrollment",
                 labels={'total_enrolments': 'Enrollments', 'district': 'District'},
                 template='plotly_white',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(yaxis=dict(autorange="reversed"), height=800,
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)',
                     font=dict(family='Inter', size=12))
    st.plotly_chart(fig, use_container_width=True)
    
    # Pincode analysis
    st.subheader("Top 15 Pincodes")
    pincode_top = df.groupby('pincode').agg({
        'total_enrolments': 'sum',
        'state': 'first',
        'district': 'first'
    }).reset_index().sort_values('total_enrolments', ascending=False).head(15)
    
    st.dataframe(
        pincode_top.style.format({'total_enrolments': '{:,.0f}'}),
        use_container_width=True
    )

with tab5:
    st.subheader("State Clustering Analysis")
    
    fig = px.scatter(state_summary, x='total_enrolments', y='child_pct',
                     color='cluster', size='total_enrolments',
                     hover_name='state', title="State Clusters: Enrollment vs Child %",
                     labels={'total_enrolments': 'Total Enrollments', 'child_pct': 'Child %'},
                     template='plotly_white',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Cluster Profiles")
    
    for c in sorted(state_summary['cluster'].unique()):
        subset = state_summary[state_summary['cluster'] == c]
        with st.expander(f"Cluster {c} ({len(subset)} states) - Avg: {subset['total_enrolments'].mean():,.0f} enrolments, {subset['child_pct'].mean():.1f}% child"):
            st.write(', '.join(subset['state'].tolist()))

with tab6:
    st.subheader("Anomaly Detection")
    
    # Daily anomalies using rolling statistics
    daily['rolling_mean'] = daily['total_enrolments'].rolling(7).mean()
    daily['rolling_std'] = daily['total_enrolments'].rolling(7).std()
    daily['z_score'] = (daily['total_enrolments'] - daily['rolling_mean']) / daily['rolling_std']
    daily['anomaly'] = daily['z_score'].abs() > 2
    
    anomalies = daily[daily['anomaly']].copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['total_enrolments'],
                              mode='lines', name='Normal', line=dict(color='#4ECDC4', width=2)))
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['total_enrolments'],
                                  mode='markers', name='Anomaly', 
                                  marker=dict(color='#FF6B6B', size=12, symbol='x')))
    fig.update_layout(
        title="Daily Enrollment with Anomalies (|Z-score| > 2)",
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if len(anomalies) > 0:
        st.markdown(f"""
        <div class="danger-box">
            <b>⚠️ {len(anomalies)} Anomalous Days Detected</b><br>
            These days show enrollment patterns significantly different from the 7-day rolling average.
        </div>
        """, unsafe_allow_html=True)
        
        anomalies['day_name'] = anomalies['date'].dt.day_name()
        st.dataframe(
            anomalies[['date', 'day_name', 'total_enrolments', 'z_score']].style.format({
                'z_score': '{:.2f}',
                'total_enrolments': '{:,.0f}'
            }),
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="success-box">
            <b>✅ No significant anomalies detected</b><br>
            All daily enrollment values are within normal range (Z-score < 2).
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <b>Built with ❤️ using Streamlit</b><br>
    Data: Aadhaar Demographics (March - December 2025) | 36M+ Records Analyzed
</div>
""", unsafe_allow_html=True)
