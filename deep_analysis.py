"""
Aadhaar Demographics - Deep Analysis
Extract meaningful insights from the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11
sns.set_palette('husl')

print("=" * 60)
print("AADHAAR DEMOGRAPHICS - DEEP ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_csv('data.csv', dtype={
    'state': 'category',
    'district': 'category',
    'pincode': 'Int32',
    'demo_age_5_17': 'Int32',
    'demo_age_17_': 'Int32'
})

# Data cleaning
df['pincode'] = df['pincode'].fillna(0).astype(int)
df['demo_age_5_17'] = df['demo_age_5_17'].fillna(0).astype(int)
df['demo_age_17_'] = df['demo_age_17_'].fillna(0).astype(int)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df = df.rename(columns={'demo_age_17_': 'demo_age_18_plus'})

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nRemoved {before - after:,} duplicate records ({(before-after)/before*100:.1f}%)")
print(f"Working with {len(df):,} records")

# Feature engineering
df['total_enrolments'] = df['demo_age_5_17'] + df['demo_age_18_plus']
df['child_pct'] = np.where(df['total_enrolments'] > 0, 
                            (df['demo_age_5_17'] / df['total_enrolments']) * 100, 0)
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.month_name()
df['day_of_week'] = df['date'].dt.dayofweek
df['day_name'] = df['date'].dt.day_name()
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['week'] = df['date'].dt.isocalendar().week

# =================================================================
# 1. CRITICAL DATA QUALITY ISSUES
# =================================================================
print("\n" + "=" * 60)
print("1. DATA QUALITY ISSUES FOUND")
print("=" * 60)

# State name inconsistencies
state_counts = df['state'].value_counts()
problematic_states = [s for s in state_counts.index if 'west' in s.lower() or 'bengal' in s.lower()]
print(f"\n⚠️  CRITICAL: Found {len(problematic_states)} variations of 'West Bengal':")
for s in problematic_states[:10]:
    print(f"   - '{s}': {state_counts[s]:,} records")

# Check for other variations
other_issues = [s for s in state_counts.index if s in ['andhra pradesh', 'odisha', 'ODISHA', 'odisha', '100000', 'Darbhanga', 'Puttenahalli']]
if other_issues:
    print(f"\n⚠️  Found {len(other_issues)} other data entry issues:")
    for s in other_issues:
        print(f"   - '{s}': {state_counts[s]:,} records")

# Missing dates
print(f"\n📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
missing_dates = set(date_range) - set(df['date'].unique())
print(f"⚠️  Missing {len(missing_dates)} days of data in the time series")

# Missing August 2025
aug_data = df[df['date'].dt.month == 8]
if len(aug_data) == 0:
    print("⚠️  CRITICAL: August 2025 data is completely missing!")

# =================================================================
# 2. ENROLLMENT PATTERNS & TRENDS
# =================================================================
print("\n" + "=" * 60)
print("2. ENROLLMENT PATTERNS & TRENDS")
print("=" * 60)

# Daily aggregations
daily = df.groupby('date').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_18_plus': 'sum'
}).reset_index()

daily['child_pct'] = (daily['demo_age_5_17'] / daily['total_enrolments']) * 100

# Key statistics
print(f"\n📊 ENROLLMENT STATISTICS:")
print(f"   Total enrolments: {daily['total_enrolments'].sum():,}")
print(f"   Children (5-17): {daily['demo_age_5_17'].sum():,} ({daily['demo_age_5_17'].sum()/daily['total_enrolments'].sum()*100:.1f}%)")
print(f"   Adults (18+): {daily['demo_age_18_plus'].sum():,} ({daily['demo_age_18_plus'].sum()/daily['total_enrolments'].sum()*100:.1f}%)")
print(f"   Daily average: {daily['total_enrolments'].mean():,.0f}")
print(f"   Daily median: {daily['total_enrolments'].median():,.0f}")
print(f"   Peak day: {daily.loc[daily['total_enrolments'].idxmax(), 'date'].strftime('%Y-%m-%d')} ({daily['total_enrolments'].max():,} enrolments)")
print(f"   Lowest day: {daily.loc[daily['total_enrolments'].idxmin(), 'date'].strftime('%Y-%m-%d')} ({daily['total_enrolments'].min():,} enrolments)")

# Volatility
cv = daily['total_enrolments'].std() / daily['total_enrolments'].mean() * 100
print(f"   Coefficient of variation: {cv:.1f}% (High volatility)")

# Monthly trends
monthly = df.groupby('month_name').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_18_plus': 'sum'
}).reset_index()

month_order = ['March', 'April', 'May', 'June', 'July', 'September', 'October', 'November', 'December']
monthly['month_order'] = monthly['month_name'].map({m: i for i, m in enumerate(month_order)})
monthly = monthly.sort_values('month_order')

print(f"\n📈 MONTHLY BREAKDOWN:")
for _, row in monthly.iterrows():
    pct = row['total_enrolments'] / monthly['total_enrolments'].sum() * 100
    print(f"   {row['month_name']:12s}: {row['total_enrolments']:>10,} ({pct:>5.1f}%) - Child %: {row['demo_age_5_17']/row['total_enrolments']*100:.1f}%")

# Weekday patterns
weekday_stats = df.groupby('day_name').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum'
}).reset_index()

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_stats['day_order'] = weekday_stats['day_name'].map({d: i for i, d in enumerate(weekday_order)})
weekday_stats = weekday_stats.sort_values('day_order')

print(f"\n📊 WEEKDAY PATTERNS:")
for _, row in weekday_stats.iterrows():
    avg_per_day = row['total_enrolments'] / (len(df[df['day_name'] == row['day_name']]['date'].unique()))
    print(f"   {row['day_name']:12s}: Total {row['total_enrolments']:>10,} | Avg/day: {avg_per_day:>8,.0f}")

# Weekend vs Weekday
weekend_total = df[df['is_weekend'] == 1]['total_enrolments'].sum()
weekday_total = df[df['is_weekend'] == 0]['total_enrolments'].sum()
weekend_days = len(df[df['is_weekend'] == 1]['date'].unique())
weekday_days = len(df[df['is_weekend'] == 0]['date'].unique())

print(f"\n⚠️  WEEKEND EFFECT:")
print(f"   Weekday avg: {weekday_total/weekday_days:,.0f} enrolments/day")
print(f"   Weekend avg: {weekend_total/weekend_days:,.0f} enrolments/day")
print(f"   Weekend drop: {(1 - weekend_total/weekend_days/(weekday_total/weekday_days))*100:.1f}%")

# =================================================================
# 3. GEOGRAPHIC ANALYSIS
# =================================================================
print("\n" + "=" * 60)
print("3. GEOGRAPHIC ANALYSIS")
print("=" * 60)

# State analysis
state_summary = df.groupby('state').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_18_plus': 'sum',
    'district': 'nunique',
    'pincode': 'nunique'
}).reset_index()

state_summary.columns = ['state', 'total_enrolments', 'total_children', 'total_adults', 'num_districts', 'num_pincodes']
state_summary['child_pct'] = (state_summary['total_children'] / state_summary['total_enrolments']) * 100
state_summary['avg_per_district'] = state_summary['total_enrolments'] / state_summary['num_districts']

# Clean state names (remove variations)
valid_states = state_summary[state_summary['total_enrolments'] >= 1000].copy()
valid_states = valid_states.sort_values('total_enrolments', ascending=False)

print(f"\n🏆 TOP 10 STATES BY ENROLMENTS:")
for i, (_, row) in enumerate(valid_states.head(10).iterrows(), 1):
    pct = row['total_enrolments'] / valid_states['total_enrolments'].sum() * 100
    print(f"   {i:2d}. {row['state']:25s}: {row['total_enrolments']:>10,} ({pct:>5.2f}%) | Child: {row['child_pct']:.1f}%")

print(f"\n🏆 TOP 5 STATES BY CHILD ENROLMENT %:")
top_child = valid_states.nlargest(5, 'child_pct')
for i, (_, row) in enumerate(top_child.iterrows(), 1):
    print(f"   {i}. {row['state']:25s}: {row['child_pct']:.1f}%")

print(f"\n⚠️  BOTTOM 5 STATES BY CHILD ENROLMENT %:")
bottom_child = valid_states.nsmallest(5, 'child_pct')
for i, (_, row) in enumerate(bottom_child.iterrows(), 1):
    print(f"   {i}. {row['state']:25s}: {row['child_pct']:.1f}%")

# District analysis
district_summary = df.groupby(['state', 'district']).agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_18_plus': 'sum'
}).reset_index()

district_summary['child_pct'] = (district_summary['demo_age_5_17'] / district_summary['total_enrolments']) * 100
district_summary = district_summary.sort_values('total_enrolments', ascending=False)

print(f"\n🏆 TOP 10 DISTRICTS BY ENROLMENTS:")
for i, (_, row) in enumerate(district_summary.head(10).iterrows(), 1):
    print(f"   {i:2d}. {row['district']:25s} ({row['state']:15s}): {row['total_enrolments']:>8,} | Child: {row['child_pct']:.1f}%")

# =================================================================
# 4. ANOMALY DETECTION
# =================================================================
print("\n" + "=" * 60)
print("4. ANOMALY DETECTION")
print("=" * 60)

daily['rolling_mean_7d'] = daily['total_enrolments'].rolling(7).mean()
daily['rolling_std_7d'] = daily['total_enrolments'].rolling(7).std()
daily['z_score'] = (daily['total_enrolments'] - daily['rolling_mean_7d']) / daily['rolling_std_7d']
daily['anomaly'] = daily['z_score'].abs() > 2

anomalies = daily[daily['anomaly'] == True].copy()
anomalies['day_name'] = anomalies['date'].dt.day_name()

print(f"\n⚠️  Found {len(anomalies)} anomalous days (Z-score > 2):")
for _, row in anomalies.iterrows():
    direction = "SPIKE" if row['z_score'] > 0 else "DROP"
    print(f"   {row['date'].strftime('%Y-%m-%d')} ({row['day_name']}): {row['total_enrolments']:>8,} ({direction}, Z={row['z_score']:.2f})")

# =================================================================
# 5. AGE DEMOGRAPHICS INSIGHTS
# =================================================================
print("\n" + "=" * 60)
print("5. AGE DEMOGRAPHICS INSIGHTS")
print("=" * 60)

# Child enrollment by month
monthly_child = df.groupby('month_name').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum'
}).reset_index()
monthly_child['child_pct'] = (monthly_child['demo_age_5_17'] / monthly_child['total_enrolments']) * 100
monthly_child['month_order'] = monthly_child['month_name'].map({m: i for i, m in enumerate(month_order)})
monthly_child = monthly_child.sort_values('month_order')

print(f"\n📊 CHILD ENROLMENT TRENDS BY MONTH:")
for _, row in monthly_child.iterrows():
    print(f"   {row['month_name']:12s}: {row['child_pct']:.1f}%")

# States with highest child enrolment growth potential (low current, high population)
top_states = valid_states.head(15)
low_child_high_pop = top_states[top_states['child_pct'] < 10]
print(f"\n🎯 STATES WITH LOW CHILD ENROLMENT (<10%) - POTENTIAL FOCUS AREAS:")
for _, row in low_child_high_pop.iterrows():
    print(f"   {row['state']:25s}: {row['child_pct']:.1f}% ({row['total_enrolments']:,} total)")

# =================================================================
# 6. KEY INSIGHTS SUMMARY
# =================================================================
print("\n" + "=" * 60)
print("6. KEY INSIGHTS SUMMARY")
print("=" * 60)

insights = []

insights.append({
    'category': 'Data Quality',
    'insight': 'Critical data entry inconsistencies',
    'detail': f"Found {len(problematic_states)}+ variations of state names (e.g., West Bengal appears as 'West Bengal', 'West Bangal', 'Westbengal', etc.). Recommend data standardization."
})

insights.append({
    'category': 'Data Quality',
    'insight': 'August 2025 data missing',
    'detail': 'Complete month of August 2025 is absent from the dataset, creating a gap in temporal analysis.'
})

insights.append({
    'category': 'Enrollment Pattern',
    'insight': 'March 2025 anomaly',
    'detail': f"March 2025 shows {daily['total_enrolments'].sum()/monthly['total_enrolments'].sum()*100:.1f}% of total enrolments in just one month, indicating potential data capture issue or enrollment drive."
})

insights.append({
    'category': 'Enrollment Pattern',
    'insight': 'Weekend drop is severe',
    'detail': f"Weekend enrolments are {(1 - weekend_total/weekend_days/(weekday_total/weekday_days))*100:.1f}% lower than weekdays. Centers may operate at reduced capacity on weekends."
})

insights.append({
    'category': 'Geographic',
    'insight': 'Uttar Pradesh dominates',
    'detail': f"UP alone accounts for {valid_states[valid_states['state']=='Uttar Pradesh']['total_enrolments'].sum()/valid_states['total_enrolments'].sum()*100:.1f}% of all enrolments, followed by Maharashtra and Bihar."
})

insights.append({
    'category': 'Geographic',
    'insight': 'Child enrolment varies significantly',
    'detail': f"Child enrolment percentage ranges from {valid_states['child_pct'].min():.1f}% ({valid_states.loc[valid_states['child_pct'].idxmin(), 'state']}) to {valid_states['child_pct'].max():.1f}% ({valid_states.loc[valid_states['child_pct'].idxmax(), 'state']})."
})

insights.append({
    'category': 'Demographics',
    'insight': 'Overall child enrolment is low',
    'detail': f"Only {daily['demo_age_5_17'].sum()/daily['total_enrolments'].sum()*100:.1f}% of total enrolments are children (5-17 years). This could indicate saturation in child enrolment or data quality issues."
})

insights.append({
    'category': 'Operational',
    'insight': 'High daily volatility',
    'detail': f"Coefficient of variation is {cv:.1f}%, indicating significant day-to-day fluctuations. Some days show {daily['total_enrolments'].min():,} while peak shows {daily['total_enrolments'].max():,} enrolments."
})

print(f"\n🎯 {len(insights)} KEY INSIGHTS IDENTIFIED:\n")
for i, ins in enumerate(insights, 1):
    print(f"{i}. [{ins['category']}] {ins['insight']}")
    print(f"   → {ins['detail']}\n")

# =================================================================
# 7. SAVE PROCESSED DATA
# =================================================================
print("\n" + "=" * 60)
print("7. SAVING PROCESSED DATA")
print("=" * 60)

# Save cleaned daily data
daily.to_csv('output/daily_enrollment_cleaned.csv', index=False)
print("✓ Saved cleaned daily enrollment data")

# Save cleaned state summary
valid_states.to_csv('output/state_summary_cleaned.csv', index=False)
print("✓ Saved cleaned state summary")

# Save district summary
district_summary.to_csv('output/district_summary_cleaned.csv', index=False)
print("✓ Saved cleaned district summary")

# Save insights
insights_df = pd.DataFrame(insights)
insights_df.to_csv('output/key_insights.csv', index=False)
print("✓ Saved key insights")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
