"""
Generate important plots for Aadhaar analysis
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
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
sns.set_palette('husl')

# Color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# Load data
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
df = df.drop_duplicates()

df['total_enrolments'] = df['demo_age_5_17'] + df['demo_age_18_plus']
df['child_pct'] = np.where(df['total_enrolments'] > 0, 
                            (df['demo_age_5_17'] / df['total_enrolments']) * 100, 0)
df['month_name'] = df['date'].dt.month_name()
df['day_name'] = df['date'].dt.day_name()
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

print("Creating visualizations...")

# =================================================================
# PLOT 1: Daily Enrollment Trend
# =================================================================
fig, ax = plt.subplots(figsize=(16, 6))

daily = df.groupby('date').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_18_plus': 'sum'
}).reset_index()

daily['rolling_mean_7d'] = daily['total_enrolments'].rolling(7).mean()

ax.fill_between(daily['date'], daily['total_enrolments'], alpha=0.3, color='#45B7D1', label='Daily Enrolments')
ax.plot(daily['date'], daily['rolling_mean_7d'], color='#FF6B6B', linewidth=2.5, label='7-Day Rolling Average')

ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Total Enrolments', fontweight='bold')
ax.set_title('Daily Enrollment Trend with 7-Day Rolling Average', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Highlight the gap
ax.axvspan(pd.Timestamp('2025-08-01'), pd.Timestamp('2025-08-31'), alpha=0.2, color='red', label='Missing August Data')

plt.tight_layout()
plt.savefig('output/plot_daily_trend.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_daily_trend.png")

# =================================================================
# PLOT 2: Monthly Enrollment Distribution
# =================================================================
fig, ax = plt.subplots(figsize=(12, 6))

month_order = ['March', 'April', 'May', 'June', 'July', 'September', 'October', 'November', 'December']
monthly = df.groupby('month_name').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum',
    'demo_age_18_plus': 'sum'
}).reset_index()

monthly['month_order'] = monthly['month_name'].map({m: i for i, m in enumerate(month_order)})
monthly = monthly.sort_values('month_order')

bars = ax.bar(monthly['month_name'], monthly['total_enrolments'], color=colors[:len(monthly)], edgecolor='black', linewidth=1)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Total Enrolments', fontweight='bold')
ax.set_title('Monthly Enrollment Distribution (2025)', fontsize=16, fontweight='bold', pad=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/plot_monthly_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_monthly_distribution.png")

# =================================================================
# PLOT 3: Top 10 States
# =================================================================
fig, ax = plt.subplots(figsize=(12, 7))

state_summary = df.groupby('state').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum'
}).reset_index()

state_summary['child_pct'] = (state_summary['demo_age_5_17'] / state_summary['total_enrolments']) * 100
state_summary = state_summary[state_summary['total_enrolments'] >= 1000]
state_summary = state_summary.sort_values('total_enrolments', ascending=False).head(10)

bars = ax.barh(state_summary['state'][::-1], state_summary['total_enrolments'][::-1], 
               color=colors[:10][::-1], edgecolor='black', linewidth=1)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    child_pct = state_summary.iloc[9-i]['child_pct']
    ax.text(width + 50000, bar.get_y() + bar.get_height()/2.,
            f'{int(width):,} ({child_pct:.1f}% children)',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Total Enrolments', fontweight='bold')
ax.set_ylabel('State', fontweight='bold')
ax.set_title('Top 10 States by Total Enrolments', fontsize=16, fontweight='bold', pad=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig('output/plot_top_states.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_top_states.png")

# =================================================================
# PLOT 4: Age Group Distribution (Stacked Bar)
# =================================================================
fig, ax = plt.subplots(figsize=(14, 7))

monthly_age = monthly.copy()

x = np.arange(len(monthly_age))
width = 0.6

p1 = ax.bar(x, monthly_age['demo_age_18_plus'], width, label='Adults (18+)', color='#45B7D1', edgecolor='black')
p2 = ax.bar(x, monthly_age['demo_age_5_17'], width, bottom=monthly_age['demo_age_18_plus'], 
            label='Children (5-17)', color='#FF6B6B', edgecolor='black')

ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Total Enrolments', fontweight='bold')
ax.set_title('Monthly Enrollment by Age Group', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(monthly_age['month_name'], rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig('output/plot_age_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_age_distribution.png")

# =================================================================
# PLOT 5: Child Enrollment % by State
# =================================================================
fig, ax = plt.subplots(figsize=(12, 7))

state_child = df.groupby('state').agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum'
}).reset_index()

state_child['child_pct'] = (state_child['demo_age_5_17'] / state_child['total_enrolments']) * 100
state_child = state_child[state_child['total_enrolments'] >= 1000]
state_child = state_child.sort_values('child_pct', ascending=False).head(15)

# Color gradient based on percentage
norm = plt.Normalize(state_child['child_pct'].min(), state_child['child_pct'].max())
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
sm.set_array([])

bars = ax.barh(state_child['state'][::-1], state_child['child_pct'][::-1],
               color=[sm.to_rgba(v) for v in state_child['child_pct'][::-1]],
               edgecolor='black', linewidth=1)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f}%',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Child Enrollment (%)', fontweight='bold')
ax.set_ylabel('State', fontweight='bold')
ax.set_title('Top 15 States by Child Enrollment Percentage', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('output/plot_child_enrollment_pct.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_child_enrollment_pct.png")

# =================================================================
# PLOT 6: Weekday Pattern
# =================================================================
fig, ax = plt.subplots(figsize=(12, 6))

weekday_stats = df.groupby('day_name').agg({
    'total_enrolments': 'sum'
}).reset_index()

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_stats['day_order'] = weekday_stats['day_name'].map({d: i for i, d in enumerate(weekday_order)})
weekday_stats = weekday_stats.sort_values('day_order')

# Count days for each weekday
days_count = df.groupby('day_name')['date'].nunique().reset_index()
days_count.columns = ['day_name', 'num_days']
weekday_stats = weekday_stats.merge(days_count, on='day_name')
weekday_stats['avg_per_day'] = weekday_stats['total_enrolments'] / weekday_stats['num_days']

colors_weekday = ['#4ECDC4' if day not in ['Saturday', 'Sunday'] else '#FF6B6B' for day in weekday_order]

bars = ax.bar(weekday_stats['day_name'], weekday_stats['avg_per_day'], color=colors_weekday, edgecolor='black', linewidth=1)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Day of Week', fontweight='bold')
ax.set_ylabel('Average Enrolments per Day', fontweight='bold')
ax.set_title('Average Daily Enrolments by Weekday (Weekdays vs Weekend)', fontsize=16, fontweight='bold', pad=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4ECDC4', edgecolor='black', label='Weekdays'),
                   Patch(facecolor='#FF6B6B', edgecolor='black', label='Weekend')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('output/plot_weekday_pattern.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_weekday_pattern.png")

# =================================================================
# PLOT 7: Top 10 Districts
# =================================================================
fig, ax = plt.subplots(figsize=(12, 7))

district_summary = df.groupby(['state', 'district']).agg({
    'total_enrolments': 'sum',
    'demo_age_5_17': 'sum'
}).reset_index()

district_summary['child_pct'] = (district_summary['demo_age_5_17'] / district_summary['total_enrolments']) * 100
district_summary = district_summary.sort_values('total_enrolments', ascending=False).head(10)

district_summary['label'] = district_summary['district'].astype(str) + '\n(' + district_summary['state'].astype(str) + ')'

bars = ax.barh(district_summary['label'][::-1], district_summary['total_enrolments'][::-1],
               color=colors[:10][::-1], edgecolor='black', linewidth=1)

for i, bar in enumerate(bars):
    width = bar.get_width()
    child_pct = district_summary.iloc[9-i]['child_pct']
    ax.text(width + 5000, bar.get_y() + bar.get_height()/2.,
            f'{int(width):,} ({child_pct:.1f}%)',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Total Enrolments', fontweight='bold')
ax.set_ylabel('District (State)', fontweight='bold')
ax.set_title('Top 10 Districts by Total Enrolments', fontsize=16, fontweight='bold', pad=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig('output/plot_top_districts.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_top_districts.png")

# =================================================================
# PLOT 8: Volatility Over Time
# =================================================================
fig, ax = plt.subplots(figsize=(16, 6))

daily['rolling_std_30d'] = daily['total_enrolments'].rolling(30).std()
daily['cv_30d'] = (daily['rolling_std_30d'] / daily['rolling_mean_7d']) * 100

ax2 = ax.twinx()

ax.fill_between(daily['date'], daily['total_enrolments'], alpha=0.3, color='#45B7D1', label='Daily Enrolments')
ax.plot(daily['date'], daily['rolling_mean_7d'], color='#45B7D1', linewidth=2, label='7-Day Avg')
ax2.plot(daily['date'], daily['cv_30d'], color='#FF6B6B', linewidth=2, linestyle='--', label='CV% (30-day)')

ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Total Enrolments', fontweight='bold', color='#45B7D1')
ax2.set_ylabel('Coefficient of Variation (%)', fontweight='bold', color='#FF6B6B')
ax.set_title('Enrollment Volatility Over Time', fontsize=16, fontweight='bold', pad=20)

ax.tick_params(axis='y', labelcolor='#45B7D1')
ax2.tick_params(axis='y', labelcolor='#FF6B6B')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig('output/plot_volatility.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: plot_volatility.png")

print("\n✅ All 8 visualizations saved to output/ folder")
