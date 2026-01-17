"""
Aadhaar Enrollment Density Analysis
This script creates a visualization showing enrollment per capita (per 100,000 population)
to identify which states have higher application rates relative to their population.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for saving files
import matplotlib.pyplot as plt
import numpy as np

# Indian State Population Data (2024 estimates in thousands)
# Source: Census projections and government data
STATE_POPULATION = {
    'uttar pradesh': 241000,
    'maharashtra': 127000,
    'bihar': 131000,
    'west bengal': 101000,
    'madhya pradesh': 88000,
    'tamil nadu': 78000,
    'rajasthan': 83000,
    'karnataka': 69000,
    'gujarat': 71000,
    'andhra pradesh': 53000,
    'odisha': 46000,
    'telangana': 40000,
    'kerala': 35000,
    'jharkhand': 40000,
    'assam': 36000,
    'punjab': 31000,
    'chhattisgarh': 32000,
    'haryana': 30000,
    'delhi': 21000,
    'jammu and kashmir': 14000,
    'uttarakhand': 12000,
    'himachal pradesh': 7500,
    'tripura': 4200,
    'meghalaya': 3800,
    'manipur': 3200,
    'nagaland': 2300,
    'goa': 1600,
    'arunachal pradesh': 1700,
    'puducherry': 1500,
    'mizoram': 1300,
    'chandigarh': 1200,
    'sikkim': 700,
    'andaman and nicobar islands': 400,
    'dadra and nagar haveli and daman and diu': 600,
    'ladakh': 300,
    'lakshadweep': 70,
}

def normalize_state_name(state_name):
    """Normalize state name for matching"""
    if not isinstance(state_name, str):
        return None
    
    # Convert to lowercase and strip
    normalized = state_name.lower().strip()
    
    # Handle common variations
    replacements = {
        '&': 'and',
        'west  bengal': 'west bengal',
        'west bangal': 'west bengal',
        'west bengli': 'west bengal',
        'westbengal': 'west bengal',
        'chhatisgarh': 'chhattisgarh',
        'orissa': 'odisha',
        'pondicherry': 'puducherry',
        'uttaranchal': 'uttarakhand',
        'jammu & kashmir': 'jammu and kashmir',
        'andaman & nicobar islands': 'andaman and nicobar islands',
        'dadra & nagar haveli': 'dadra and nagar haveli and daman and diu',
        'daman & diu': 'dadra and nagar haveli and daman and diu',
        'daman and diu': 'dadra and nagar haveli and daman and diu',
        'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized

# List of invalid state entries (cities, pincodes, etc.)
INVALID_ENTRIES = {
    '100000', 'balanagar', 'darbhanga', 'jaipur', 'madanapalle', 
    'nagpur', 'puttenahalli', 'raja annamalai puram'
}

print("Loading Aadhaar demographic data...")
df = pd.read_csv('api_data_aadhar_demographic_merged.csv')

print(f"Total records: {len(df):,}")
print(f"Unique state entries: {df['state'].nunique()}")

# Normalize state names
df['state_normalized'] = df['state'].apply(normalize_state_name)

# Remove invalid entries
df = df[~df['state_normalized'].isin(INVALID_ENTRIES)]
df = df[df['state_normalized'].notna()]

# Calculate total enrollments per state
df['total_enrollment'] = df['demo_age_5_17'] + df['demo_age_17_']

# Group by normalized state name and sum enrollments
state_enrollments = df.groupby('state_normalized')['total_enrollment'].sum().reset_index()
state_enrollments.columns = ['state', 'total_enrollments']

print("\n--- State Enrollment Totals (after normalization) ---")
print(state_enrollments.sort_values('total_enrollments', ascending=False).head(10))

# Add population data
state_enrollments['population_thousands'] = state_enrollments['state'].map(STATE_POPULATION)

# Calculate enrollment density (per 100,000 population)
state_enrollments['enrollment_per_100k'] = (
    state_enrollments['total_enrollments'] / 
    (state_enrollments['population_thousands'] * 1000) * 100000
)

# Check for missing population data
missing_pop = state_enrollments[state_enrollments['population_thousands'].isna()]
if len(missing_pop) > 0:
    print("\n⚠️ Missing population data for:")
    for state in missing_pop['state'].values:
        print(f"  - {state}")

# Filter to only states with population data
density_data = state_enrollments[state_enrollments['population_thousands'].notna()].copy()
density_data = density_data.sort_values('enrollment_per_100k', ascending=False)

# Convert state names back to title case for display
density_data['display_state'] = density_data['state'].str.title()

print("\n--- Top 10 States by Enrollment Density (per 100k population) ---")
for idx, row in density_data.head(10).iterrows():
    print(f"  {row['display_state']}: {row['enrollment_per_100k']:.1f} per 100k (Total: {row['total_enrollments']:,})")

# ============================================
# Create Visualization
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(20, 12))

# Color palette - gradient from high (green) to low (red)
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(density_data)))

# --- Chart 1: Enrollment Density (per 100k population) ---
ax1 = axes[0]
bars1 = ax1.barh(
    range(len(density_data)),
    density_data['enrollment_per_100k'],
    color=colors,
    edgecolor='white',
    linewidth=0.5
)

ax1.set_yticks(range(len(density_data)))
ax1.set_yticklabels(density_data['display_state'], fontsize=9)
ax1.set_xlabel('Enrollments per 100,000 Population', fontsize=12, fontweight='bold')
ax1.set_ylabel('State/UT', fontsize=12, fontweight='bold')
ax1.set_title('🎯 Aadhaar Enrollment DENSITY by State\n(Normalized by Population - Shows True Application Rate)', fontsize=14, fontweight='bold', color='darkgreen')

# Add value labels
for i, (bar, value) in enumerate(zip(bars1, density_data['enrollment_per_100k'])):
    ax1.text(value + 50, i, f'{value:.0f}', va='center', ha='left', fontsize=8)

ax1.invert_yaxis()  # Highest at top
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add median line
median_density = density_data['enrollment_per_100k'].median()
ax1.axvline(median_density, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(median_density + 100, len(density_data) - 2, f'Median: {median_density:.0f}', 
         color='red', fontsize=10, fontweight='bold')

# --- Chart 2: Raw Enrollments for comparison ---
ax2 = axes[1]
raw_data = state_enrollments.dropna().sort_values('total_enrollments', ascending=False).copy()
raw_data['display_state'] = raw_data['state'].str.title()

colors2 = plt.cm.Blues(np.linspace(0.9, 0.3, len(raw_data)))

bars2 = ax2.barh(
    range(len(raw_data)),
    raw_data['total_enrollments'],
    color=colors2,
    edgecolor='white',
    linewidth=0.5
)

ax2.set_yticks(range(len(raw_data)))
ax2.set_yticklabels(raw_data['display_state'], fontsize=9)
ax2.set_xlabel('Total Enrollments', fontsize=12, fontweight='bold')
ax2.set_ylabel('State/UT', fontsize=12, fontweight='bold')
ax2.set_title('📊 Raw Enrollment Numbers\n(Before Population Normalization)', fontsize=14, fontweight='bold', color='darkblue')

# Format x-axis with millions
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('enrollment_density_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n✅ Chart saved as 'enrollment_density_comparison.png'")

# ============================================
# Create a cleaner single-chart for density only
# ============================================

fig2, ax = plt.subplots(figsize=(14, 12))

# Color based on density value
norm_values = (density_data['enrollment_per_100k'] - density_data['enrollment_per_100k'].min()) / \
              (density_data['enrollment_per_100k'].max() - density_data['enrollment_per_100k'].min())
colors3 = plt.cm.viridis(norm_values)

bars = ax.barh(
    range(len(density_data)),
    density_data['enrollment_per_100k'],
    color=colors3,
    edgecolor='white',
    linewidth=0.5,
    height=0.7
)

ax.set_yticks(range(len(density_data)))
ax.set_yticklabels(density_data['display_state'], fontsize=10)
ax.set_xlabel('Enrollments per 100,000 Population', fontsize=14, fontweight='bold')
ax.set_title('🎯 Which States Have Highest Aadhaar Application Rate?\n(Enrollment Density - Normalized by Population)', fontsize=16, fontweight='bold')

# Add value labels with color
for i, (bar, value) in enumerate(zip(bars, density_data['enrollment_per_100k'])):
    color = 'white' if value > density_data['enrollment_per_100k'].median() else 'black'
    ax.text(value - 100, i, f'{value:.0f}', va='center', ha='right', fontsize=9, 
            fontweight='bold', color=color)

ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, density_data['enrollment_per_100k'].max() * 1.05)

# Add annotations
ax.annotate('Higher = More applications\nper capita', xy=(0.95, 0.02), xycoords='axes fraction',
            fontsize=10, ha='right', style='italic', alpha=0.7)

plt.tight_layout()
plt.savefig('enrollment_density_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Single density chart saved as 'enrollment_density_chart.png'")

# ============================================
# Additional Insights
# ============================================

print("\n" + "="*60)
print("KEY INSIGHTS - ENROLLMENT DENSITY ANALYSIS")
print("="*60)

# Top 5 by density
print("\n🏆 Top 5 States by Enrollment Rate (per 100k population):")
top5_density = density_data.head(5)
for idx, row in top5_density.iterrows():
    print(f"   {row['display_state']}: {row['enrollment_per_100k']:.1f} per 100k")

# Bottom 5 by density
print("\n📉 Bottom 5 States by Enrollment Rate (per 100k population):")
bottom5_density = density_data.tail(5)
for idx, row in bottom5_density.iterrows():
    print(f"   {row['display_state']}: {row['enrollment_per_100k']:.1f} per 100k")

# Compare raw vs density rankings
print("\n📊 States that Rank HIGHER when Normalized by Population:")
raw_ranking = state_enrollments.dropna().sort_values('total_enrollments', ascending=False)['state'].tolist()
density_ranking = density_data['state'].tolist()

for i, state in enumerate(density_ranking[:15]):
    if state in raw_ranking:
        raw_rank = raw_ranking.index(state) + 1
        density_rank = i + 1
        improvement = raw_rank - density_rank
        if improvement >= 5:
            state_display = state.title()
            print(f"   🔼 {state_display}: Raw #{raw_rank} → Density #{density_rank} (↑{improvement} positions)")

print("\n📊 States that Rank LOWER when Normalized by Population:")
for i, state in enumerate(density_ranking):
    if state in raw_ranking:
        raw_rank = raw_ranking.index(state) + 1
        density_rank = i + 1
        decline = density_rank - raw_rank
        if decline >= 5:
            state_display = state.title()
            print(f"   🔻 {state_display}: Raw #{raw_rank} → Density #{density_rank} (↓{decline} positions)")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\nThe density chart shows which states have a HIGHER APPLICATION RATE")
print("relative to their population. States at the top have more people applying")
print("per capita, regardless of their total population size.")
print("\nThis is more useful than raw numbers because it removes the bias")
print("caused by larger states having more people.")
