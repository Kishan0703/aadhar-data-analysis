# Aadhaar Demographics Analysis

A comprehensive data science project analyzing Aadhaar enrollment data across Indian states, districts, and pincodes.

## Dataset
- ~2 million rows of Aadhaar enrollment data
- Date range: March 2025 - December 2025
- Columns: date, state, district, pincode, demo_age_5_17, demo_age_18_plus
- **Note:** August 2025 data is missing from the source

## Project Structure
- `data.csv` — Raw dataset
- `deep_analysis.py` — Deep analysis script extracting meaningful insights
- `create_plots.py` — Generates high-quality visualization plots
- `app.py` — Interactive Streamlit dashboard for visual exploration
- `AADHAAR_ANALYSIS_REPORT.md` — Comprehensive analysis report with insights
- `output/` — Generated CSV outputs and plots
- `insights.ipynb`, `insights1.ipynb` — Original Jupyter notebook explorations

## Setup

```bash
pip install -r requirements.txt
```

## Run Deep Analysis

```bash
python deep_analysis.py
```

## Generate Plots

```bash
python create_plots.py
```

## Launch Dashboard

```bash
streamlit run app.py
```

## Key Findings

### Data Quality Issues
- **9 variations** of "West Bengal" found in state names
- **21.6% duplicate records** removed during cleaning
- **August 2025 data completely missing**
- **Extreme volatility** with CV = 220%

### Enrollment Patterns
- **Saturday anomaly**: 2.5x higher enrolments than average weekdays
- **March 2025** accounts for 22.7% of all enrolments
- **Uttar Pradesh** dominates with 17.7% of total enrolments
- **Only 9.8%** children enrolled (5-17 years)

### Geographic Insights
- Top 3 states (UP, Maharashtra, Bihar) = 38% of enrolments
- **Maharashtra** has lowest child enrollment rate (5.4%)
- **Thane & Pune** are top districts

## Dashboard Features
1. **Overview** — Key insights, daily trends, age-group breakdown, monthly patterns
2. **State Analysis** — Top states, child enrollment %, summary tables
3. **Trends & Seasonality** — Weekday patterns, monthly child % trends
4. **Geography** — Top districts and pincodes
5. **Clusters** — K-Means state clustering visualization
6. **Anomalies** — Daily anomaly detection with Z-scores

## Visualizations Generated
- `plot_daily_trend.png` — Daily enrollment with 7-day rolling average
- `plot_monthly_distribution.png` — Monthly enrollment breakdown
- `plot_top_states.png` — Top 10 states by enrollment
- `plot_age_distribution.png` — Age group distribution by month
- `plot_child_enrollment_pct.png` — Child enrollment % by state
- `plot_weekday_pattern.png` — Weekday enrollment patterns
- `plot_top_districts.png` — Top 10 districts
- `plot_volatility.png` — Enrollment volatility over time

## Report
See `AADHAAR_ANALYSIS_REPORT.md` for detailed analysis and recommendations.
