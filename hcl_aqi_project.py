
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

file_name = 'indian_cities_raw_aqi_dataset_133cities_2016_2025.csv'

df = pd.read_csv(file_name)

years = [str(y) for y in range(2016, 2026)]

# Check required columns
required_columns = ['City', 'State', 'Region'] + years

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Convert AQI columns into numeric values
for year in years:
    df[year] = pd.to_numeric(df[year], errors='coerce')

# Remove rows without city or state names
df = df.dropna(subset=['City', 'State'])

# ==========================================================
# 2. DATASET OVERVIEW
# ==========================================================

print("\n" + "=" * 65)
print("                 1. DATASET OVERVIEW")
print("=" * 65)

print("Total Cities Covered :", df['City'].nunique())
print("Total States / UTs   :", df['State'].nunique())
print("Total Regions        :", df['Region'].nunique())
print("Missing AQI Values   :", df[years].isnull().sum().sum())

print("\nMissing Values Year Wise:")
print(df[years].isnull().sum())

if 'Overall_Category' in df.columns:
    print("\nCategory Breakdown:")
    print(df['Overall_Category'].value_counts().to_string())

# ==========================================================
# 3. STATISTICAL SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("             2. STATISTICAL SUMMARY (2016-2025)")
print("=" * 65)

print(df[years].describe().round(2).to_string())

# Calculate 10-year mean if not already available
df['10Yr_Mean_AQI'] = df[years].mean(axis=1)

# Calculate percentage change from 2016 to 2025
df['10Yr_Change_%'] = (
    (df['2025'] - df['2016']) / df['2016'].replace(0, np.nan)
) * 100

# ==========================================================
# 4. TOP 5 MOST POLLUTED CITIES
# ==========================================================

print("\n" + "=" * 65)
print("             3. TOP 5 MOST POLLUTED CITIES")
print("=" * 65)

most_polluted = df.sort_values(
    '10Yr_Mean_AQI', ascending=False
)

print(
    most_polluted[
        ['City', 'State', '10Yr_Mean_AQI', '10Yr_Change_%']
    ].head(5).round(2).to_string(index=False)
)

# ==========================================================
# 5. TOP 5 CLEANEST CITIES
# ==========================================================

print("\n" + "=" * 65)
print("             4. TOP 5 CLEANEST CITIES")
print("=" * 65)

cleanest = df.sort_values(
    '10Yr_Mean_AQI', ascending=True
)

print(
    cleanest[
        ['City', 'State', '10Yr_Mean_AQI', '10Yr_Change_%']
    ].head(5).round(2).to_string(index=False)
)

# ==========================================================
# 6. MOST IMPROVED AND WORST DECLINE
# ==========================================================

print("\n" + "=" * 65)
print("             5. AQI IMPROVEMENT ANALYSIS")
print("=" * 65)

print("\nMost Improved Cities (Largest AQI Decrease):")

print(
    df.sort_values('10Yr_Change_%')[
        ['City', 'State', '2016', '2025', '10Yr_Change_%']
    ].head(5).round(2).to_string(index=False)
)

print("\nCities with Worsening AQI:")

worsened = df[df['10Yr_Change_%'] > 0].sort_values(
    '10Yr_Change_%', ascending=False
)

print(
    worsened[
        ['City', 'State', '2016', '2025', '10Yr_Change_%']
    ].round(2).to_string(index=False)
)

# ==========================================================
# 7. REGIONAL AND STATE ANALYSIS
# ==========================================================

print("\n" + "=" * 65)
print("             6. REGIONAL & STATE ANALYSIS")
print("=" * 65)

region_summary = df.groupby('Region').agg(
    Cities=('City', 'nunique'),
    Mean_AQI=('10Yr_Mean_AQI', 'mean'),
    Avg_Change_Pct=('10Yr_Change_%', 'mean')
).sort_values('Mean_AQI', ascending=False).round(2)

print("\nRegional Summary:")
print(region_summary.to_string())

state_summary = df.groupby('State')['10Yr_Mean_AQI'].mean()

print("\nTop 5 States by Average AQI:")
print(state_summary.sort_values(ascending=False).head(5).round(2))

# ==========================================================
# 8. STATE-WISE AQI HEATMAP
# ==========================================================

state_df = df.groupby('State')[years].mean()

state_df['Mean'] = state_df.mean(axis=1)

state_df = state_df.sort_values(
    by='Mean', ascending=False
).drop(columns='Mean')

fig, ax = plt.subplots(figsize=(14, 11))

im = ax.imshow(
    state_df.values,
    cmap='YlOrRd',
    aspect='auto'
)

ax.set_xticks(np.arange(len(years)))
ax.set_yticks(np.arange(len(state_df.index)))

ax.set_xticklabels(years, fontsize=10)
ax.set_yticklabels(state_df.index, fontsize=9)

# Add AQI values inside cells
for i in range(len(state_df.index)):
    for j in range(len(years)):

        val = state_df.values[i, j]

        if pd.notna(val):
            text_color = 'white' if val > 165 else 'black'

            ax.text(
                j, i, f'{val:.0f}',
                ha='center',
                va='center',
                color=text_color,
                fontsize=8
            )

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Average AQI')

ax.set_title(
    'State-Wise Annual AQI Heatmap (2016-2025)',
    fontsize=14,
    fontweight='bold',
    pad=15
)

ax.set_xlabel('Year')
ax.set_ylabel('State / UT')

plt.tight_layout()
plt.show()

# ==========================================================
# 9. AQI TREND OF ALL 133 CITIES
# ==========================================================

plt.figure(figsize=(13, 7))

# Plot all cities in light gray
for _, row in df.iterrows():

    city_values = pd.to_numeric(
        row[years], errors='coerce'
    )

    plt.plot(
        years,
        city_values.values,
        color='lightgray',
        linewidth=0.8,
        alpha=0.6
    )

# Highlight selected cities
highlight_cities = [
    'Ghaziabad',
    'Delhi',
    'Mumbai',
    'Bengaluru',
    'Aizawl'
]

colors = [
    'red',
    'orange',
    'blue',
    'green',
    'purple'
]

for city, color in zip(highlight_cities, colors):

    city_data = df[
        df['City'].str.strip().str.lower() == city.lower()
    ]

    if city_data.empty:
        print(f"\nWarning: {city} not found in dataset")
        continue

    if len(city_data) > 1:
        print(f"\nWarning: Multiple entries found for {city}; using first match")

    city_row = city_data.iloc[0]

    city_values = pd.to_numeric(
        city_row[years], errors='coerce'
    )

    plt.plot(
        years,
        city_values.values,
        marker='o',
        linewidth=2.5,
        color=color,
        label=city
    )

plt.title(
    '10-Year AQI Trajectory: All Cities vs Selected Cities',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Mean AQI', fontsize=12)

plt.grid(
    True,
    linestyle='--',
    alpha=0.4
)

plt.legend(
    title='Highlighted Cities',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()

# ==========================================================
# 10. PROJECT COMPLETED
# ==========================================================

print("\n" + "=" * 65)
print("             AQI DATA ANALYSIS COMPLETED")
print("=" * 65)

print("Total Cities Analyzed:", df['City'].nunique())
print("Years Analyzed: 2016-2025")
print("Visualizations Generated: 2")
print("Analysis Completed Successfully!")
