"""
Saccade Metrics Visualization
Compares saccade behaviors between successful and unsuccessful pilots
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Load data
df = pd.read_csv('datasets/AOI_DGMs.csv')

# Add success category
df['Success_Category'] = df['Approach_Score'].apply(
    lambda x: 'Successful' if x >= 0.7 else 'Unsuccessful'
)

# Define AOI list for per AOI analysis
aois = ['Alt_VSI', 'AI', 'ASI', 'SSI', 'TI_HSI', 'RPM', 'Window', 'NoAOI']

# Convert all AOI metric columns to numeric, replacing #NULL! with NaN
metric_suffixes = [
    'total_number_of_saccades',
    'mean_saccade_duration',
    'Average_Peak_Saccade_Velocity'
]

for aoi in aois:
    for suffix in metric_suffixes:
        col_name = f'{aoi}_{suffix}'
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')


# ============================================================================
# VISUALIZATION 1: Overall Saccade Count Comparison (Whole Screen)
# ============================================================================

fig1 = go.Figure()

for category in ['Successful', 'Unsuccessful']:
    data = df[df['Success_Category'] == category]['total_number_of_saccades']
    
    fig1.add_trace(go.Box(
        y=data,
        name=category,
        marker_color='#2ecc71' if category == 'Successful' else '#e74c3c',
        boxmean='sd'
    ))

fig1.update_layout(
    title='Total Saccade Count: Successful vs Unsuccessful Pilots',
    yaxis_title='Number of Saccades',
    xaxis_title='Pilot Category',
    template='plotly_white',
    height=500,
    showlegend=True
)

fig1.write_html('saccade_count_comparison.html')
print("[OK] Created: saccade_count_comparison.html")

# ============================================================================
# VISUALIZATION 2: Multi metric Saccade Comparison (2x2 Grid)
# ============================================================================

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Mean Saccade Duration',
        'Mean Saccade Length', 
        'Peak Saccade Velocity',
        'Fixation to Saccade Ratio'
    )
)

metrics_to_plot = [
    ('mean_saccade_duration', 1, 1, 'Duration (s)'),
    ('mean_saccade_length', 1, 2, 'Length (px)'),
    ('Average_Peak_Saccade_Velocity', 2, 1, 'Velocity (°/s)'),
    ('fixation_to_saccade_ratio', 2, 2, 'Ratio')
]

for metric, row, col, ylabel in metrics_to_plot:
    for category, color in [('Successful', '#2ecc71'), ('Unsuccessful', '#e74c3c')]:
        data = df[df['Success_Category'] == category][metric]
        
        fig2.add_trace(
            go.Box(
                y=data,
                name=category,
                marker_color=color,
                showlegend=(row == 1 and col == 1),
                boxmean='sd'
            ),
            row=row, col=col
        )
    
    fig2.update_yaxes(title_text=ylabel, row=row, col=col)

fig2.update_layout(
    title_text='Saccade Metrics Comparison: Successful vs Unsuccessful Pilots',
    height=800,
    template='plotly_white'
)

fig2.write_html('saccade_metrics_comparison.html')
print("[OK] Created: saccade_metrics_comparison.html")

# ============================================================================
# VISUALIZATION 3: Saccade Count by AOI (Grouped Bar Chart)
# ============================================================================

# Prepare data for AOI comparison
aoi_data = []
for aoi in aois:
    col_name = f'{aoi}_total_number_of_saccades'
    for category in ['Successful', 'Unsuccessful']:
        mean_count = df[df['Success_Category'] == category][col_name].mean()
        aoi_data.append({
            'AOI': aoi,
            'Category': category,
            'Mean_Saccades': mean_count
        })

aoi_df = pd.DataFrame(aoi_data)

fig3 = go.Figure()

for category, color in [('Successful', '#2ecc71'), ('Unsuccessful', '#e74c3c')]:
    data = aoi_df[aoi_df['Category'] == category]
    
    fig3.add_trace(go.Bar(
        x=data['AOI'],
        y=data['Mean_Saccades'],
        name=category,
        marker_color=color,
        text=data['Mean_Saccades'].round(1),
        textposition='auto'
    ))

fig3.update_layout(
    title='Mean Saccade Count by Area of Interest',
    xaxis_title='Area of Interest',
    yaxis_title='Mean Number of Saccades',
    barmode='group',
    template='plotly_white',
    height=500
)

fig3.write_html('saccade_count_by_aoi.html')
print("[OK] Created: saccade_count_by_aoi.html")

# ============================================================================
# VISUALIZATION 4: Interactive Saccade Metrics Heatmap by AOI
# ============================================================================

# Create comparison matrix for mean saccade duration by AOI
heatmap_data = []
for aoi in aois:
    duration_col = f'{aoi}_mean_saccade_duration'
    successful_mean = df[df['Success_Category'] == 'Successful'][duration_col].mean()
    unsuccessful_mean = df[df['Success_Category'] == 'Unsuccessful'][duration_col].mean()
    heatmap_data.append({
        'AOI': aoi,
        'Successful': successful_mean,
        'Unsuccessful': unsuccessful_mean,
        'Difference': successful_mean - unsuccessful_mean
    })

heatmap_df = pd.DataFrame(heatmap_data)

fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Successful Pilots', 'Unsuccessful Pilots'),
    horizontal_spacing=0.15
)

# Successful pilots heatmap
fig4.add_trace(
    go.Heatmap(
        z=heatmap_df['Successful'].values.reshape(-1, 1),
        x=['Mean Duration'],
        y=heatmap_df['AOI'],
        colorscale='Greens',
        text=heatmap_df['Successful'].round(3).values.reshape(-1, 1),
        texttemplate='%{text}s',
        textfont={"size": 10},
        hovertemplate='AOI: %{y}<br>Duration: %{z:.3f}s<extra></extra>',
        showscale=True,
        colorbar=dict(x=0.45)
    ),
    row=1, col=1
)

# Unsuccessful pilots heatmap
fig4.add_trace(
    go.Heatmap(
        z=heatmap_df['Unsuccessful'].values.reshape(-1, 1),
        x=['Mean Duration'],
        y=heatmap_df['AOI'],
        colorscale='Reds',
        text=heatmap_df['Unsuccessful'].round(3).values.reshape(-1, 1),
        texttemplate='%{text}s',
        textfont={"size": 10},
        hovertemplate='AOI: %{y}<br>Duration: %{z:.3f}s<extra></extra>',
        showscale=True,
        colorbar=dict(x=1.0)
    ),
    row=1, col=2
)

fig4.update_layout(
    title='Mean Saccade Duration by AOI: Heatmap Comparison',
    height=600,
    template='plotly_white'
)

fig4.write_html('saccade_duration_heatmap.html')
print("[OK] Created: saccade_duration_heatmap.html")

# ============================================================================
# VISUALIZATION 5: Saccade Efficiency Scatter Plot
# ============================================================================

fig5 = go.Figure()

for category, color, symbol in [('Successful', '#2ecc71', 'circle'), ('Unsuccessful', '#e74c3c', 'square')]:
    data = df[df['Success_Category'] == category]
    
    fig5.add_trace(go.Scatter(
        x=data['total_number_of_saccades'],
        y=data['mean_saccade_length'],
        mode='markers',
        name=category,
        marker=dict(
            color=color,
            size=10,
            symbol=symbol,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=data['PID'],
        hovertemplate='<b>Pilot %{text}</b><br>' +
                      'Total Saccades: %{x}<br>' +
                      'Mean Length: %{y:.1f} px<br>' +
                      '<extra></extra>'
    ))

fig5.update_layout(
    title='Saccade Efficiency: Total Count vs Mean Length',
    xaxis_title='Total Number of Saccades',
    yaxis_title='Mean Saccade Length (pixels)',
    template='plotly_white',
    height=500,
    hovermode='closest'
)

fig5.write_html('saccade_efficiency.html')
print("[OK] Created: saccade_efficiency.html")

# ============================================================================
# VISUALIZATION 6: Peak Velocity Comparison by AOI
# ============================================================================

velocity_data = []
for aoi in aois:
    vel_col = f'{aoi}_Average_Peak_Saccade_Velocity'
    for category in ['Successful', 'Unsuccessful']:
        mean_vel = df[df['Success_Category'] == category][vel_col].mean()
        velocity_data.append({
            'AOI': aoi,
            'Category': category,
            'Mean_Velocity': mean_vel
        })

vel_df = pd.DataFrame(velocity_data)

fig6 = go.Figure()

for category, color in [('Successful', '#2ecc71'), ('Unsuccessful', '#e74c3c')]:
    data = vel_df[vel_df['Category'] == category]
    
    fig6.add_trace(go.Bar(
        x=data['AOI'],
        y=data['Mean_Velocity'],
        name=category,
        marker_color=color,
        text=data['Mean_Velocity'].round(1),
        textposition='auto'
    ))

fig6.update_layout(
    title='Average Peak Saccade Velocity by AOI',
    xaxis_title='Area of Interest',
    yaxis_title='Average Peak Velocity (°/s)',
    barmode='group',
    template='plotly_white',
    height=500
)

fig6.write_html('saccade_velocity_by_aoi.html')
print("[OK] Created: saccade_velocity_by_aoi.html")

# ============================================================================
#S UMMARY
# ============================================================================

print("\n" + "="*80)
print("SACCADE METRICS SUMMARY STATISTICS")
print("="*80)

metrics_summary = {
    'Total Saccade Count': 'total_number_of_saccades',
    'Mean Saccade Duration (s)': 'mean_saccade_duration',
    'Mean Saccade Length (px)': 'mean_saccade_length',
    'Average Peak Velocity (°/s)': 'Average_Peak_Saccade_Velocity',
    'Fixation to Saccade Ratio': 'fixation_to_saccade_ratio'
}

for metric_name, metric_col in metrics_summary.items():
    print(f"\n{metric_name}:")
    print("-" * 70)
    
    successful = df[df['Success_Category'] == 'Successful'][metric_col]
    unsuccessful = df[df['Success_Category'] == 'Unsuccessful'][metric_col]
    
    print(f"  Successful   - Mean: {successful.mean():.3f}, SD: {successful.std():.3f}")
    print(f"  Unsuccessful - Mean: {unsuccessful.mean():.3f}, SD: {unsuccessful.std():.3f}")
    print(f"  Difference   - {successful.mean() - unsuccessful.mean():.3f}")
    
    if unsuccessful.mean() != 0:
        diff_percent = ((successful.mean() - unsuccessful.mean()) / unsuccessful.mean()) * 100
        print(f"  % Difference - {diff_percent:.1f}%")

# Per AOI Summary
print("\n" + "="*80)
print("PER AOI SACCADE COUNT SUMMARY")
print("="*80)

for aoi in aois:
    col_name = f'{aoi}_total_number_of_saccades'
    print(f"\n{aoi}:")
    print("-" * 70)
    
    successful = df[df['Success_Category'] == 'Successful'][col_name]
    unsuccessful = df[df['Success_Category'] == 'Unsuccessful'][col_name]
    
    print(f"  Successful   - Mean: {successful.mean():.2f}")
    print(f"  Unsuccessful - Mean: {unsuccessful.mean():.2f}")
    print(f"  Difference   - {successful.mean() - unsuccessful.mean():.2f}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Calculate which AOIs show biggest differences
aoi_differences = []
for aoi in aois:
    col_name = f'{aoi}_total_number_of_saccades'
    successful_mean = df[df['Success_Category'] == 'Successful'][col_name].mean()
    unsuccessful_mean = df[df['Success_Category'] == 'Unsuccessful'][col_name].mean()
    diff = successful_mean - unsuccessful_mean
    aoi_differences.append((aoi, diff, abs(diff)))

aoi_differences.sort(key=lambda x: x[2], reverse=True)

print("\nTop 3 AOIs with LARGEST differences in saccade count:")
for i, (aoi, diff, abs_diff) in enumerate(aoi_differences[:3], 1):
    direction = "MORE" if diff > 0 else "FEWER"
    print(f"  {i}. {aoi}: Successful pilots had {direction} saccades ({diff:+.2f})")

print("\n" + "="*80)
print("All visualizations created successfully!")
print("="*80)
print("\nGenerated files:")
print("  1. saccade_count_comparison.html")
print("  2. saccade_metrics_comparison.html")
print("  3. saccade_count_by_aoi.html")
print("  4. saccade_duration_heatmap.html")
print("  5. saccade_efficiency.html")
print("  6. saccade_velocity_by_aoi.html")