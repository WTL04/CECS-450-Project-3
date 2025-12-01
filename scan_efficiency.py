"""
Scan Efficiency Analysis
Analyzes AOI transition patterns and scanning efficiency between successful and unsuccessful pilots
Author: Russell
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

# Load main DGM data for pilot success categories
dgm_df = pd.read_csv('datasets/AOI_DGMs.csv')
dgm_df['Success_Category'] = dgm_df['Approach_Score'].apply(
    lambda x: 'Successful' if x >= 0.7 else 'Unsuccessful'
)

# Load pattern data from Excel sheets (handling misspelling in sheets)
collapsed_xls = pd.ExcelFile('datasets/Collapsed Patterns (Group).xlsx')
expanded_xls = pd.ExcelFile('datasets/Expanded Patterns (Group).xlsx')

# Map to actual sheet names (with misspellings)
successful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Succesful Excluding No AOI(A)')
unsuccessful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Unsuccesful Excluding No AOI(A)')
successful_expanded = pd.read_excel(expanded_xls, sheet_name='Succesful Excluding No AOI(A)')
unsuccessful_expanded = pd.read_excel(expanded_xls, sheet_name='Unsuccesful Excluding No AOI(A)')

# AOI mapping
aoi_map = {
    'A': 'No_AOI',
    'B': 'Alt_VSI',
    'C': 'AI',
    'D': 'TI_HSI',
    'E': 'SSI',
    'F': 'ASI',
    'G': 'RPM',
    'H': 'Window'
}

# Instrument AOIs only (excluding No_AOI and Window for core analysis)
instrument_aois = ['B', 'C', 'D', 'E', 'F', 'G']  # Alt_VSI, AI, TI_HSI, SSI, ASI, RPM

print("="*80)
print("SCAN EFFICIENCY ANALYSIS")
print("="*80)

# ============================================================================
# ANALYSIS 1: AOI Transition Matrix
# ============================================================================

def build_transition_matrix(pattern_df):
    """Build transition count matrix from pattern sequences weighted by frequency"""
    transitions = {}

    pattern_col = 'Pattern String'
    freq_col = 'Proportional Pattern Frequency'

    for _, row in pattern_df.iterrows():
        pattern = row[pattern_col]
        frequency = row[freq_col]

        if pd.isna(pattern):
            continue

        pattern = str(pattern).upper()
        # Build transitions from this pattern, weighted by its frequency
        for i in range(len(pattern) - 1):
            from_aoi = pattern[i]
            to_aoi = pattern[i + 1]
            key = (from_aoi, to_aoi)
            transitions[key] = transitions.get(key, 0) + frequency

    return transitions

# Build transition matrices for each category using collapsed patterns
successful_transitions = build_transition_matrix(successful_collapsed)
unsuccessful_transitions = build_transition_matrix(unsuccessful_collapsed)

# Create transition matrix visualization
all_aois = sorted(set([k[0] for k in successful_transitions.keys()] +
                      [k[1] for k in successful_transitions.keys()] +
                      [k[0] for k in unsuccessful_transitions.keys()] +
                      [k[1] for k in unsuccessful_transitions.keys()]))

def create_matrix(transitions, aois):
    """Convert transition dict to matrix"""
    matrix = np.zeros((len(aois), len(aois)))
    for i, from_aoi in enumerate(aois):
        for j, to_aoi in enumerate(aois):
            matrix[i, j] = transitions.get((from_aoi, to_aoi), 0)
    return matrix

successful_matrix = create_matrix(successful_transitions, all_aois)
unsuccessful_matrix = create_matrix(unsuccessful_transitions, all_aois)

# Normalize by row (from each AOI, what % goes to each other AOI)
successful_matrix_norm = successful_matrix / (successful_matrix.sum(axis=1, keepdims=True) + 1e-10) * 100
unsuccessful_matrix_norm = unsuccessful_matrix / (unsuccessful_matrix.sum(axis=1, keepdims=True) + 1e-10) * 100

# Create heatmap comparison
aoi_labels = [aoi_map.get(a, a) for a in all_aois]

fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Successful Pilots', 'Unsuccessful Pilots'),
    horizontal_spacing=0.12
)

fig1.add_trace(
    go.Heatmap(
        z=successful_matrix_norm,
        x=aoi_labels,
        y=aoi_labels,
        colorscale='Greens',
        text=successful_matrix_norm.round(1),
        texttemplate='%{text}%',
        textfont={"size": 9},
        hovertemplate='From: %{y}<br>To: %{x}<br>Frequency: %{z:.1f}%<extra></extra>',
        showscale=True,
        colorbar=dict(x=0.46, title="% of<br>Transitions")
    ),
    row=1, col=1
)

fig1.add_trace(
    go.Heatmap(
        z=unsuccessful_matrix_norm,
        x=aoi_labels,
        y=aoi_labels,
        colorscale='Reds',
        text=unsuccessful_matrix_norm.round(1),
        texttemplate='%{text}%',
        textfont={"size": 9},
        hovertemplate='From: %{y}<br>To: %{x}<br>Frequency: %{z:.1f}%<extra></extra>',
        showscale=True,
        colorbar=dict(x=1.02, title="% of<br>Transitions")
    ),
    row=1, col=2
)

fig1.update_xaxes(title_text="To AOI", row=1, col=1)
fig1.update_xaxes(title_text="To AOI", row=1, col=2)
fig1.update_yaxes(title_text="From AOI", row=1, col=1)
fig1.update_yaxes(title_text="From AOI", row=1, col=2)

fig1.update_layout(
    title='AOI Transition Frequency Heatmap (Normalized by Row)',
    height=600,
    template='plotly_white'
)

fig1.write_html('scan_efficiency_transition_matrix.html')
print("[OK] Created: scan_efficiency_transition_matrix.html")

# ============================================================================
# ANALYSIS 2: Transition Efficiency Metrics
# ============================================================================

def calculate_pattern_metrics(pattern_df, category):
    """
    Calculate weighted average efficiency metrics from pattern data
    """
    pattern_col = 'Pattern String'
    freq_col = 'Proportional Pattern Frequency'

    lengths = []
    unique_counts = []

    for _, row in pattern_df.iterrows():
        pattern = row[pattern_col]
        frequency = row[freq_col]

        if pd.isna(pattern):
            continue

        pattern = str(pattern).upper()
        length = len(pattern)
        unique_aois = len(set(pattern))

        # Weight by frequency
        lengths.extend([length] * int(frequency * 1000))
        unique_counts.extend([unique_aois] * int(frequency * 1000))

    return lengths, unique_counts

# Calculate efficiency metrics for both categories
succ_lengths, succ_unique = calculate_pattern_metrics(successful_collapsed, 'Successful')
unsucc_lengths, unsucc_unique = calculate_pattern_metrics(unsuccessful_collapsed, 'Unsuccessful')

# Build dataframe for visualization
efficiency_data = []
for length in succ_lengths:
    efficiency_data.append({'Category': 'Successful', 'Pattern_Length': length})
for length in unsucc_lengths:
    efficiency_data.append({'Category': 'Unsuccessful', 'Pattern_Length': length})

efficiency_df = pd.DataFrame(efficiency_data)

unique_data = []
for unique in succ_unique:
    unique_data.append({'Category': 'Successful', 'Unique_AOIs': unique})
for unique in unsucc_unique:
    unique_data.append({'Category': 'Unsuccessful', 'Unique_AOIs': unique})

unique_df = pd.DataFrame(unique_data)

# Visualization: Pattern Efficiency Comparison
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Pattern Length Distribution', 'Unique AOI Coverage')
)

# Pattern Length
for category, color in [('Successful', '#2ecc71'), ('Unsuccessful', '#e74c3c')]:
    data = efficiency_df[efficiency_df['Category'] == category]['Pattern_Length']

    fig2.add_trace(
        go.Box(
            y=data,
            name=category,
            marker_color=color,
            showlegend=True,
            boxmean='sd'
        ),
        row=1, col=1
    )

# Unique AOIs
for category, color in [('Successful', '#2ecc71'), ('Unsuccessful', '#e74c3c')]:
    data = unique_df[unique_df['Category'] == category]['Unique_AOIs']

    fig2.add_trace(
        go.Box(
            y=data,
            name=category,
            marker_color=color,
            showlegend=False,
            boxmean='sd'
        ),
        row=1, col=2
    )

fig2.update_yaxes(title_text='Pattern Length', row=1, col=1)
fig2.update_yaxes(title_text='Unique AOIs', row=1, col=2)

fig2.update_layout(
    title='Scan Pattern Efficiency Metrics (Collapsed Patterns, Excluding No_AOI)',
    height=500,
    template='plotly_white'
)

fig2.write_html('scan_efficiency_metrics.html')
print("[OK] Created: scan_efficiency_metrics.html")

# ============================================================================
# ANALYSIS 3: Top Transition Pairs Comparison
# ============================================================================

def get_top_transitions(transitions, n=10):
    """Get top N most frequent transitions"""
    sorted_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    return sorted_trans[:n]

top_successful = get_top_transitions(successful_transitions, 15)
top_unsuccessful = get_top_transitions(unsuccessful_transitions, 15)

# Create comparison bar chart
transition_comparison = []

# Get all unique transitions from top lists
all_top_transitions = set([t[0] for t in top_successful] + [t[0] for t in top_unsuccessful])

for trans in all_top_transitions:
    from_aoi, to_aoi = trans
    label = f"{aoi_map.get(from_aoi, from_aoi)} → {aoi_map.get(to_aoi, to_aoi)}"

    succ_count = successful_transitions.get(trans, 0)
    unsucc_count = unsuccessful_transitions.get(trans, 0)

    transition_comparison.append({
        'Transition': label,
        'Successful': succ_count,
        'Unsuccessful': unsucc_count,
        'Difference': succ_count - unsucc_count
    })

transition_comp_df = pd.DataFrame(transition_comparison)
transition_comp_df = transition_comp_df.sort_values('Difference', ascending=False).head(15)

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    y=transition_comp_df['Transition'],
    x=transition_comp_df['Successful'],
    name='Successful',
    orientation='h',
    marker_color='#2ecc71'
))

fig3.add_trace(go.Bar(
    y=transition_comp_df['Transition'],
    x=transition_comp_df['Unsuccessful'],
    name='Unsuccessful',
    orientation='h',
    marker_color='#e74c3c'
))

fig3.update_layout(
    title='Top 15 AOI Transitions: Successful vs Unsuccessful Pilots',
    xaxis_title='Transition Count',
    yaxis_title='AOI Transition',
    barmode='group',
    template='plotly_white',
    height=700,
    yaxis={'categoryorder': 'total ascending'}
)

fig3.write_html('scan_efficiency_top_transitions.html')
print("[OK] Created: scan_efficiency_top_transitions.html")

# ============================================================================
# ANALYSIS 4: Average Pattern Length Comparison (Expanded vs Collapsed)
# ============================================================================

def calc_avg_pattern_length(pattern_df):
    """Calculate weighted average pattern length"""
    pattern_col = 'Pattern String'
    freq_col = 'Proportional Pattern Frequency'

    total_length = 0
    total_freq = 0

    for _, row in pattern_df.iterrows():
        pattern = row[pattern_col]
        frequency = row[freq_col]

        if pd.isna(pattern):
            continue

        pattern = str(pattern).upper()
        total_length += len(pattern) * frequency
        total_freq += frequency

    return total_length / total_freq if total_freq > 0 else 0

# Calculate average lengths
succ_collapsed_avg = calc_avg_pattern_length(successful_collapsed)
unsucc_collapsed_avg = calc_avg_pattern_length(unsuccessful_collapsed)
succ_expanded_avg = calc_avg_pattern_length(successful_expanded)
unsucc_expanded_avg = calc_avg_pattern_length(unsuccessful_expanded)

# Create comparison visualization
fig4 = go.Figure()

categories = ['Successful', 'Unsuccessful']
collapsed_vals = [succ_collapsed_avg, unsucc_collapsed_avg]
expanded_vals = [succ_expanded_avg, unsucc_expanded_avg]

fig4.add_trace(go.Bar(
    x=categories,
    y=collapsed_vals,
    name='Collapsed Patterns',
    marker_color='#3498db',
    text=[f'{v:.2f}' for v in collapsed_vals],
    textposition='auto'
))

fig4.add_trace(go.Bar(
    x=categories,
    y=expanded_vals,
    name='Expanded Patterns',
    marker_color='#9b59b6',
    text=[f'{v:.2f}' for v in expanded_vals],
    textposition='auto'
))

fig4.update_layout(
    title='Average Pattern Length: Collapsed vs Expanded<br><sub>Larger gap = more repetitive scanning within same AOIs</sub>',
    yaxis_title='Average Pattern Length',
    xaxis_title='Pilot Category',
    barmode='group',
    template='plotly_white',
    height=500
)

fig4.write_html('scan_efficiency_complexity.html')
print("[OK] Created: scan_efficiency_complexity.html")

# ============================================================================
# ANALYSIS 5: Network Diagram of Transitions
# ============================================================================

# Create network graph showing transition flows for successful pilots
import plotly.graph_objects as go

# Get top transitions for successful pilots
top_trans = get_top_transitions(successful_transitions, 20)

# Build edge list
edge_x = []
edge_y = []
edge_text = []

# Position nodes in a circle
node_positions = {}
n_nodes = len(all_aois)
for i, aoi in enumerate(all_aois):
    angle = 2 * np.pi * i / n_nodes
    node_positions[aoi] = (np.cos(angle), np.sin(angle))

# Create edges
for (from_aoi, to_aoi), count in top_trans:
    x0, y0 = node_positions[from_aoi]
    x1, y1 = node_positions[to_aoi]

    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_text.append(f"{aoi_map.get(from_aoi, from_aoi)}→{aoi_map.get(to_aoi, to_aoi)}: {count}")

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = [node_positions[aoi][0] for aoi in all_aois]
node_y = [node_positions[aoi][1] for aoi in all_aois]
node_text = [aoi_map.get(aoi, aoi) for aoi in all_aois]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color='#2ecc71',
        size=30,
        line=dict(width=2, color='white')
    )
)

fig5 = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='AOI Transition Network - Successful Pilots (Top 20 Transitions)',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                template='plotly_white'
                ))

fig5.write_html('scan_efficiency_network.html')
print("[OK] Created: scan_efficiency_network.html")

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SCAN EFFICIENCY SUMMARY STATISTICS")
print("="*80)

print("\n--- PATTERN EFFICIENCY METRICS (Collapsed, Excluding No_AOI) ---")
print("\nPattern Length:")
print("-" * 70)
successful_len = efficiency_df[efficiency_df['Category'] == 'Successful']['Pattern_Length']
unsuccessful_len = efficiency_df[efficiency_df['Category'] == 'Unsuccessful']['Pattern_Length']

print(f"  Successful   - Mean: {successful_len.mean():.3f}, SD: {successful_len.std():.3f}")
print(f"  Unsuccessful - Mean: {unsuccessful_len.mean():.3f}, SD: {unsuccessful_len.std():.3f}")
print(f"  Difference   - {successful_len.mean() - unsuccessful_len.mean():.3f}")

if unsuccessful_len.mean() != 0:
    diff_percent = ((successful_len.mean() - unsuccessful_len.mean()) / unsuccessful_len.mean()) * 100
    print(f"  % Difference - {diff_percent:.1f}%")

print("\nUnique AOI Coverage:")
print("-" * 70)
successful_unique = unique_df[unique_df['Category'] == 'Successful']['Unique_AOIs']
unsuccessful_unique = unique_df[unique_df['Category'] == 'Unsuccessful']['Unique_AOIs']

print(f"  Successful   - Mean: {successful_unique.mean():.3f}, SD: {successful_unique.std():.3f}")
print(f"  Unsuccessful - Mean: {unsuccessful_unique.mean():.3f}, SD: {unsuccessful_unique.std():.3f}")
print(f"  Difference   - {successful_unique.mean() - unsuccessful_unique.mean():.3f}")

if unsuccessful_unique.mean() != 0:
    diff_percent = ((successful_unique.mean() - unsuccessful_unique.mean()) / unsuccessful_unique.mean()) * 100
    print(f"  % Difference - {diff_percent:.1f}%")

print("\n--- AVERAGE PATTERN LENGTHS (Collapsed vs Expanded) ---")
print("-" * 70)
print(f"  Successful   - Collapsed: {succ_collapsed_avg:.3f}, Expanded: {succ_expanded_avg:.3f}")
print(f"                 Ratio: {succ_expanded_avg/succ_collapsed_avg:.3f}")
print(f"  Unsuccessful - Collapsed: {unsucc_collapsed_avg:.3f}, Expanded: {unsucc_expanded_avg:.3f}")
print(f"                 Ratio: {unsucc_expanded_avg/unsucc_collapsed_avg:.3f}")

print("\n--- TOP 5 TRANSITIONS FOR SUCCESSFUL PILOTS ---")
print("-" * 70)
for i, ((from_aoi, to_aoi), count) in enumerate(top_successful[:5], 1):
    print(f"  {i}. {aoi_map.get(from_aoi, from_aoi)} -> {aoi_map.get(to_aoi, to_aoi)}: {count:.3f} transitions")

print("\n--- TOP 5 TRANSITIONS FOR UNSUCCESSFUL PILOTS ---")
print("-" * 70)
for i, ((from_aoi, to_aoi), count) in enumerate(top_unsuccessful[:5], 1):
    print(f"  {i}. {aoi_map.get(from_aoi, from_aoi)} -> {aoi_map.get(to_aoi, to_aoi)}: {count:.3f} transitions")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Calculate key differences
avg_pattern_diff = successful_len.mean() - unsuccessful_len.mean()
avg_unique_diff = successful_unique.mean() - unsuccessful_unique.mean()
succ_complexity_ratio = succ_expanded_avg / succ_collapsed_avg
unsucc_complexity_ratio = unsucc_expanded_avg / unsucc_collapsed_avg
avg_complexity_diff = succ_complexity_ratio - unsucc_complexity_ratio

print(f"\n1. Pattern Length: Successful pilots had {'SHORTER' if avg_pattern_diff < 0 else 'LONGER'} scan patterns")
print(f"   ({avg_pattern_diff:+.2f} AOIs on average)")

print(f"\n2. AOI Coverage: Successful pilots covered {'MORE' if avg_unique_diff > 0 else 'FEWER'} unique AOIs")
print(f"   ({avg_unique_diff:+.2f} unique AOIs on average)")

print(f"\n3. Scanning Repetition: Successful pilots had {'MORE' if avg_complexity_diff > 0 else 'LESS'} repetitive fixations")
print(f"   (Expansion ratio difference: {avg_complexity_diff:+.2f})")

# Total transitions
total_succ_trans = sum(successful_transitions.values())
total_unsucc_trans = sum(unsuccessful_transitions.values())
print(f"\n4. Total Transition Frequency: Successful: {total_succ_trans:.2f}, Unsuccessful: {total_unsucc_trans:.2f}")

print("\n" + "="*80)
print("All scan efficiency visualizations created successfully!")
print("="*80)
print("\nGenerated files:")
print("  1. scan_efficiency_transition_matrix.html - Heatmap of AOI-to-AOI transitions")
print("  2. scan_efficiency_metrics.html - Pattern efficiency metrics comparison")
print("  3. scan_efficiency_top_transitions.html - Top transition pairs comparison")
print("  4. scan_efficiency_complexity.html - Scan path complexity analysis")
print("  5. scan_efficiency_network.html - Network diagram of successful pilot transitions")
