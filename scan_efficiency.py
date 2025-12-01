import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load main DGM data for pilot success categories
dgm_df = pd.read_csv('datasets/AOI_DGMs.csv')
dgm_df['Success_Category'] = dgm_df['Approach_Score'].apply(
    lambda x: 'Successful' if x >= 0.7 else 'Unsuccessful'
)

# Load pattern data from Excel sheets
collapsed_xls = pd.ExcelFile('datasets/Collapsed Patterns (Group).xlsx')

successful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Succesful Excluding No AOI(A)')
unsuccessful_collapsed = pd.read_excel(collapsed_xls, sheet_name='Unsuccesful Excluding No AOI(A)')

# AOI mapping
aoi_map = {
    'B': 'Alt_VSI',
    'C': 'AI',
    'D': 'TI_HSI',
    'E': 'SSI',
    'F': 'ASI',
    'G': 'RPM'
}

print("="*80)
print("SCAN EFFICIENCY ANALYSIS V2")
print("="*80)

# ============================================================================
# VIZ 1: CRITICAL TRANSITION DIFFERENCES
# What do successful pilots do differently?
# ============================================================================

def get_transition_differences(succ_df, unsucc_df):
    """Find which transitions differ most between groups"""
    pattern_col = 'Pattern String'
    freq_col = 'Proportional Pattern Frequency'

    succ_trans = {}
    unsucc_trans = {}

    # Build transition dicts
    for df, trans_dict in [(succ_df, succ_trans), (unsucc_df, unsucc_trans)]:
        for _, row in df.iterrows():
            pattern = str(row[pattern_col]).upper()
            frequency = row[freq_col]

            if pattern == 'nan':
                continue

            for i in range(len(pattern) - 1):
                key = (pattern[i], pattern[i+1])
                trans_dict[key] = trans_dict.get(key, 0) + frequency

    # Normalize by total transitions
    succ_total = sum(succ_trans.values())
    unsucc_total = sum(unsucc_trans.values())

    succ_trans = {k: v/succ_total*100 for k, v in succ_trans.items()}
    unsucc_trans = {k: v/unsucc_total*100 for k, v in unsucc_trans.items()}

    # Calculate differences
    all_transitions = set(list(succ_trans.keys()) + list(unsucc_trans.keys()))
    differences = []

    for trans in all_transitions:
        succ_pct = succ_trans.get(trans, 0)
        unsucc_pct = unsucc_trans.get(trans, 0)
        diff = succ_pct - unsucc_pct

        # Only keep meaningful differences
        if abs(diff) > 0.5:  # At least 0.5% difference
            from_aoi, to_aoi = trans
            differences.append({
                'Transition': f"{aoi_map.get(from_aoi, from_aoi)} -> {aoi_map.get(to_aoi, to_aoi)}",
                'Difference': diff,
                'Successful': succ_pct,
                'Unsuccessful': unsucc_pct
            })

    return sorted(differences, key=lambda x: abs(x['Difference']), reverse=True)[:12]

diff_data = get_transition_differences(successful_collapsed, unsuccessful_collapsed)
diff_df = pd.DataFrame(diff_data)

# Create diverging bar chart
colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff_df['Difference']]

fig1 = go.Figure()

fig1.add_trace(go.Bar(
    y=diff_df['Transition'],
    x=diff_df['Difference'],
    orientation='h',
    marker_color=colors,
    text=[f'{d:+.1f}pp' for d in diff_df['Difference']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>' +
                  'Successful: %{customdata[0]:.1f}%<br>' +
                  'Unsuccessful: %{customdata[1]:.1f}%<br>' +
                  'Difference: %{x:+.1f} percentage points<br>' +
                  '<extra></extra>',
    customdata=diff_df[['Successful', 'Unsuccessful']].values
))

fig1.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)

fig1.update_layout(
    title='What Scanning Behaviors Separate Success from Failure?<br>' +
          '<sub>Green = Successful pilots do this MORE | Red = Unsuccessful pilots do this MORE</sub>',
    xaxis_title='Difference in Transition Frequency (percentage points)',
    yaxis_title='',
    template='plotly_white',
    height=600,
    yaxis={'categoryorder': 'total ascending'}
)

fig1.write_html('scan_efficiency_transition_differences.html')
print("[OK] Created: scan_efficiency_transition_differences.html")

# ============================================================================
# VIZ 2: INSTRUMENT ATTENTION - Where do pilots focus?
# ============================================================================

def calculate_instrument_attention(pattern_df):
    """Calculate percentage of time each instrument appears in patterns"""
    pattern_col = 'Pattern String'
    freq_col = 'Proportional Pattern Frequency'

    appearances = {}
    total = 0

    for _, row in pattern_df.iterrows():
        pattern = str(row[pattern_col]).upper()
        frequency = row[freq_col]

        if pattern == 'nan':
            continue

        # Count each AOI appearance in this pattern
        for aoi in pattern:
            appearances[aoi] = appearances.get(aoi, 0) + frequency
            total += frequency

    # Convert to percentages
    return {k: v/total*100 for k, v in appearances.items()}

succ_attention = calculate_instrument_attention(successful_collapsed)
unsucc_attention = calculate_instrument_attention(unsuccessful_collapsed)

attention_data = []
for aoi in aoi_map.keys():
    attention_data.append({
        'Instrument': aoi_map[aoi],
        'Successful': succ_attention.get(aoi, 0),
        'Unsuccessful': unsucc_attention.get(aoi, 0)
    })

attention_df = pd.DataFrame(attention_data)
attention_df['Difference'] = attention_df['Successful'] - attention_df['Unsuccessful']
attention_df = attention_df.sort_values('Successful', ascending=False)

fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Instrument Attention Distribution', 'Difference (Successful - Unsuccessful)'),
    specs=[[{"type": "bar"}, {"type": "bar"}]],
    horizontal_spacing=0.15,
    column_widths=[0.55, 0.45]
)

# Left: Grouped bar
fig2.add_trace(go.Bar(
    x=attention_df['Instrument'],
    y=attention_df['Successful'],
    name='Successful',
    marker_color='#2ecc71',
    text=[f'{v:.1f}%' for v in attention_df['Successful']],
    textposition='outside'
), row=1, col=1)

fig2.add_trace(go.Bar(
    x=attention_df['Instrument'],
    y=attention_df['Unsuccessful'],
    name='Unsuccessful',
    marker_color='#e74c3c',
    text=[f'{v:.1f}%' for v in attention_df['Unsuccessful']],
    textposition='outside'
), row=1, col=1)

# Right: Difference
colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in attention_df['Difference']]
fig2.add_trace(go.Bar(
    x=attention_df['Instrument'],
    y=attention_df['Difference'],
    marker_color=colors,
    showlegend=False,
    text=[f'{d:+.1f}pp' for d in attention_df['Difference']],
    textposition='outside'
), row=1, col=2)

fig2.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2, line_width=2)

fig2.update_xaxes(title_text="", row=1, col=1)
fig2.update_xaxes(title_text="", row=1, col=2)
fig2.update_yaxes(title_text="% of Fixations", row=1, col=1)
fig2.update_yaxes(title_text="Percentage Points", row=1, col=2)

fig2.update_layout(
    title='Which Instruments Get The Most Attention?',
    height=500,
    template='plotly_white',
    barmode='group'
)

fig2.write_html('scan_efficiency_instrument_attention.html')
print("[OK] Created: scan_efficiency_instrument_attention.html")

# ============================================================================
# VIZ 3: SCAN PATH EFFICIENCY - Direct vs Backtracking
# ============================================================================

def analyze_scan_efficiency(pattern_df):
    """
    Measure scan path efficiency:
    - Backtrack rate: How often do they return to the previous instrument?
    - Cycle efficiency: avg transitions per unique instrument visited
    """
    pattern_col = 'Pattern String'
    freq_col = 'Proportional Pattern Frequency'

    total_backtracks = 0
    total_transitions = 0

    pattern_lengths = []
    unique_instruments = []

    for _, row in pattern_df.iterrows():
        pattern = str(row[pattern_col]).upper()
        frequency = row[freq_col]

        if pattern == 'nan':
            continue

        # Count immediate backtracks (A->B->A pattern)
        backtracks = 0
        for i in range(len(pattern) - 2):
            if pattern[i] == pattern[i+2] and pattern[i] != pattern[i+1]:
                backtracks += 1

        total_backtracks += backtracks * frequency
        total_transitions += (len(pattern) - 1) * frequency

        # Pattern metrics
        pattern_lengths.extend([len(pattern)] * int(frequency * 1000))
        unique_instruments.extend([len(set(pattern))] * int(frequency * 1000))

    backtrack_rate = (total_backtracks / total_transitions * 100) if total_transitions > 0 else 0
    avg_pattern_len = np.mean(pattern_lengths)
    avg_unique = np.mean(unique_instruments)
    efficiency = (avg_unique / avg_pattern_len * 100) if avg_pattern_len > 0 else 0

    return {
        'backtrack_rate': backtrack_rate,
        'avg_pattern_length': avg_pattern_len,
        'avg_unique_instruments': avg_unique,
        'efficiency_score': efficiency
    }

succ_eff = analyze_scan_efficiency(successful_collapsed)
unsucc_eff = analyze_scan_efficiency(unsuccessful_collapsed)

# Create comparison
metrics = ['Backtrack Rate (%)', 'Avg Pattern Length', 'Avg Unique Instruments', 'Efficiency Score (%)']
succ_values = [succ_eff['backtrack_rate'], succ_eff['avg_pattern_length'],
               succ_eff['avg_unique_instruments'], succ_eff['efficiency_score']]
unsucc_values = [unsucc_eff['backtrack_rate'], unsucc_eff['avg_pattern_length'],
                 unsucc_eff['avg_unique_instruments'], unsucc_eff['efficiency_score']]

fig3 = go.Figure()

x = np.arange(len(metrics))
width = 0.35

fig3.add_trace(go.Bar(
    x=metrics,
    y=succ_values,
    name='Successful',
    marker_color='#2ecc71',
    text=[f'{v:.2f}' for v in succ_values],
    textposition='outside'
))

fig3.add_trace(go.Bar(
    x=metrics,
    y=unsucc_values,
    name='Unsuccessful',
    marker_color='#e74c3c',
    text=[f'{v:.2f}' for v in unsucc_values],
    textposition='outside'
))

fig3.update_layout(
    title='Scan Path Efficiency Comparison<br>' +
          '<sub>Lower backtrack = more decisive | Higher efficiency = better coverage</sub>',
    yaxis_title='Value',
    xaxis_title='',
    barmode='group',
    template='plotly_white',
    height=500
)

fig3.write_html('scan_efficiency_metrics.html')
print("[OK] Created: scan_efficiency_metrics.html")

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("KEY FINDINGS - SCAN EFFICIENCY")
print("="*80)

print("\n1. INSTRUMENT ATTENTION")
print("-" * 70)
top_instrument = attention_df.iloc[0]
print(f"   Most scanned overall: {top_instrument['Instrument']}")
print(f"      Successful: {top_instrument['Successful']:.1f}%")
print(f"      Unsuccessful: {top_instrument['Unsuccessful']:.1f}%")

biggest_diff = attention_df.loc[attention_df['Difference'].abs().idxmax()]
print(f"\n   Biggest difference: {biggest_diff['Instrument']} ({biggest_diff['Difference']:+.1f}pp)")
if biggest_diff['Difference'] > 0:
    print(f"      -> Successful pilots focus MORE on {biggest_diff['Instrument']}")
else:
    print(f"      -> Unsuccessful pilots focus MORE on {biggest_diff['Instrument']}")

print("\n2. SCANNING EFFICIENCY")
print("-" * 70)
print(f"   Backtrack Rate:")
print(f"      Successful: {succ_eff['backtrack_rate']:.1f}%")
print(f"      Unsuccessful: {unsucc_eff['backtrack_rate']:.1f}%")
backtrack_diff = unsucc_eff['backtrack_rate'] - succ_eff['backtrack_rate']
print(f"      -> Unsuccessful pilots backtrack {backtrack_diff:+.1f}pp MORE")

print(f"\n   Efficiency Score (unique instruments / pattern length):")
print(f"      Successful: {succ_eff['efficiency_score']:.1f}%")
print(f"      Unsuccessful: {unsucc_eff['efficiency_score']:.1f}%")
eff_diff = succ_eff['efficiency_score'] - unsucc_eff['efficiency_score']
print(f"      -> Successful pilots are {eff_diff:+.1f}pp more efficient")

print("\n3. CRITICAL TRANSITIONS (Top 3 differences)")
print("-" * 70)
for idx, row in diff_df.head(3).iterrows():
    direction = "MORE" if row['Difference'] > 0 else "LESS"
    who = "Successful" if row['Difference'] > 0 else "Unsuccessful"
    print(f"   {row['Transition']}")
    print(f"      {who} pilots do this {abs(row['Difference']):.1f}pp {direction}")

print("\n" + "="*80)
print("MAIN TAKEAWAY:")
print("="*80)

print(f"\nSuccessful pilots demonstrate:")
print(f"  - {abs(backtrack_diff):.1f}pp LESS backtracking (more decisive)")
print(f"  - {abs(eff_diff):.1f}pp HIGHER efficiency (better coverage per transition)")
print(f"  - {abs(biggest_diff['Difference']):.1f}pp {'MORE' if biggest_diff['Difference'] > 0 else 'LESS'} attention on {biggest_diff['Instrument']}")

print("\n" + "="*80)
print(f"Generated 3 focused visualizations")
print("="*80)
