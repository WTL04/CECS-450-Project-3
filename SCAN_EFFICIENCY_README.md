# Scan Efficiency Analysis

**Author:** Russell
**Date:** November 30, 2025

## Overview

This analysis examines the scanning efficiency and AOI transition patterns between successful and unsuccessful pilots during ILS approaches. It addresses the research question: *"What scan behaviors characterize successful approaches?"*

## Files

- **scan_efficiency.py** - Main analysis script
- **scan_efficiency_transition_matrix.html** - Heatmap showing AOI-to-AOI transition frequencies
- **scan_efficiency_metrics.html** - Pattern length and AOI coverage distributions
- **scan_efficiency_top_transitions.html** - Top 15 most frequent transitions compared
- **scan_efficiency_complexity.html** - Collapsed vs expanded pattern length comparison
- **scan_efficiency_network.html** - Network diagram of successful pilot transitions

## Key Findings

### 1. Scan Pattern Length
- **Successful pilots had LONGER scan patterns (+5.3%)**
- Successful: Mean = 3.37 AOIs per pattern
- Unsuccessful: Mean = 3.20 AOIs per pattern
- **Insight:** Successful pilots perform more comprehensive scans

### 2. AOI Coverage
- **Successful pilots covered MORE unique AOIs (+2.0%)**
- Successful: Mean = 2.66 unique AOIs
- Unsuccessful: Mean = 2.60 unique AOIs
- **Insight:** Successful pilots have broader situational awareness

### 3. Scanning Repetition
- **Successful pilots had LESS repetitive fixations (-6.4%)**
- Successful expansion ratio: 1.080
- Unsuccessful expansion ratio: 1.147
- **Insight:** Successful pilots make more efficient use of their fixations

### 4. Top Instrument Transitions

**Successful Pilots Most Frequent:**
1. AI → TI_HSI (27.2% of transitions)
2. AI → Alt_VSI (26.8%)
3. Alt_VSI → AI (26.6%)
4. TI_HSI → AI (24.1%)
5. AI → ASI (12.6%)

**Unsuccessful Pilots Most Frequent:**
1. Alt_VSI → AI (14.4%)
2. AI → Alt_VSI (14.1%)
3. AI → TI_HSI (12.5%)
4. TI_HSI → AI (11.3%)
5. AI → ASI (8.5%)

**Key Difference:** Successful pilots have **54% higher overall transition frequency**, indicating more active scanning behavior.

## Methodology

### Data Sources
- **Collapsed Patterns (Group).xlsx** - Pattern sequences with repeated AOIs removed
- **Expanded Patterns (Group).xlsx** - Full pattern sequences including repetitions
- **AOI_DGMs.csv** - Descriptive gaze measures per pilot

### Analysis Techniques

1. **Transition Matrix Analysis**
   - Built weighted transition frequency matrices from pattern data
   - Normalized by row to show transition probabilities
   - Visualized as heatmaps for easy comparison

2. **Pattern Efficiency Metrics**
   - Pattern length: Total AOIs in collapsed sequences
   - Unique AOI coverage: Distinct instruments scanned
   - Weighted by proportional pattern frequency

3. **Complexity Analysis**
   - Compared expanded vs collapsed pattern lengths
   - Ratio indicates repetitive fixation behavior
   - Lower ratio = more efficient scanning

4. **Network Visualization**
   - Nodes represent AOIs
   - Edges represent transitions
   - Shows scanning flow patterns

## Visualization Design Decisions

### Color Coding
- **Green (#2ecc71)** - Successful pilots
- **Red (#e74c3c)** - Unsuccessful pilots
- **Blue (#3498db)** - Collapsed patterns
- **Purple (#9b59b6)** - Expanded patterns

### Interactive Features
- **Hover tooltips** - Show exact values and percentages
- **Box plots** - Display distribution statistics (mean, SD, outliers)
- **Grouped bar charts** - Enable direct category comparison

## Topics from CECS 450

### Visual Encodings
- **Position** - Transition matrix rows/columns encode from/to AOIs
- **Color saturation** - Encodes transition frequency in heatmaps
- **Length** - Bar charts encode frequency and pattern length
- **Network layout** - Circular layout for transition network

### Interaction Techniques
- **Hover** - Details-on-demand for all visualizations
- **Filtering** - Analysis excludes "No_AOI" to focus on instrument scanning
- **Comparison** - Side-by-side layouts enable successful vs unsuccessful comparison

### Data Processing
- **Aggregation** - Patterns weighted by proportional frequency
- **Normalization** - Transition matrices normalized by row
- **Transformation** - Collapsed vs expanded for efficiency analysis

## Running the Analysis

```bash
python scan_efficiency.py
```

**Requirements:**
- pandas
- plotly
- numpy
- openpyxl

## Interpretation Guide

### Transition Matrix Heatmap
- Darker colors = more frequent transitions
- Diagonal = self-transitions (fixating same AOI multiple times)
- Compare row patterns between successful/unsuccessful

### Pattern Efficiency Metrics
- Box plots show distribution and outliers
- Mean line shows average behavior
- SD shows consistency of scanning patterns

### Top Transitions Bar Chart
- Longer bars = more frequent transitions
- Look for transitions unique to successful pilots
- Indicates critical instrument checking sequences

### Complexity Comparison
- Gap between collapsed and expanded = repetitiveness
- Smaller gap = efficient, decisive scanning
- Larger gap = revisiting same instruments repeatedly

### Network Diagram
- Edge density shows scanning activity
- Clusters indicate related instruments scanned together
- Node centrality indicates critical instruments

## Conclusions

Successful pilots demonstrate:
1. **More comprehensive scanning** - Longer, more complete patterns
2. **Better situational awareness** - Cover more unique instruments
3. **Higher efficiency** - Less repetitive fixations
4. **More active monitoring** - 54% more transitions overall

The most critical transition for successful pilots is **AI ↔ TI_HSI**, suggesting that maintaining awareness of both attitude and navigation is crucial for successful ILS approaches.
