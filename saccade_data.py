import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the data
df = pd.read_csv('datasets/AOI_DGMs.csv')

# Check what saccade columns we have
saccade_cols = [col for col in df.columns if 'saccade' in col.lower()]
print("Available saccade columns:")
for col in saccade_cols:
    print(f"  - {col}")