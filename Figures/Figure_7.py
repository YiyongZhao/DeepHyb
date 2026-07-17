import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 1. Text Configuration --------------------------
main_title = "Model confusion matrices"
cbar_label = "Proportion (row-normalized)"  

# Title dictionary: Descriptive titles (no S3/S9-S17 prefix in plot titles)
# Key = sheet name (for mapping), Value = display title
table_titles = {
    "S3": "Model (simulation dataset)",
    "S9": "Model 1 (butterfly-4-features-dataset1)",
    "S10": "Model 2 (butterfly-4-features-dataset2)",
    "S11": "Model 3 (butterfly-4-features-dataset3)",
    "S12": "Model 4 (butterfly-4-features-dataset4)",
    "S13": "Model 9 (butterfly-256-one-base-site-patterns-dataset1)",
    "S14": "Model 10 (butterfly-256-one-base-site-patterns-dataset2)",
    "S15": "Model 11 (butterfly-256-one-base-site-patterns-dataset3)",
    "S16": "Model 12 (butterfly-256-one-base-site-patterns-dataset4)",
    "S17": "Model 16 (butterfly-75-summary-kmer-site-patterns-dataset4)"
}

# -------------------------- 2. Adjustable Parameters --------------------------
gap_scale = 0.45    # Overall gap scaling (1.0=default; ↑=more spacing)
font_scale = 1.0    # Font scaling (1.0=default; ↑=larger text)
n_cols = 2          # Number of subplot columns
dpi = 300           # Save resolution (publication quality)

# -------------------------- 3. Font Settings --------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6 * font_scale       # Base font size
plt.rcParams['axes.labelsize'] = 8 * font_scale  # Axis label size
plt.rcParams['axes.titlesize'] = 10 * font_scale # Subplot title size
plt.rcParams['xtick.labelsize'] = 5 * font_scale # X-tick label size
plt.rcParams['ytick.labelsize'] = 5 * font_scale # Y-tick label size

# -------------------------- 4. Color Configuration --------------------------
# Unified colormap (0→light blue, 1→dark blue) for consistent comparison
cmap_unified = LinearSegmentedColormap.from_list("proportion_blue", ["#F0F8FF", "#0056b3"])
grid_color = "#E0E0E0"                           # Grid line color
text_color_high = "white"                        # Text color for high proportion (>0.5)
text_color_low = "black"                         # Text color for low proportion (≤0.5)
vmin = 0                                         # Minimum proportion value
vmax = 1                                         # Maximum proportion value

# -------------------------- 5. Core Functions --------------------------
def convert_to_proportion(cm):
    """
    Convert confusion matrix from raw counts to row-normalized proportion.
    Avoid division by zero by setting row sum to 1 if empty.
    
    Parameters:
        cm (np.array): Confusion matrix with raw counts
    Returns:
        np.array: Row-normalized proportion matrix
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    proportion_cm = cm / row_sums
    return proportion_cm

def load_confusion_matrix(excel_path, sheet_name):
    """
    Load and clean confusion matrix data from Excel sheet.
    
    Parameters:
        excel_path (str): Path to Excel file
        sheet_name (str): Name of target sheet (e.g., "S3", "S9")
    Returns:
        tuple: (count_data, proportion_data, labels, cols) if successful;
               (None, None, None, None) if failed
    """
    try:
        # Read Excel with format correction (match S3/S9-S17 structure)
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            skiprows=1,  # Skip title row
            header=0,    # Second row = column names
            index_col=0  # First column = row labels
        )
        # Clean invalid/missing data
        df = df.dropna(axis=0, how='all')  # Drop empty rows
        df = df[~df.index.isna()]          # Drop rows with empty index
        labels = df.index.tolist()         # Row labels (True class)
        cols = df.columns.tolist()         # Column labels (Predicted class)
        
        # Convert to integer (fill NaN with 0)
        count_data = df.fillna(0).apply(pd.to_numeric, errors='coerce').values.astype(int)
        # Convert to proportion matrix for color mapping
        proportion_data = convert_to_proportion(count_data)
        
        print(f"✅ Loaded {sheet_name}: Count Shape={count_data.shape}, Proportion Shape={proportion_data.shape}")
        return count_data, proportion_data, labels, cols
    
    except Exception as e:
        print(f"❌ Failed to load {sheet_name}: {str(e)[:50]}")
        return None, None, None, None

# -------------------------- 6. Load All Tables --------------------------
excel_path = "Supplementary_Tables20251205.xlsx"
tables = ["S3"] + [f"S{i}" for i in range(9, 18)]  # Target sheets: S3 + S9-S17
cm_data = {}  # Store matrix data: {sheet: (count_data, proportion_data, labels, cols)}

# Load data for each sheet
for sheet in tables:
    count_data, proportion_data, labels, cols = load_confusion_matrix(excel_path, sheet)
    if count_data is not None:
        cm_data[sheet] = (count_data, proportion_data, labels, cols)

# Filter valid tables (skip failed loads)
valid_tables = list(cm_data.keys())
n_tables = len(valid_tables)
n_rows = int(np.ceil(n_tables / n_cols))  # Calculate number of subplot rows

# Show "Predicted class" label only for last two valid subplots
last_two_indices = []
if n_tables >= 1:
    last_two_indices = [max(0, n_tables-2), n_tables-1]
last_two_indices = list(set(last_two_indices))  # Remove duplicates (for n_tables=1)

# -------------------------- 7. Plot All Confusion Matrices --------------------------
# Create subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols * gap_scale, 8 * n_rows * gap_scale))
axes = axes.flatten()  # Flatten 2D axes array for easy iteration

# Plot each valid table
for idx, sheet in enumerate(valid_tables):
    ax = axes[idx]
    count_data, proportion_data, labels, cols = cm_data[sheet]
    
    # Plot heatmap (use proportion data for color mapping)
    im = ax.imshow(proportion_data, cmap=cmap_unified, aspect="auto", vmin=vmin, vmax=vmax)
    
    # Set subplot title (only descriptive title, no S3/S9-S17 prefix)
    ax.set_title(table_titles.get(sheet, sheet), fontweight="bold", pad=10 * gap_scale)
    
    # Set axis labels
    ax.set_ylabel("True class", fontweight="bold", labelpad=8 * gap_scale)
    if idx in last_two_indices:
        ax.set_xlabel("Predicted class", fontweight="bold", labelpad=8 * gap_scale)
    
    # Set ticks and labels (rotate x-ticks to avoid overlap)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(cols, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    
    # Style optimization
    ax.spines['top'].set_visible(False)    # Hide top spine
    ax.spines['right'].set_visible(False)  # Hide right spine
    # Add light grid lines
    ax.grid(visible=True, color=grid_color, linestyle="-", linewidth=0.3, alpha=0.2)
    ax.set_axisbelow(True)                 # Place grid below heatmap
    
    # Add raw count annotations (only show values > 0)
    for i in range(len(labels)):
        for j in range(len(cols)):
            val_count = count_data[i, j]
            val_prop = proportion_data[i, j]
            if val_count > 0:
                # Choose text color based on proportion
                text_color = text_color_high if val_prop > 0.5 else text_color_low
                ax.text(
                    j, i, f"{val_count}", 
                    ha="center", va="center", 
                    color=text_color, fontsize=4 * font_scale
                )
    
    # Add colorbar for each subplot
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label(cbar_label, fontsize=7 * font_scale, fontweight="bold")
    cbar.ax.tick_params(labelsize=5 * font_scale)

# Hide empty subplots (if any)
for idx in range(n_tables, len(axes)):
    axes[idx].set_visible(False)

# -------------------------- 8. Global Layout & Save --------------------------
# Set main title
plt.suptitle(main_title, fontweight="bold", fontsize=12 * font_scale, y=0.98)
# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(
    top=0.95 if n_rows == 1 else 0.93,
    hspace=0.6 * gap_scale,
    wspace=0.5 * gap_scale,
    left=0.08,
    right=0.98,
    bottom=0.08
)

# Save as high-resolution SVG (vector format for publications)
plt.savefig(
    "Figure_7.svg",
    format="svg",
    dpi=dpi,
    bbox_inches="tight"
)

# Print completion message
print(f"\n✅ All valid confusion matrices saved!")
print(f"   Valid tables plotted: {valid_tables}")
print(f"   Predicted Class label shown for indices: {last_two_indices}")
print(f"   Output file: Figure_7.svg")