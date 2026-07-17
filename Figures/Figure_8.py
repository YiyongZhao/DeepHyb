import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import matplotlib.patches as mpatches

# -------------------------- 1. Basic Configuration --------------------------
main_title = "Error Pattern Analysis (S9-S17) - Error Count Proportion"
table_titles = {
    "S9": "Model 1 (butterfly-4-features-dataset1)",
    "S10": "Model 2 (butterfly-4-features-dataset2)",
    "S11": "Model 3 (butterfly-4-features-dataset3)",
    "S12": "Model 4 (butterfly-4-features-dataset4)",
    "S13": "Model 9 (butterfly-256-one-base-site-patterns-dataset1)",
    "S14": "Model 10 (butterfly-256-one-base-site-patterns-dataset2)",
    "S15": "Model 11 (butterfly-256-one-base-site-patterns-dataset3)",
    "S16": "Model 12 (butterfly-256-one-base-site-patterns-dataset4)",
    "S17": "Model 16 (butterfly-256-seq-kmer-patterns-patterns-dataset4)"
}

# Sheet专属颜色（S9-S17）
sheet_colors = {
    "S9": "#ff7f0e", "S10": "#2ca02c", "S11": "#d62728", 
    "S12": "#9467bd", "S13": "#8c564b", "S14": "#e377c2", 
    "S15": "#7f7f7f", "S16": "#bcbd22", "S17": "#17becf"
}

dpi = 300

# Font Settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

# Color Configuration
grid_color = "#E0E0E0"

# -------------------------- 2. Core Functions (Error Proportion Focus) --------------------------
def convert_to_proportion(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    proportion_cm = cm / row_sums
    return proportion_cm

def load_confusion_matrix(excel_path, sheet_name):
    """Load single confusion matrix (raw counts)"""
    try:
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            skiprows=1,
            header=0,
            index_col=0
        )
        df = df.dropna(axis=0, how='all')
        df = df[~df.index.isna()]
        labels = df.index.tolist()
        cols = df.columns.tolist()
        count_data = df.fillna(0).apply(pd.to_numeric, errors='coerce').values.astype(int)
        proportion_data = convert_to_proportion(count_data)
        print(f"✅ Loaded {sheet_name}: Count Shape={count_data.shape}, Labels={len(labels)}")
        return count_data, proportion_data, labels, cols
    except Exception as e:
        print(f"❌ Failed to load {sheet_name}: {str(e)[:50]}")
        return None, None, None, None

def extract_error_features(count_data, proportion_data, labels, sheet_name):
    """Extract error features (focus on error proportion: count / total errors of sheet)"""
    # 1. Error count matrix (core)
    error_counts = count_data.copy()
    np.fill_diagonal(error_counts, 0)
    
    # 2. Total statistics
    total_error_count = np.sum(error_counts)
    total_sample_count = np.sum(count_data)
    overall_error_rate = total_error_count / total_sample_count if total_sample_count > 0 else 0
    
    # 3. Error pairs (with proportion: count / total_error_count)
    error_pairs = []
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            if i != j and error_counts[i, j] > 0:
                # 核心：计算该错误对占Sheet总错误数的比例
                pair_proportion = error_counts[i, j] / total_error_count if total_error_count > 0 else 0
                error_pairs.append({
                    "sheet": sheet_name,
                    "source": source,
                    "target": target,
                    "count": error_counts[i, j],  # 原始错误数（参考）
                    "proportion": pair_proportion,  # 错误占比（核心）
                    "source_sample_count": count_data[i].sum()
                })
    
    # Sort by proportion (descending)
    error_pairs_sorted = sorted(error_pairs, key=lambda x: x["proportion"], reverse=True)
    top_error_pairs = error_pairs_sorted[:5]
    
    # 4. Class error proportion (class total errors / sheet total errors)
    class_error_counts = error_counts.sum(axis=1)
    class_error_proportion = {}
    for cls_idx, cls_name in enumerate(labels):
        class_error_proportion[cls_name] = class_error_counts[cls_idx] / total_error_count if total_error_count > 0 else 0
    
    return {
        "sheet": sheet_name,
        "error_counts": error_counts,
        "class_error_proportion": class_error_proportion,  # 类别错误占比
        "top_error_pairs": top_error_pairs,
        "total_error_count": total_error_count,
        "total_sample_count": total_sample_count,
        "overall_error_rate": overall_error_rate,
        "labels": labels,
        "all_error_pairs": error_pairs
    }

def plot_combined_error_patterns(all_error_features, save_path):
    """
    Single combined visualization (error proportion) for S9-S17:
    1. Stacked bar: Top 10 error pairs (proportion by sheet)
    2. Grouped bar: Top 10 classes (error proportion by sheet)
    3. Summary: Total error count per sheet (baseline reference)
    """
    # -------------------------- Prepare combined data (error proportion) --------------------------
    # 1. Collect all error pairs (with proportion)
    all_pairs = []
    for sheet_features in all_error_features.values():
        all_pairs.extend(sheet_features["all_error_pairs"])
    
    # 2. Aggregate error pairs by (source, target) → sheet: proportion
    pair_sheet_proportion = {}
    for pair in all_pairs:
        key = (pair["source"], pair["target"])
        if key not in pair_sheet_proportion:
            pair_sheet_proportion[key] = {}
        pair_sheet_proportion[key][pair["sheet"]] = pair["proportion"]
    
    # 3. Top 10 error pairs (by max proportion across sheets)
    pair_max_proportion = {
        key: max(list(sheet_props.values())) for key, sheet_props in pair_sheet_proportion.items()
    }
    top_pairs_total = sorted(pair_max_proportion.items(), key=lambda x: x[1], reverse=True)[:10]
    top_pair_keys = [k for k, _ in top_pairs_total]
    
    # 4. Prepare data for stacked bar (top pairs × sheets: proportion)
    pair_sheet_data = {}
    for pair_key in top_pair_keys:
        pair_sheet_data[pair_key] = {
            sheet: pair_sheet_proportion.get(pair_key, {}).get(sheet, 0) 
            for sheet in all_error_features.keys()
        }
    
    # 5. Collect class error proportion across sheets
    class_sheet_proportion = {}
    for sheet, features in all_error_features.items():
        for cls, prop in features["class_error_proportion"].items():
            if cls not in class_sheet_proportion:
                class_sheet_proportion[cls] = {}
            class_sheet_proportion[cls][sheet] = prop
    
    # 6. Top 10 classes with highest average proportion
    cls_avg_proportion = {
        cls: np.mean(list(props.values())) for cls, props in class_sheet_proportion.items()
    }
    top_classes = [cls for cls, _ in sorted(cls_avg_proportion.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # 7. Total error count per sheet (reference)
    sheet_total_errors = {
        sheet: features["total_error_count"] for sheet, features in all_error_features.items()
    }
    
    # -------------------------- Create single combined plot --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=dpi)
    fig.suptitle(main_title, fontweight="bold", fontsize=14, y=0.98)
    valid_sheets = list(all_error_features.keys())
    
    # -------------------------- 1. Stacked bar: Top 10 error pairs (proportion) --------------------------
    ax1 = axes[0]
    pair_labels = [f"{s} → {t}" for s, t in top_pair_keys]
    bottom = np.zeros(len(top_pair_keys))
    
    for sheet in valid_sheets:
        props = [pair_sheet_data[pk].get(sheet, 0) for pk in top_pair_keys]
        if sum(props) > 0:
            ax1.bar(
                pair_labels, props, bottom=bottom, 
                color=sheet_colors.get(sheet, "#000000"), label=sheet
            )
            bottom += props
    
    ax1.set_title("Top 10 Error Pairs (Proportion of Sheet Total Errors)", fontweight="bold")
    ax1.set_xlabel("Error Pair (True → Predicted)")
    ax1.set_ylabel("Proportion of Total Errors (0-1)")
    ax1.set_xticklabels(pair_labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(title="Sheet", bbox_to_anchor=(1.05, 1), loc="upper left")
    # 添加百分比刻度
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    
    # -------------------------- 2. Grouped bar: Top 10 classes (proportion) --------------------------
    ax2 = axes[1]
    x = np.arange(len(top_classes))
    width = 0.1  # Bar width
    offset = - (len(valid_sheets) * width) / 2
    
    for sheet in valid_sheets:
        props = [class_sheet_proportion[cls].get(sheet, 0) for cls in top_classes]
        ax2.bar(
            x + offset, props, width, 
            color=sheet_colors.get(sheet, "#000000"), label=sheet
        )
        offset += width
    
    ax2.set_title("Top 10 Classes (Proportion of Sheet Total Errors)", fontweight="bold")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Proportion of Total Errors (0-1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_classes, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    # 添加百分比刻度
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    
    # -------------------------- 3. Total error count per sheet (reference) --------------------------
    ax3 = axes[2]
    sheets = list(sheet_total_errors.keys())
    counts = [sheet_total_errors[s] for s in sheets]
    colors = [sheet_colors.get(s, "#000000") for s in sheets]
    
    ax3.bar(sheets, counts, color=colors)
    ax3.set_title("Total Error Count per Sheet (Reference)", fontweight="bold")
    ax3.set_xlabel("Sheet")
    ax3.set_ylabel("Total Error Count")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for idx, count in enumerate(counts):
        ax3.text(idx, count + 5, f"{count}", ha="center", va="bottom", fontsize=8)
    
    # -------------------------- Layout adjustment --------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.3)
    
    # Add color legend (for all sheets)
    legend_patches = [
        mpatches.Patch(color=sheet_colors[sheet], label=f"{sheet}: {table_titles.get(sheet, sheet)}")
        for sheet in valid_sheets
    ]
    fig.legend(
        handles=legend_patches, 
        title="Sheet & Dataset",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=8
    )
    
    # Save single SVG
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Single combined plot saved: {save_path}")

def generate_error_report(all_error_features, save_path):
    """Generate error proportion report for S9-S17"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("ERROR PATTERN ANALYSIS REPORT (S9-S17) - ERROR PROPORTION\n")
        f.write("="*80 + "\n\n")
        f.write("⚠️ Error Proportion = Single Error Pair Count / Total Errors of the Sheet\n\n")
        
        # 1. Per-sheet summary
        f.write("1. PER-SHEET ERROR PROPORTION SUMMARY\n")
        f.write("-"*50 + "\n")
        for sheet, features in all_error_features.items():
            f.write(f"\n📊 Sheet {sheet}: {table_titles.get(sheet, sheet)}\n")
            f.write(f"   Total Samples: {features['total_sample_count']}\n")
            f.write(f"   Total Errors: {features['total_error_count']}\n")
            f.write(f"   Overall Error Rate: {features['overall_error_rate']:.4f} ({features['overall_error_rate']:.2%})\n")
            f.write(f"   Top 5 Error Pairs (by Proportion):\n")
            for idx, pair in enumerate(features["top_error_pairs"], 1):
                f.write(f"     {idx}. {pair['source']} → {pair['target']}: {pair['proportion']:.4f} ({pair['proportion']:.2%}) (count: {pair['count']})\n")
        
        # 2. Combined summary
        f.write("\n\n2. COMBINED ERROR SUMMARY (S9-S17)\n")
        f.write("-"*50 + "\n")
        # Total statistics
        total_global_errors = sum([f["total_error_count"] for f in all_error_features.values()])
        total_global_samples = sum([f["total_sample_count"] for f in all_error_features.values()])
        f.write(f"Total Errors (S9-S17): {total_global_errors}\n")
        f.write(f"Total Samples (S9-S17): {total_global_samples}\n")
        f.write(f"Global Error Rate: {total_global_errors/total_global_samples:.4f} ({total_global_errors/total_global_samples:.2%})\n")
        
        # Top 10 error pairs (max proportion across sheets)
        all_pairs = []
        for features in all_error_features.values():
            all_pairs.extend(features["all_error_pairs"])
        pair_max_proportion = {}
        for pair in all_pairs:
            key = (pair["source"], pair["target"])
            if key not in pair_max_proportion or pair["proportion"] > pair_max_proportion[key]:
                pair_max_proportion[key] = pair["proportion"]
        top_10_pairs = sorted(pair_max_proportion.items(), key=lambda x: x[1], reverse=True)[:10]
        
        f.write("\nTop 10 Error Pairs (Max Proportion Across Sheets):\n")
        for idx, ((source, target), prop) in enumerate(top_10_pairs, 1):
            f.write(f"   {idx}. {source} → {target}: {prop:.4f} ({prop:.2%})\n")
        
        f.write("\n" + "="*80 + "\nEND OF REPORT\n" + "="*80 + "\n")
    print(f"✅ Error proportion report saved: {save_path}")

# -------------------------- 3. Main Execution --------------------------
if __name__ == "__main__":
    # Step 1: Load confusion matrices (S9-S17)
    excel_path = "Supplementary_Tables20251205.xlsx"
    tables = [f"S{i}" for i in range(9, 18)]  # S9-S17 only
    all_cm_data = {}
    all_error_features = {}
    
    for sheet in tables:
        count_data, proportion_data, labels, cols = load_confusion_matrix(excel_path, sheet)
        if count_data is not None:
            all_cm_data[sheet] = (count_data, proportion_data, labels, cols)
            # Step 2: Extract error features (proportion focus)
            error_features = extract_error_features(count_data, proportion_data, labels, sheet)
            all_error_features[sheet] = error_features
    
    # Step 3: Generate single combined plot (proportion)
    if all_error_features:
        plot_combined_error_patterns(all_error_features, "Combined_Error_Proportion_Analysis_S9-S17.svg")
        # Generate report
        generate_error_report(all_error_features, "Error_Proportion_Analysis_Report_S9-S17.txt")
        
        # Step 4: Print summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY (S9-S17 - ERROR PROPORTION)")
        print("="*80)
        print(f"✅ Processed {len(all_error_features)} sheets (S9-S17)")
        print("✅ Error Proportion = Single Error Pair Count / Sheet Total Errors")
        print("✅ Single combined plot saved: Combined_Error_Proportion_Analysis_S9-S17.svg")
        print("✅ Report saved: Error_Proportion_Analysis_Report_S9-S17.txt")
        print(f"✅ Total errors across S9-S17: {sum([f['total_error_count'] for f in all_error_features.values()])}")
    else:
        print("❌ No valid data loaded (S9-S17) - analysis aborted!")