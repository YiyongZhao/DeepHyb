import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Construct data
# Data contains SHAP values for different patterns across 9 models
data = {
    "legend": [
        "Model 1", "Model 2", 
        "Model 3", "Model 4", 
        "Model 9", "Model 10", 
        "Model 11", "Model 12",
        "Model 16"
    ],
    "AAAAh": [0.002340879, 0.00217013, 0.001638538, 0.002175862, 2.11265E-05, 0.007470807, 0, 0, 0.029174953],
    "AAABh": [0.035061879, 0.020059667, 0.015021892, 0.0151354, 0.062603381, 0.028732529, 0.055448402, 0.092634199, 0.120237739],
    "AABAh": [0.116629957, 0.129433478, 0.055557213, 0.073703261, 0.202436259, 0.117190951, 0.132333886, 0.116995356, 0.105070695],
    "AABBh": [0.11056048, 0.081752051, 0.103592523, 0.093649178, 0.1133194, 0.144652666, 0.13138957, 0.155647204, 0.158833579],
    "AABCh": [0, 0, 0, 0, 0, 0, 0, 0, 0.028259364],
    "ABAAh": [0.208072631, 0.241636385, 0.191655567, 0.26087255, 0.264216757, 0.126340556, 0.20235122, 0.113763881, 0.073502772],
    "ABABh": [0.202317518, 0.243389134, 0.293958728, 0.233744092, 0.177876903, 0.239537558, 0.215820021, 0.19758572, 0.09728294],
    "ABACh": [0, 0, 0, 0, 0, 0, 0, 0, 0.011213171],
    "ABBAh": [0.323883983, 0.280585134, 0.337839209, 0.31990653, 0.179526165, 0.334629536, 0.262653362, 0.323373555, 0.104375942],
    "BAAAh": [0.001132674, 0.000974021, 0.000736329, 0.000813126, 7.76765E-09, 0.001445397, 3.53813E-06, 8.68675E-08, 0.015400673],
    "ABBCh": [0, 0, 0, 0, 0, 0, 0, 0, 0.113942649],
    "CABCh": [0, 0, 0, 0, 0, 0, 0, 0, 0.005634276],
    "BACAh": [0, 0, 0, 0, 0, 0, 0, 0, 0.043306332],
    "BCAAh": [0, 0, 0, 0, 0, 0, 0, 0, 0.07124233],
    "ABCDh": [0, 0, 0, 0, 0, 0, 0, 0, 0.022522585]
}

# 2. Convert to DataFrame and reshape from wide to long format
# Long format is required for seaborn's grouped barplot visualization
df = pd.DataFrame(data)
df_long = df.melt(id_vars="legend", var_name="Pattern ", value_name="SHAP Value")

# 3. Plot grouped bar chart
plt.figure(figsize=(12, 7))  # Adjust figure size for better readability of x-axis labels
sns.barplot(
    data=df_long, 
    x="Pattern ",       # X-axis: Different patterns (e.g., AAAAh, AAABh)
    y="SHAP Value",     # Y-axis: SHAP value magnitude
    hue="legend",       # Grouping: Different models (Model 1 to Model 16)
    palette="husl"      # Color palette for distinct model identification
)

# 4. Enhance plot aesthetics
plt.xlabel("Pattern ", fontsize=12)  # X-axis label
plt.ylabel("SHAP Value", fontsize=12)  # Y-axis label (removed % as values are decimals)
plt.ylim(0, 0.35)  # Restrict Y-axis range to emphasize value differences

# 5. Customize legend
# Remove frame and title, position at upper right corner
plt.legend(frameon=False, title=None, loc='upper right', bbox_to_anchor=(0.99, 0.95))

# 6. Adjust layout to prevent label clipping
plt.xticks(rotation=45, ha="right")  # Rotate x-labels for better readability
plt.tight_layout(rect=[0, 0, 1, 1])  # Ensure all elements fit within the figure

# 7. Save as SVG vector format (editable in Adobe Illustrator)
# SVG preserves vector paths for lossless editing and scaling
plt.savefig("Figure_3.svg", format="svg", bbox_inches="tight")

# Optional: Display the plot in the output
plt.show()