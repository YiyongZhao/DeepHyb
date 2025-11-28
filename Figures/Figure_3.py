import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Construct data
# Data includes 5 evaluation metrics across 9 models
data = {
    "legend": [
        "HYB Accuracy", "HYB Precision", 
        "HYB Recall", "HYB F1 Score", 
        "Combinations Accuracy"
    ],
    "Model 1": [0.935976336, 0.936175407, 0.935976336, 0.935818906, 0.99027494],
    "Model 2": [0.9465, 0.9467, 0.9465, 0.9464, 0.989283282],
    "Model 3": [0.9511, 0.9512, 0.9511, 0.9511, 0.989968505],
    "Model 4": [0.9372, 0.9373, 0.9372, 0.9370, 0.9902111],
    "Model 9": [0.952608955, 0.952680466, 0.952608955, 0.952537274, 0.994292646],
    "Model 10": [0.943117978, 0.94333767, 0.943117978, 0.942982156, 0.990168539],
    "Model 11": [0.956005277, 0.956033893, 0.956005277, 0.955954393, 0.995003405],
    "Model 12": [0.950629894, 0.950827486, 0.950629894, 0.950523708, 0.994096868],
    "Model 16": [0.8648, 0.8656, 0.8648, 0.8650, 0.9539]
}

# 2. Convert to DataFrame and reshape from wide to long format
# Long format is required for seaborn's grouped barplot
df = pd.DataFrame(data)
df_long = df.melt(id_vars="legend", var_name="Model ", value_name="Accuracy")

# 3. Plot grouped bar chart
plt.figure(figsize=(10, 6))  # Set figure size (width, height) in inches
sns.barplot(
    data=df_long, 
    x="Model ",       # X-axis: Model identifiers
    y="Accuracy",     # Y-axis: Metric values
    hue="legend",     # Grouping: Different evaluation metrics
    palette="Set2"    # Color palette for distinction between metrics
)

# 4. Enhance plot aesthetics
plt.xlabel("Model ", fontsize=12)  # X-axis label
plt.ylabel("Accuracy", fontsize=12)  # Y-axis label
plt.ylim(0.85, 1.0)  # Restrict Y-axis range to highlight differences

# 5. Customize legend
# Remove frame and title, position at upper right
plt.legend(frameon=False, title=None, loc='upper right', bbox_to_anchor=(0.95, 0.95))

# 6. Adjust layout to prevent clipping
plt.tight_layout(rect=[0, 0, 1, 1])  # Ensure all elements fit within figure bounds

# 7. Save as SVG vector format (editable in Adobe Illustrator)
# SVG preserves vector paths for无损 editing (scaling without quality loss)
plt.savefig("Figure_2.svg", format="svg", bbox_inches="tight")

# Optional: Display the plot in the console
plt.show()