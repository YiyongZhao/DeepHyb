import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

data = {
    "legend": ["Hyb accuracy", "Hyb precision", "Hyb recall", "Hyb F1 Score", "Combinations accuracy"],
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

df = pd.DataFrame(data)
df_long = df.melt(id_vars="legend", var_name="Model ", value_name="Performance metric value")

fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
fig.set_tight_layout(False)  

sns.barplot(
    data=df_long, 
    x="Model ",       
    y="Performance metric value",     
    hue="legend",     
    palette="Set2",
    ax=ax
)

ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False) 
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

ax.set_ylim(0.85, 1.01)  
ax.set_xlim(-0.5, 8.5)  
ax.set_xlabel("", fontsize=12)  
ax.set_ylabel("Performance metric value", fontsize=12, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)  

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.tick_params(axis='y', pad=5)  

ax.legend(
    frameon=False, 
    title=None, 
    loc='upper right',
    bbox_to_anchor=(1.05, 0.99),  
    fontsize=10,
    handlelength=1.5
)

def draw_brace_with_label(ax, start_model_idx, end_model_idx, y_brace, label, y_label, label_rotation=0):

    start_x = start_model_idx
    end_x = end_model_idx
    mid_x = (start_x + end_x) / 2
    
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
    left_ctrl_x = start_x + (mid_x - start_x) * 0.5
    left_ctrl_y = y_brace - 0.008
    
    left_vertices = [
        (start_x, y_brace),
        (left_ctrl_x, left_ctrl_y),
        (mid_x, y_brace - 0.01),
    ]
    left_codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    left_path = Path(left_vertices, left_codes)
    
    right_ctrl_x = end_x - (end_x - mid_x) * 0.5
    right_ctrl_y = y_brace - 0.008
    
    right_vertices = [
        (mid_x, y_brace - 0.01),
        (right_ctrl_x, right_ctrl_y),
        (end_x, y_brace),
    ]
    right_codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    right_path = Path(right_vertices, right_codes)
    
    left_patch = patches.PathPatch(
        left_path, 
        facecolor='none', 
        edgecolor='black', 
        linewidth=1.2,
        capstyle='round'
    )
    right_patch = patches.PathPatch(
        right_path, 
        facecolor='none', 
        edgecolor='black', 
        linewidth=1.2,
        capstyle='round'
    )
    ax.add_patch(left_patch)
    ax.add_patch(right_patch)
    
    ax.text(
        mid_x, y_label, label,
        ha='center', va='top',
        fontsize=9,
        fontweight='bold',
        rotation=label_rotation
    )


brace_y = 0.835 
label_y = 0.820 



draw_brace_with_label(
    ax, 
    start_model_idx=0, 
    end_model_idx=3, 
    y_brace=brace_y, 
    label="Combined 4 features",  
    y_label=label_y
)

draw_brace_with_label(
    ax, 
    start_model_idx=4, 
    end_model_idx=7, 
    y_brace=brace_y, 
    label="256-one-base-site-patterns",  
    y_label=label_y
)

model16_x = 8  
ax.text(
    model16_x, label_y, "75-summary-kmer-site-patterns",  
    ha='center', va='top',
    fontsize=9,
    fontweight='bold'
)

plt.subplots_adjust(
    bottom=0.18,       
    right=1.10,         
    left=0.05,          
    top=0.98,           
    wspace=0,
    hspace=0
)

fig.savefig(
    "Figure_5.svg",
    format="svg",
    dpi=300,
    pad_inches=0.05,    
    transparent=False,
    facecolor='white',
    bbox_inches=fig.get_tightbbox(fig.canvas.get_renderer())  
)

