import matplotlib.pyplot as plt
import numpy as np

# ========== 全局参数 ==========
gap_scale = 0.9
font_scale = 1.0

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8 * font_scale
plt.rcParams['axes.labelsize'] = 10 * font_scale
plt.rcParams['axes.titlesize'] = 12 * font_scale
plt.rcParams['legend.fontsize'] = 9 * font_scale
plt.rcParams['xtick.labelsize'] = 7 * font_scale
plt.rcParams['ytick.labelsize'] = 8 * font_scale

# 颜色
color_deephyd = '#2E86AB'
color_hyde = '#A23B72'
color_grid = '#E0E0E0'

# 文字标签
title_text = "Performance comparison: DeepHyb vs HyDe"
ylabel_score = "Score"
xlabel_types = "Four-taxon combinations"
legend_deephyd = "DeepHyb"
legend_hyde = "HyDe"

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
types = [
    "O_S1_S1_S1", "O_S1_S1_S2", "O_S1_S1_H", "O_S1_S1_S3", "O_S1_S2_S2",
    "O_S1_S2_H", "O_S1_H_H", "O_S1_S2_S3", "O_S1_H_S3", "O_S1_S3_S3",
    "O_S2_S2_S2", "O_S2_S2_H", "O_S2_H_H", "O_H_H_H", "O_S2_S2_S3",
    "O_S2_H_S3", "O_H_H_S3", "O_S2_S3_S3", "O_H_S3_S3", "O_S3_S3_S3"
]

# ========== 三组数据 ==========
data = [
    # Set 1
    {
        'deephyd_metrics': [0.8773, 0.5, 1.0, 0.6667],
        'hyde_metrics': [0.8501, 0.4501, 1.0, 0.6202],
        'deephyd_types': [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ],
        'hyde_types': [
            0.9920, 0.9649, 0.9667, 0.9667, 0.9660, 0.9633, 0.9638, 0.9626,
            1.0, 0.9573, 0.9840, 0.9620, 0.9622, 0.9980, 0.9611, 0.0,
            0.9622, 0.9602, 0.9684, 0.9880
        ]
    },
    # Set 2
    {
        'deephyd_metrics': [0.8774, 0.5002, 0.9958, 0.6659],
        'hyde_metrics': [0.8498, 0.4496, 1.0, 0.6203],
        'deephyd_types': [
            1.0, 1.0, 1.0, 1.0, 1.0, 0.998889, 1.0, 0.999852,
            0.995778, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.006444,
            1.0, 0.999778, 1.0, 1.0
        ],
        'hyde_types': [
            0.996, 0.962888889, 0.964, 0.967555556, 0.962888889,
            0.95637037, 0.967111111, 0.964296296, 1.0, 0.958888889,
            0.992, 0.969555556, 0.963111111, 0.99, 0.959555556,
            0.0, 0.960666667, 0.959333333, 0.973111111, 0.994
        ]
    },
    # Set 3
    {
        'deephyd_metrics': [0.8773, 0.5000, 1.0, 0.6667],
        'hyde_metrics': [0.8512, 0.4519, 1.0, 0.6225],
        'deephyd_types': [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ],
        'hyde_types': [
            0.984, 0.964, 0.964222222, 0.968444444, 0.968666667,
            0.956518519, 0.966222222, 0.964074074, 1.0, 0.976444444,
            0.99, 0.963555556, 0.968888889, 0.994, 0.965333333,
            0.0, 0.964888889, 0.967111111, 0.966666667, 0.99
        ]
    }
]

# ========== 绘图函数 ==========
def plot_metrics(ax, d_metrics, h_metrics, show_ylabel=True):
    x = np.arange(len(metrics))
    width = 0.35 * gap_scale

    bars1 = ax.bar(x - width/2, d_metrics, width,
                   label=legend_deephyd, color=color_deephyd,
                   alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, h_metrics, width,
                   label=legend_hyde, color=color_hyde,
                   alpha=0.8, edgecolor='white', linewidth=1)

    # 纵轴上限 1.05 留 buffer，刻度仅到 1.0
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis='y', alpha=0.3, color=color_grid)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel:
        ax.set_ylabel(ylabel_score, fontweight='bold')

    # 显示数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=7 * font_scale)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=7 * font_scale)

    ax.legend(loc='upper right', frameon=True, fancybox=True,
              shadow=True, framealpha=0.8, fontsize=8 * font_scale)


def plot_types(ax, d_types, h_types, show_ylabel=True, show_xlabel=True):
    x = np.arange(len(types))

    ax.plot(x, d_types, marker='o', linewidth=2.5 * gap_scale,
            markersize=4 * gap_scale, color=color_deephyd,
            label=legend_deephyd, alpha=0.9)
    ax.plot(x, h_types, marker='s', linewidth=2.5 * gap_scale,
            markersize=4 * gap_scale, color=color_hyde,
            label=legend_hyde, alpha=0.9)

    # 纵轴上限 1.05 留 buffer，刻度设置只到 1.0
    ax.set_ylim(-0.05, 1.05)
    # 基础刻度 0, 0.2, 0.4, 0.6, 0.8, 1.0，并强制包含 0.95
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if 0.95 not in yticks:
        yticks.append(0.95)
        yticks.sort()
    ax.set_yticks(yticks)

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha='right', rotation_mode='anchor')
    ax.grid(axis='y', alpha=0.3, color=color_grid)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel:
        ax.set_ylabel(ylabel_score, fontweight='bold')
    if show_xlabel:
        ax.set_xlabel(xlabel_types, fontweight='bold', labelpad=15 * gap_scale)

    # 图例位置手动控制（可根据需要调整 bbox_to_anchor 的数值）
    ax.legend(bbox_to_anchor=(1, 0.89), loc='upper right',
              frameon=True, fancybox=True, shadow=True,
              framealpha=0.8, fontsize=8 * font_scale)


# ========== 显式定义组标签 ==========
set_labels = ["Sequence length = 100,000; mutation rate = 0.01", "Sequence length = 50,000; mutation rate = 0.01", "Sequence length = 100,000; mutation rate = 0.05"]   # 可单独修改

# ========== 创建 2 行 3 列子图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18 * gap_scale, 11 * gap_scale),
                         gridspec_kw={'height_ratios': [1, 2.5]})

# 循环绘制三列
for col, d in enumerate(data):
    ax_m = axes[0, col]
    ax_t = axes[1, col]

    show_ylabel = (col == 0)

    plot_metrics(ax_m, d['deephyd_metrics'], d['hyde_metrics'],
                 show_ylabel=show_ylabel)
    plot_types(ax_t, d['deephyd_types'], d['hyde_types'],
               show_ylabel=show_ylabel, show_xlabel=True)

    # 组标签
    ax_m.text(0.02, 1.08, set_labels[col], transform=ax_m.transAxes,
              fontweight='bold', fontsize=10 * font_scale, va='top')

    # 在折线图上添加 y=0.95 虚线
    ax_t.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# 总标题
fig.suptitle(title_text, fontweight='bold', fontsize=14 * font_scale, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Figure_3.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()