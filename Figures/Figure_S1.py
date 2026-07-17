import matplotlib.pyplot as plt

# 使 SVG 文字可编辑
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(8, 6))

# 坐标轴范围
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.4, 1.8)
ax.set_xlabel("Subpopulations", fontsize=12)
ax.set_ylabel("Time (4N$_0$ generations)", fontsize=12)

# 时间参考线
for y in [0.5, 1.0, 1.5]:
    ax.axhline(y=y, color='gray', linestyle=':', alpha=0.4, zorder=0)

ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.set_yticklabels(["present", "0.5", "1.0", "1.5"])
ax.set_xticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 终端 x 坐标：O, S2, S1, H, S3
xpos = {
    'O': 0.0,
    'S2': 1.0,
    'S1': 2.0,
    'H': 3.0,
    'S3': 4.0,
}

# 内部节点坐标
root = (3.0, 1.5)
node10 = (3.0, 1.0)
node05 = (3.0, 0.5)

# ---- 画主干树 ----
# 外群 O 分支（t=1.5）
ax.plot([root[0], xpos['O']], [root[1], root[1]], 'k-', lw=2, zorder=1)
ax.plot([xpos['O'], xpos['O']], [root[1], 0], 'k-', lw=2, zorder=1)

# 根到 t=1.0 节点
ax.plot([root[0], node10[0]], [root[1], node10[1]], 'k-', lw=2, zorder=1)

# S2 分支（t=1.0）
ax.plot([node10[0], xpos['S2']], [node10[1], node10[1]], 'k-', lw=2, zorder=1)
ax.plot([xpos['S2'], xpos['S2']], [node10[1], 0], 'k-', lw=2, zorder=1)

# t=1.0 到 t=0.5 节点
ax.plot([node10[0], node05[0]], [node10[1], node05[1]], 'k-', lw=2, zorder=1)

# S1 和 S3 分支（t=0.5）
ax.plot([node05[0], xpos['S1']], [node05[1], node05[1]], 'k-', lw=2, zorder=1)
ax.plot([xpos['S1'], xpos['S1']], [node05[1], 0], 'k-', lw=2, zorder=1)

ax.plot([node05[0], xpos['S3']], [node05[1], node05[1]], 'k-', lw=2, zorder=1)
ax.plot([xpos['S3'], xpos['S3']], [node05[1], 0], 'k-', lw=2, zorder=1)

# ---- 杂交基因流（两条虚线表示两种拓扑） ----
# H 与 S1 更近（红色虚线）
ax.plot([xpos['S1'], xpos['H']], [0.15, 0], 'r--', lw=2, zorder=2, label='H closer to S1')
# H 与 S3 更近（蓝色虚线）
ax.plot([xpos['S3'], xpos['H']], [0.15, 0], 'b--', lw=2, zorder=2, label='H closer to S3')

# ---- 终端标签 ----
for name, x in xpos.items():
    ax.text(x, -0.05, name, ha='center', va='top', fontsize=12, fontweight='bold')

ax.text(xpos['H'], -0.15, "(hybrid)", ha='center', va='top', fontsize=9, color='red')

# ---- 标记时间点 ----
ax.text(root[0] + 0.08, root[1] + 0.02, "1.5", fontsize=10, ha='left', va='bottom')
ax.text(node10[0] + 0.08, node10[1] + 0.02, "1.0", fontsize=10, ha='left', va='bottom')
ax.text(node05[0] + 0.08, node05[1] + 0.02, "0.5", fontsize=10, ha='left', va='bottom')

# ---- 标注 S2 不相关 ----
ax.text(xpos['S2'] + 0.05, 0.2, "unrelated", fontsize=9, color='gray', rotation=90, ha='left', va='center')

# ---- 图例 ----
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()

# 保存为 SVG 矢量图
plt.savefig("Figure_S1.svg", format="svg", bbox_inches="tight")

# 如需显示，取消下一行注释
# plt.show()