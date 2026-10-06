import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#6b6b6b", "#e6e6e3"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#a8a8a5"

# (label, public, private, highlight, note)
ROWS = [
    ("head v1",                    0.952, 0.919, None,   ""),
    ("flow2 + gapfill + head v1",  0.952, 0.920, "sel",  "the submission I selected"),
    ("head v2",                    0.950, 0.921, "best", "best private score — not selected"),
    ("another team's public head", 0.950, 0.917, None,   ""),
    ("head v1, half displacement", 0.949, 0.919, None,   ""),
    ("base stack, no head",        0.947, 0.914, None,   ""),
]

fig, ax = plt.subplots(figsize=(11, 4.9), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

ys = list(range(len(ROWS)))[::-1]
for y, (label, pub, pri, hl, note) in zip(ys, ROWS):
    col = {"sel": BLUE, "best": ORANGE}.get(hl, NEUTRAL)
    ax.plot([pri, pub], [y, y], color=col, lw=2.2 if hl else 1.5,
            alpha=0.95 if hl else 0.5, solid_capstyle="round", zorder=2)
    ax.scatter([pub], [y], s=78, facecolor=SURFACE, edgecolor=col, lw=2.2, zorder=3)
    ax.scatter([pri], [y], s=86, facecolor=col, edgecolor=SURFACE, lw=1.6, zorder=3)
    ax.text(pri - 0.00125, y, f"{pri:.3f}", ha="right", va="center",
            fontsize=10, color=INK if hl else MUTED,
            fontweight="bold" if hl else "normal")
    ax.text(pub + 0.00125, y, f"{pub:.3f}", ha="left", va="center",
            fontsize=10, color=MUTED)
    if note:
        ax.text((pri + pub) / 2, y + 0.3, note, ha="center", va="bottom",
                fontsize=9.5, color=col, fontweight="bold")

ax.set_xlim(0.9095, 0.9575)
ax.set_ylim(-0.75, len(ROWS) - 0.35)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS],
                   fontsize=10.5)
for tick, (_, _, _, hl, _) in zip(ax.get_yticklabels(), ROWS):
    tick.set_color(INK if hl else MUTED)
    tick.set_fontweight("bold" if hl else "normal")

ax.set_xticks([0.915, 0.925, 0.935, 0.945, 0.955])
ax.set_xlabel("Competition score  (mean Jaccard)", fontsize=10, color=MUTED, labelpad=9)
ax.tick_params(axis="x", colors=MUTED, labelsize=9.5, length=0)
ax.tick_params(axis="y", length=0, pad=10)
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)

ax.set_title("The public leaderboard ranked these submissions in the wrong order",
             fontsize=14, color=INK, fontweight="bold", loc="left", pad=34)
ax.text(0, 1.055, "Rows ordered by public score, best at the top. On the private split that order breaks.",
        transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")

ax.legend(handles=[
    Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED,
           markeredgewidth=2.2, markersize=8.5, label="Public  (29% of held-out videos)"),
    Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=SURFACE,
           markersize=9.5, label="Private  (71%)"),
], loc="upper left", bbox_to_anchor=(0, -0.145), frameon=False, fontsize=10,
   labelcolor=MUTED, handletextpad=0.6, ncol=2, columnspacing=2.4)

fig.savefig("biohub-pub-vs-priv.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
print("ok")
