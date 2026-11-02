"""Figure 1: DRiFT workflow. Boxes are placed on a fixed grid with clear gaps; every arrow is drawn
between explicit edge anchor points so directions and connections are exact."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

W, H = 17.0, 9.0
BW, BH = 2.6, 1.4           # box size
COL = {"in": "#dbe9f8", "phys": "#e3f4ec", "noise": "#fde9dc", "loss": "#ece9fb", "out": "#fff2cc"}
EDGE = {"in": "#2a78d6", "phys": "#1baf7a", "noise": "#eb6834", "loss": "#6250d6", "out": "#c98500"}
R1, R2, R3 = 7.25, 4.6, 1.75

boxes = {
    # name: (x_center, y_center, label, style)
    "coef":  (1.8, R1, "Pulse coefficients\n$\\mathbf{c}\\in\\mathbb{R}^{n_u\\times K}$", "in"),
    "ctrl":  (5.2, R1, "Band-limited controls\n$u_c(t)=\\Sigma_l\\,c_{cl}\\sin(l\\pi t/T)$", "in"),
    "prop":  (8.6, R1, "Exact ZOH\npropagation\n$V_k,\\;U_k,\\;U_N$", "phys"),
    "tog":   (12.0, R1, "Toggling frame\n(Gauss-Legendre)\n$X_{kq}=W^{\\dagger}E_kW$", "phys"),
    "ff":    (15.3, R1, "Leakage-resolved\nfilter functions\n$\\bar{\\Phi}_j(\\omega)$", "phys"),
    "adam":  (1.8, R2, "Adam update\n$\\mathbf{c}\\leftarrow\\mathbf{c}-\\eta\\,\\hat{m}/\\sqrt{\\hat{v}}$", "loss"),
    "inom":  (8.6, R2, "Nominal\ninfidelity\n$I_{\\mathrm{nom}}(U_N)$", "phys"),
    "loss":  (12.0, R2, "Objective $J$\n$I_{\\mathrm{nom}}+\\frac{d}{d+1}\\Sigma_j W_j+\\rho$", "loss"),
    "ks":    (15.3, R2, "Spectral knapsack\nworst case\n$W_j(\\mathbf{c})$", "noise"),
    "cert":  (1.8, R3, "Robust pulse +\ncertificate $F_{\\mathrm{cert}}$", "out"),
    "drift": (5.2, R3, "Drift monitor:\nrecalibrate only if\n$\\hat{S}_j\\notin\\mathcal{S}_j$", "out"),
    "spec":  (12.0, R3, "Noise spectroscopy\n$\\hat{S}_j(\\omega)$, Ramsey $\\sigma^2_j$", "noise"),
    "aset":  (15.3, R3, "Ambiguity set $\\mathcal{S}_j$\n(box + variance\nbudget)", "noise"),
}


def anchor(name, side):
    x, y = boxes[name][0], boxes[name][1]
    return {"l": (x - BW / 2, y), "r": (x + BW / 2, y), "t": (x, y + BH / 2), "b": (x, y - BH / 2)}[side]


fig, ax = plt.subplots(figsize=(W * 0.62, H * 0.62))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

for nm, (x, y, txt, st) in boxes.items():
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=COL[st], ec=EDGE[st], lw=1.4, zorder=2))
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.6, zorder=3, linespacing=1.35)


def arrow(a, sa, b, sb, style="-", color="#333333", text=None, toff=(0, 0.0), rad=0.0, lw=1.4,
          via=None, tpos=None):
    """Straight or curved arrow between edge anchors; `via` = list of corner points for an
    orthogonal polyline whose last segment carries the arrow head."""
    p, q = anchor(a, sa), anchor(b, sb)
    pts = [p] + (via or []) + [q]
    if via:
        # draw every segment in the line style, then a short solid stub that carries a clean head
        for u, v in zip(pts[:-1], pts[1:]):
            ax.plot([u[0], v[0]], [u[1], v[1]], color=color, lw=lw, ls=style, zorder=1)
        u, v = pts[-2], pts[-1]
        dx, dy = v[0] - u[0], v[1] - u[1]
        n = (dx ** 2 + dy ** 2) ** 0.5
        tail = (v[0] - 0.35 * dx / n, v[1] - 0.35 * dy / n)
        ax.add_patch(FancyArrowPatch(tail, v, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                                     linestyle="-", shrinkA=0, shrinkB=2, zorder=1))
    else:
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                                     linestyle=style, shrinkA=2, shrinkB=2,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))
    if text:
        if tpos is None:
            tpos = ((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1])
        ax.text(tpos[0], tpos[1], text, ha="center", va="center", fontsize=7.6, color=color,
                bbox=dict(fc="white", ec="none", pad=0.8), zorder=4)


# forward pass (solid); arrow labels sit beside vertical arrows or above the gap of horizontal ones
arrow("coef", "r", "ctrl", "l")
arrow("ctrl", "r", "prop", "l")
arrow("prop", "r", "tog", "l")
arrow("tog", "r", "ff", "l")
arrow("prop", "b", "inom", "t", text="$U_N$", toff=(0.38, 0))
arrow("ff", "b", "ks", "t", text="$\\bar{\\Phi}_j$", toff=(0.38, 0))
arrow("inom", "r", "loss", "l")
arrow("ks", "l", "loss", "r")
arrow("spec", "r", "aset", "l")
arrow("aset", "t", "ks", "b", text="$\\ell_j,\\,h_j,\\,B_j$", toff=(0.62, 0))
# reverse-mode gradient (dashed): objective -> optimiser -> coefficients, routed below the middle row
yg = 3.25
arrow("loss", "b", "adam", "r", style="--", color="#6250d6",
      via=[(12.0, yg), (4.25, yg), (4.25, R2)],
      text="reverse-mode $\\nabla_{\\mathbf{c}}J$  (Danskin: worst-case $S^{\\star}_j$ held fixed)",
      tpos=(8.1, yg))
arrow("adam", "t", "coef", "b", style="--", color="#6250d6")
# outputs
arrow("adam", "b", "cert", "t", text="converged $\\mathbf{c}^{\\star}$", toff=(0.75, 0))
arrow("cert", "r", "drift", "l")
arrow("aset", "b", "drift", "b", style=":", color="#c98500",
      via=[(15.3, 0.45), (5.2, 0.45)], text="membership test uses $\\mathcal{S}_j$", tpos=(10.2, 0.45))

# legend for line styles (placed in free space, no overlap)
lx, ly = 0.35, 8.65
ax.plot([lx, lx + 0.6], [ly, ly], color="#333333", lw=1.4)
ax.text(lx + 0.75, ly, "forward computation", va="center", fontsize=7.8)
ax.plot([lx + 3.3, lx + 3.9], [ly, ly], color="#6250d6", lw=1.4, ls="--")
ax.text(lx + 4.05, ly, "reverse-mode gradient", va="center", fontsize=7.8)
ax.plot([lx + 6.8, lx + 7.4], [ly, ly], color="#c98500", lw=1.4, ls=":")
ax.text(lx + 7.55, ly, "deployment / certification", va="center", fontsize=7.8)

fig.tight_layout(pad=0.2)
fig.savefig("fig1_architecture.png", dpi=600)
fig.savefig("fig1_architecture.pdf")
print("saved")
