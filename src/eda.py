"""
eda.py
------------------------------------------------------------------
Exploración de datos + figuras para el reporte.
Lee data/processed/team_match_dataset.csv y guarda PNG en reports/figures/.
------------------------------------------------------------------
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df = pd.read_csv("data/processed/team_match_dataset.csv")
os.makedirs("reports/figures", exist_ok=True)

C = {"W": "#2BB48A", "D": "#9AA3BD", "L": "#E2576B", "warn": "#D99A2B",
     "accent": "#5B79F0", "grid": "#E6E8EF", "ink": "#2A3147"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": C["ink"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"],
    "axes.grid": True, "grid.color": C["grid"], "axes.axisbelow": True, "figure.dpi": 150,
})
pct = lambda ax: ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

n = len(df)
print(f"Filas equipo-partido: {n} | partidos: {df.match_id.nunique()} | torneos: {df.torneo.nunique()}")
print("Por torneo:\n", df.groupby("torneo").match_id.nunique())

# ---------- Fig 1: resultado por estado al descanso ----------
order = ["Lead", "Level", "Trail"]
lbl = {"Lead": "Iba ganando", "Level": "Empatando", "Trail": "Perdiendo"}
ct = (df.groupby("estado_ht")["resultado_ft"].value_counts(normalize=True)
        .unstack().reindex(order)[["W", "D", "L"]])
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(order)); w = 0.26
for i, (r, nm, col) in enumerate([("W", "Ganó", C["W"]), ("D", "Empató", C["D"]), ("L", "Perdió", C["L"])]):
    ax.bar([p + (i - 1) * w for p in x], ct[r].values, w, label=nm, color=col)
ax.set_xticks(list(x)); ax.set_xticklabels([lbl[o] for o in order])
ax.set_title("Resultado final (90') según el marcador al descanso", fontweight="bold", loc="left")
ax.set_ylabel("Proporción"); pct(ax); ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.10))
fig.tight_layout(); fig.savefig("reports/figures/fig1_estado_ht.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: boxplot xg_diff por resultado ----------
fig, ax = plt.subplots(figsize=(7, 4))
data = [df[df.resultado_ft == r]["xg_diff"] for r in ["W", "D", "L"]]
bp = ax.boxplot(data, patch_artist=True, tick_labels=["Ganó", "Empató", "Perdió"],
                widths=.55, medianprops=dict(color=C["ink"], linewidth=1.6))
for patch, col in zip(bp["boxes"], [C["W"], C["D"], C["L"]]):
    patch.set_facecolor(col); patch.set_alpha(.55); patch.set_edgecolor(C["ink"])
ax.axhline(0, color=C["ink"], lw=.8, ls="--", alpha=.6)
ax.set_title("Diferencial de xG del primer tiempo, por resultado final", fontweight="bold", loc="left")
ax.set_ylabel("xG equipo − xG rival (1er tiempo)")
fig.tight_layout(); fig.savefig("reports/figures/fig2_xgdiff_box.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: dominio engañoso (headline) ----------
lead = df[df.estado_ht == "Lead"]
dom_nolead = df[(df.xg_diff > 0.3) & (df.estado_ht != "Lead")]
groups = [("Iba ganando\nal descanso", lead),
          ("Dominó el xG\nsin ir ganando", dom_nolead),
          ("Todos los\nequipos", df)]
winrate = [(g["resultado_ft"] == "W").mean() for _, g in groups]
ns = [len(g) for _, g in groups]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([g[0] for g in groups], winrate, color=[C["W"], C["warn"], C["accent"]], width=.6)
for b, wr, nn in zip(bars, winrate, ns):
    ax.text(b.get_x() + b.get_width() / 2, wr + .02, f"{wr*100:.0f}%\n(n={nn})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("¿Dominar el juego = ganar? El 'dominio engañoso'", fontweight="bold", loc="left")
ax.set_ylabel("% que ganó el partido"); pct(ax); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig("reports/figures/fig3_dominio_enganoso.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: grupos vs eliminación ----------
fig, ax = plt.subplots(figsize=(7, 4))
fases = ["Grupos", "Eliminación"]
wr_sub = lambda s: (s["resultado_ft"] == "W").mean() if len(s) else 0
lead_g = [wr_sub(lead[lead.fase == f]) for f in fases]
dom_g = [wr_sub(dom_nolead[dom_nolead.fase == f]) for f in fases]
x = range(len(fases)); w = .34
ax.bar([p - w / 2 for p in x], lead_g, w, label="Iba ganando", color=C["W"])
ax.bar([p + w / 2 for p in x], dom_g, w, label="Dominó sin ir ganando", color=C["warn"])
for i, f in enumerate(fases):
    ng = len(dom_nolead[dom_nolead.fase == f])
    ax.text(i + w / 2, dom_g[i] + .02, f"n={ng}", ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(fases)
ax.set_title("Tasa de victoria por fase del torneo", fontweight="bold", loc="left")
ax.set_ylabel("% que ganó"); pct(ax); ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.10))
fig.tight_layout(); fig.savefig("reports/figures/fig4_fase.png", bbox_inches="tight"); plt.close(fig)

# ---------- Resúmenes para el texto del reporte ----------
print("\n[Tabla] Resultado por estado al descanso (%):")
print((ct * 100).round(0).astype(int))
print("\n[Tabla] xg_diff por resultado:")
print(df.groupby("resultado_ft")["xg_diff"].agg(["mean", "count"]).round(3))
print(f"\nDominio engañoso (xg_diff>0.3 y NO líder): n={len(dom_nolead)} "
      f"winrate={(dom_nolead.resultado_ft=='W').mean():.3f} "
      f"drawrate={(dom_nolead.resultado_ft=='D').mean():.3f} "
      f"lossrate={(dom_nolead.resultado_ft=='L').mean():.3f}")
print(f"Iba ganando: n={len(lead)} winrate={(lead.resultado_ft=='W').mean():.3f}")
print("\nWinrate dominante-no-líder por fase:",
      {f: round(wr_sub(dom_nolead[dom_nolead.fase == f]), 3) for f in fases})
