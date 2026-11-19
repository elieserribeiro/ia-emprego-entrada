"""Etapa 5 — Estatísticas descritivas do artigo.

Tabelas: 1º semestre (T1+T2) de 2022, 2023 e 2026, por quartil de exposição × idade (Brasil, NE, RN).
Figura: índice trimestral de admissões em soma móvel de 4 trimestres (remove a sazonalidade), base 2022T3 = 100.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .util import caminho, log

MEDIDAS = ["adm", "adm_sup", "adm_base", "adm_aprendiz", "des"]


def _grupo(cfg, faixa):
    if faixa in cfg["idade"]["jovem"]:
        return "18-24"
    if faixa == "25_29":
        return "25-29"
    if faixa in cfg["idade"]["adulto"]:
        return "30+"
    return "até 17"


def _com_exposicao(cfg, df):
    ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"))[["cbo4", "expo_quartil"]]
    df = df.merge(ex, on="cbo4", how="left")
    df["quartil"] = df.expo_quartil.map(lambda q: f"Q{int(q)}" if pd.notna(q) else "sem escore")
    df["idade"] = df.faixa.map(lambda f: _grupo(cfg, f))
    return df


def _tabela(df, medidas, anos):
    t = df[df.ano.isin(anos)].groupby(["quartil", "idade", "ano"])[medidas].sum().unstack("ano")
    for m in medidas:
        if all((m, a) in t for a in anos):
            t[(m, "var_26_23_%")] = 100 * (t[(m, anos[-1])] / t[(m, anos[1])] - 1)
            t[(m, "var_26_22_%")] = 100 * (t[(m, anos[-1])] / t[(m, anos[0])] - 1)
    t.columns = [f"{m}_{a}" for m, a in t.columns]
    return t.reset_index()


def rodar(cfg):
    anos = [2022, 2023, 2026]
    pr_t = _com_exposicao(cfg, pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet")))
    het = _com_exposicao(cfg, pd.read_parquet(caminho(cfg, "processed", "painel_heterogeneidade.parquet")))
    pr = pr_t[pr_t.tri.isin([1, 2])]
    het = het[het.tri.isin([1, 2])]
    if not set(anos) <= set(pr.ano):
        anos = sorted(pr.ano.unique())[-3:]
        log.warning("tabelas de 1º semestre com os anos disponíveis: %s", anos)
    med = [m for m in MEDIDAS if m in pr]
    out = caminho(cfg, "outputs", "tabelas", "x").parent

    _tabela(pr, med, anos).to_csv(out / "desc_brasil_quartil_idade.csv", index=False)
    for dim, valor, nome in (("nordeste", "NE", "nordeste"), ("rn", "RN", "rn")):
        _tabela(het[(het.dim == dim) & (het.valor == valor)], med, anos).to_csv(
            out / f"desc_{nome}_quartil_idade.csv", index=False)

    # composição: participação do quartil mais exposto nas admissões, por grupo
    comp = (het[het.ano.isin(anos)].assign(q4=lambda d: (d.quartil == "Q4") * d.adm)
            .groupby(["dim", "valor", "ano"])[["q4", "adm"]].sum())
    comp["pct_admissoes_em_Q4"] = 100 * comp.q4 / comp.adm
    comp.reset_index().to_csv(out / "desc_composicao_Q4.csv", index=False)

    # figura: índice trimestral, soma móvel de 4 trimestres, base 2022T3 = 100
    ref = cfg["periodo"]["trimestre_referencia"]
    s = pr_t[pr_t.quartil.str.startswith("Q") & pr_t.idade.isin(["18-24", "30+"])]
    s = s.groupby(["idade", "quartil", "t"]).adm.sum().reset_index().sort_values("t")
    s["movel4"] = s.groupby(["idade", "quartil"]).adm.transform(lambda x: x.rolling(4).sum())
    base = s[s.t == ref].set_index(["idade", "quartil"]).movel4
    s["indice"] = 100 * s.movel4 / s.set_index(["idade", "quartil"]).index.map(base)
    s.to_csv(out / "desc_indice_trimestral_movel4.csv", index=False)
    s = s.dropna(subset=["indice"])
    ts = sorted(s.t.unique())
    pos = {t: i for i, t in enumerate(ts)}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, grupo in zip(axs, ["18-24", "30+"]):
        for q, d in s[s.idade == grupo].groupby("quartil"):
            ax.plot(d.t.map(pos), d.indice, label=q, lw=2.5 if q == "Q4" else 1.2,
                    color="#1f5fbf" if q == "Q4" else None, alpha=1 if q == "Q4" else 0.6)
        if cfg["periodo"]["trimestre_chatgpt"] in pos:
            ax.axvline(pos[cfg["periodo"]["trimestre_chatgpt"]], ls="--", color="0.6", lw=0.8)
        ax.axhline(100, color="0.8", lw=0.8)
        ax.set_title(f"Admissões, {grupo} anos (4 trim. móveis; 2022T3 = 100)", fontsize=10)
        marc = [t for t in ts if t % 10 == 4]
        ax.set_xticks([pos[t] for t in marc], [f"{t // 10}T4" for t in marc], fontsize=8)
    axs[0].legend(title="Quartil de exposição", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", "indice_admissoes_quartil.png"), dpi=200)
    plt.close(fig)
    log.info("descritivas gravadas em %s", out)
