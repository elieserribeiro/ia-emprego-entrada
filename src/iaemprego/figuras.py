"""Figuras e tabelas finais do artigo "Ponte estreita", em português e inglês.

Não roda modelos: lê os CSVs de outputs/tabelas (e, para a Tabela 1, uma contagem rápida no DuckDB, guardada
em cache). Saída: outputs/artigo/pt/ e outputs/artigo/en/ com figura1..6.png e tabela1..5 (.md e .csv).
Coeficientes PPML são mostrados como 100 × coeficiente (≈ variação % por desvio-padrão de exposição).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .util import caminho, log

AZUL, LARANJA, CINZA, VERDE = "#1f5fbf", "#d0612a", "#7a7a7a", "#2e8b57"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight"})

T = {  # textos por idioma
    "pt": {
        "dec": ",", "ano": "Ano", "chatgpt": "ChatGPT", "pandemia": "pandemia", "registro": "registro de raça",
        "pct": "efeito (% por DP de exposição)", "pp": "p.p.", "ref22": "referência: mesmo trimestre de 2022",
        "f1_t": "Exposição à IA por grande grupo ocupacional (CBO)",
        "f1_x": "Escore médio de exposição (OIT, 0–1), ponderado pelas admissões de 2022",
        "f1_adm": "das admissões",
        "f2_t": "Mercado como um todo: ocupações mais × menos expostas (todas as idades)",
        "f2_p": ["Admissões", "Admissões com ensino superior", "Salário de admissão (log × 100)"],
        "f3_t": "Admissões de jovens nas ocupações expostas, relativas aos adultos (30+)",
        "f3_l": ["18–24 anos (primeira entrada)", "25–29 anos (degrau seguinte)"],
        "f4_t": "Jovens de 25–29 anos: a queda acelerou depois do ChatGPT?",
        "f4_l": ["coeficiente (event study, 2020+)", "tendência com quebra em 2022T4",
                 "se a inclinação anterior continuasse"],
        "f5_t": "Jovens de 25–29 anos: efeito por grupo (pós-2023, por DP de exposição)",
        "f5_p": ["Admissões", "Admissões com ensino superior"],
        "f5_pd": "p da diferença (Holm)",
        "f6_t": "Diferença negra − branca nas admissões em ocupações expostas",
        "f6_p": ["18–24 anos", "25–29 anos"],
        "grupos": {"homem": "Homens", "mulher": "Mulheres", "branca": "Brancas", "negra": "Negras",
                   "medio": "Ensino médio", "superior": "Ensino superior", "1": "Norte", "2": "Nordeste",
                   "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"},
        "dims": {"sexo": "Sexo", "raca": "Raça/cor", "instrucao": "Escolaridade", "regiao": "Região"},
        "gg": {"0": "Forças armadas", "1": "Dirigentes e gerentes", "2": "Profissionais das ciências e artes",
               "3": "Técnicos de nível médio", "4": "Serviços administrativos", "5": "Serviços e comércio",
               "6": "Agropecuária, florestal e pesca", "7": "Produção industrial (7)",
               "8": "Produção industrial contínua (8)", "9": "Manutenção e reparação"},
        "desf": {"adm": "Admissões", "adm_sup": "Admissões com superior", "adm_base": "Admissões na base salarial",
                 "des": "Desligamentos", "ocup": "Ocupados", "formal": "Ocupados formais",
                 "informal": "Ocupados informais", "recente": "Recém-entrados (< 1 ano)",
                 "recente_formal": "Recém-entrados formais", "recente_informal": "Recém-entrados informais"},
        "nota": "Nota: 100 × coeficiente PPML (≈ % por desvio-padrão de exposição à IA), pós-2023 contra 2022. "
                "Erros-padrão entre parênteses, agrupados por ocupação. *** p<0,01; ** p<0,05; * p<0,10.",
    },
    "en": {
        "dec": ".", "ano": "Year", "chatgpt": "ChatGPT", "pandemia": "pandemic", "registro": "race recording",
        "pct": "effect (% per SD of exposure)", "pp": "p.p.", "ref22": "reference: same quarter of 2022",
        "f1_t": "AI exposure by major occupational group (CBO)",
        "f1_x": "Mean exposure score (ILO, 0–1), weighted by 2022 hires",
        "f1_adm": "of hires",
        "f2_t": "Labor market as a whole: more × less exposed occupations (all ages)",
        "f2_p": ["Hires", "Hires with college degree", "Hiring wage (log × 100)"],
        "f3_t": "Hiring of young workers in exposed occupations, relative to adults (30+)",
        "f3_l": ["Ages 18–24 (first entry)", "Ages 25–29 (next step)"],
        "f4_t": "Ages 25–29: did the decline accelerate after ChatGPT?",
        "f4_l": ["coefficient (event study, 2020+)", "trend with a break in 2022Q4", "if the prior slope continued"],
        "f5_t": "Ages 25–29: effect by group (post-2023, per SD of exposure)",
        "f5_p": ["Hires", "Hires with college degree"],
        "f5_pd": "p of difference (Holm)",
        "f6_t": "Black − white difference in hires in exposed occupations",
        "f6_p": ["Ages 18–24", "Ages 25–29"],
        "grupos": {"homem": "Men", "mulher": "Women", "branca": "White", "negra": "Black",
                   "medio": "High school", "superior": "College", "1": "North", "2": "Northeast",
                   "3": "Southeast", "4": "South", "5": "Center-West"},
        "dims": {"sexo": "Sex", "raca": "Race", "instrucao": "Education", "regiao": "Region"},
        "gg": {"0": "Armed forces", "1": "Managers", "2": "Professionals", "3": "Technicians",
               "4": "Clerical support", "5": "Services and sales", "6": "Agriculture, forestry, fishing",
               "7": "Industrial production (7)", "8": "Continuous industrial production (8)",
               "9": "Maintenance and repair"},
        "desf": {"adm": "Hires", "adm_sup": "Hires with college degree", "adm_base": "Hires at the wage floor",
                 "des": "Separations", "ocup": "Employed", "formal": "Formal employed",
                 "informal": "Informal employed", "recente": "Recent entrants (< 1 year)",
                 "recente_formal": "Formal recent entrants", "recente_informal": "Informal recent entrants"},
        "nota": "Note: 100 × PPML coefficient (≈ % per SD of AI exposure), post-2023 vs. 2022. Standard errors in "
                "parentheses, clustered by occupation. *** p<0.01; ** p<0.05; * p<0.10.",
    },
}


# --------------------------------------------------------------------------------------------- utilidades
def _tab(cfg, nome):
    arq = caminho(cfg, "outputs", "tabelas", nome)
    return pd.read_csv(arq) if arq.exists() else None


def _rot_t(t, lang):
    t = int(t)
    return f"{t // 10}{'T' if lang == 'pt' else 'Q'}{t % 10}"


def _num(x, lang, casas=1):
    s = f"{x:.{casas}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    s = s.replace("-", "−")
    return s.replace(".", ",") if lang == "pt" else s


def _estrelas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _cel(coef, ep, p, lang, esc=100):
    if pd.isna(coef):
        return "—"
    return f"{_num(esc * coef, lang)}{_estrelas(p)} ({_num(esc * ep, lang)})"


def _eixo_tempo(ax, per, lang, cfg, ref_zero=None, marcar=True):
    """Marcações comuns: pandemia (cinza), ChatGPT (tracejado), rótulos anuais."""
    per = list(per)
    x = np.arange(len(per))
    a, b = cfg["periodo"]["pandemia"]
    dentro = [i for i, v in enumerate(per) if a <= v <= b]
    if dentro and marcar:
        ax.axvspan(min(dentro) - 0.5, max(dentro) + 0.5, color="0.93", zorder=0)
        ax.text(np.mean(dentro), 1.0, T[lang]["pandemia"], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7, color="0.45")
    cg = [i for i, v in enumerate(per) if v == cfg["periodo"]["trimestre_chatgpt"]]
    if cg:
        ax.axvline(cg[0], ls="--", color="0.45", lw=0.9)
        ax.text(cg[0] + 0.2, 0.93, T[lang]["chatgpt"], transform=ax.get_xaxis_transform(), fontsize=7,
                color="0.35")
    ax.axhline(0, color="0.6", lw=0.7)
    marc = [i for i, v in enumerate(per) if int(v) % 10 == 1]
    ax.set_xticks(marc, [str(int(per[i]) // 10) for i in marc])
    return x


def _com_ref(df, refs):
    z = pd.DataFrame({"periodo": refs, "coef": 0.0, "ic_inf": 0.0, "ic_sup": 0.0, "ep": 0.0})
    df = df[["periodo", "coef", "ic_inf", "ic_sup", "ep"]].dropna(subset=["periodo"])
    return pd.concat([df, z]).drop_duplicates("periodo").sort_values("periodo").reset_index(drop=True)


def _salvar(fig, pasta, nome):
    if pasta.name == "pt":  # vírgula decimal nos eixos numéricos (os eixos de anos têm rótulos fixos)
        from matplotlib.ticker import FuncFormatter, ScalarFormatter
        fmt = FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−").replace(".", ","))
        for ax in fig.axes:
            for eixo in (ax.xaxis, ax.yaxis):
                if isinstance(eixo.get_major_formatter(), ScalarFormatter):
                    eixo.set_major_formatter(fmt)
    fig.savefig(pasta / f"{nome}.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- figuras
def figura1(cfg, lang, pasta):
    ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"))
    ex["gg"] = ex.cbo4.astype(str).str.zfill(4).str[0]
    ex = ex[ex.gg != "0"]
    g = ex.groupby("gg").apply(lambda d: pd.Series({"expo": np.average(d.expo, weights=d.adm_ref + 1e-9),
                                                    "adm": d.adm_ref.sum()}), include_groups=False)
    g["share"] = 100 * g.adm / g.adm.sum()
    g = g.sort_values("expo")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    cores = [AZUL for _ in g.index]
    ax.barh([T[lang]["gg"][k] for k in g.index], g.expo, color=cores)
    for i, (e, s) in enumerate(zip(g.expo, g.share)):
        ax.text(e + 0.005, i, f"{_num(e, lang, 2)}  ({_num(s, lang)}% {T[lang]['f1_adm']})", va="center",
                fontsize=7.5, color="0.25")
    ax.set_xlabel(T[lang]["f1_x"])
    ax.set_xlim(0, g.expo.max() * 1.45)
    ax.set_title(T[lang]["f1_t"], loc="left")
    _salvar(fig, pasta, "figura1")


def figura2(cfg, lang, pasta):
    m = _tab(cfg, "mercado.csv")
    if m is None:
        return
    refs = [20221, 20222, 20223, 20224]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, y, tit in zip(axes, ("adm", "adm_sup", "log_salario_admissao"), T[lang]["f2_p"]):
        s = m[(m.fonte == "CAGED") & (m.desfecho == y) & (m.modelo == "mercado_event_shift_share")]
        if s.empty:
            s = m[(m.fonte == "CAGED") & (m.desfecho == y) & (m.modelo == "mercado_event")]
        s = _com_ref(s, refs)
        x = _eixo_tempo(ax, s.periodo, lang, cfg)
        r = x[s.periodo // 10 == 2022]
        ax.axvspan(r.min() - 0.5, r.max() + 0.5, color="#fde9d9", zorder=0)
        ax.errorbar(x, 100 * s.coef, yerr=[100 * (s.coef - s.ic_inf), 100 * (s.ic_sup - s.coef)], fmt="o-",
                    color=AZUL, ms=3, lw=1, capsize=1.5)
        ax.set_title(tit, loc="left", fontsize=9)
    axes[0].set_ylabel(T[lang]["pct"])
    fig.suptitle(T[lang]["f2_t"], x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.01, -0.03, T[lang]["ref22"] + ("; com controles de pandemia e ciclo setorial" if lang == "pt"
                                               else "; with pandemic and sector-cycle controls"),
             fontsize=7, color="0.4")
    fig.tight_layout()
    _salvar(fig, pasta, "figura2")


def figura3(cfg, lang, pasta):
    r = _tab(cfg, "modelos_principais.csv")
    if r is None:
        return
    ref = cfg["periodo"]["trimestre_referencia"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, (mod, cor, lab) in enumerate((("tripla_event_com_controles", AZUL, T[lang]["f3_l"][0]),
                                         ("tripla_25_29_event_com_controles", LARANJA, T[lang]["f3_l"][1]))):
        s = r[(r.modelo == mod) & (r.desfecho == "adm") & (r["var"] == "treat")]
        s = _com_ref(s, [ref])
        x = _eixo_tempo(ax, s.periodo, lang, cfg) if k == 0 else np.arange(len(s))
        ax.errorbar(x + (k - 0.5) * 0.22, 100 * s.coef, yerr=[100 * (s.coef - s.ic_inf), 100 * (s.ic_sup - s.coef)],
                    fmt="o-", color=cor, ms=3.5, lw=1, capsize=1.5, label=lab)
    ax.set_ylabel(T[lang]["pct"])
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(T[lang]["f3_t"], loc="left")
    _salvar(fig, pasta, "figura3")


def figura4(cfg, lang, pasta):
    rob, ac = _tab(cfg, "robustez.csv"), _tab(cfg, "aceleracao.csv")
    if rob is None or ac is None:
        return
    ref, cg = cfg["periodo"]["trimestre_referencia"], cfg["periodo"]["trimestre_chatgpt"]
    s = rob[(rob.grupo == "25_29") & (rob.variante == "serie_completa_event") & rob.periodo.notna()]
    s = _com_ref(s, [ref])
    a = ac[(ac.grupo == "25_29") & (ac.modelo == "aceleracao_sem_pandemia")].set_index("termo").coef
    idx = lambda t: (np.asarray(t, int) // 10 - 2020) * 4 + np.asarray(t, int) % 10 - 1
    k = idx(s.periodo) - idx([cg])[0]
    k_ref = idx([ref])[0] - idx([cg])[0]
    reta = a["inclinacao_pre"] * (k - k_ref) + a["mudanca_inclinacao"] * np.maximum(k, 0)
    contraf = a["inclinacao_pre"] * (k - k_ref)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = _eixo_tempo(ax, s.periodo, lang, cfg)
    ax.errorbar(x, 100 * s.coef, yerr=[100 * (s.coef - s.ic_inf), 100 * (s.ic_sup - s.coef)], fmt="o", color=AZUL,
                ms=3.5, capsize=1.5, label=T[lang]["f4_l"][0])
    ax.plot(x, 100 * reta, color=LARANJA, lw=1.8, label=T[lang]["f4_l"][1])
    ax.plot(x[k >= 0], 100 * contraf[k >= 0], color=LARANJA, lw=1.2, ls=":", label=T[lang]["f4_l"][2])
    ax.set_ylabel(T[lang]["pct"])
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(T[lang]["f4_t"], loc="left")
    _salvar(fig, pasta, "figura4")


def figura5(cfg, lang, pasta):
    d = _tab(cfg, "diferencas_grupos.csv")
    if d is None:
        return
    d = d[d.grupo == "25_29"].copy()
    d["v"] = d.termo.str.replace("efeito::", "", regex=False).astype(str).str.replace(r"\.0$", "", regex=True)
    blocos = [("sexo", ["homem", "mulher"]), ("raca", ["branca", "negra"]), ("instrucao", ["medio", "superior"]),
              ("regiao", ["1", "2", "3", "4", "5"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, y, tit in zip(axes, ("adm", "adm_sup"), T[lang]["f5_p"]):
        linhas, pos, ylab, yy = [], [], [], 0
        for dim, niveis in blocos:
            if dim == "instrucao" and y == "adm_sup":
                yy += len(niveis) + 1
                continue
            s = d[(d.dim == dim) & (d.desfecho == y) & d.termo.str.startswith("efeito::")].set_index("v")
            for v in niveis:
                if v in s.index:
                    r = s.loc[v]
                    ax.errorbar(100 * r.coef, yy, xerr=196 * r.ep, fmt="o", color=AZUL, ms=4, capsize=2)
                yy += 1
            dd = d[(d.dim == dim) & (d.desfecho == y) & (d.termo.str.startswith("diferenca::") if dim != "regiao"
                                                           else d.termo.eq("wald_todos_iguais"))]
            if len(dd) and dd.p_holm.notna().any():
                p = dd.p_holm.min()
                ax.text(1.0, yy - len(niveis) / 2 - 0.5, f"p = {_num(p, lang, 3)}", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7.5, color=LARANJA if p < 0.05 else "0.45",
                        fontweight="bold" if p < 0.05 else "normal")
            yy += 1
        ax.axvline(0, color="0.6", lw=0.7)
        ax.set_title(tit, loc="left", fontsize=9)
        ax.set_xlabel(T[lang]["pct"])
    ticks, labs, yy = [], [], 0
    for dim, niveis in blocos:
        for v in niveis:
            ticks.append(yy)
            labs.append(f"{T[lang]['grupos'][v]}")
            yy += 1
        yy += 1
    axes[0].set_yticks(ticks, labs)
    axes[0].invert_yaxis()
    fig.suptitle(T[lang]["f5_t"], x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.99, -0.02, T[lang]["f5_pd"], ha="right", fontsize=7, color="0.4")
    fig.tight_layout()
    _salvar(fig, pasta, "figura5")


def figura6(cfg, lang, pasta):
    e = _tab(cfg, "evento_raca.csv")
    if e is None:
        return
    ref = cfg["periodo"]["trimestre_referencia"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, g, tit in zip(axes, ("18_24", "25_29"), T[lang]["f6_p"]):
        s = e[(e.grupo == g) & (e.desfecho == "adm") & (e.termo == "diferenca::negra-branca")]
        s = _com_ref(s, [ref])
        x = _eixo_tempo(ax, s.periodo, lang, cfg)
        reg = [i for i, v in enumerate(s.periodo) if v == 20241]
        if reg:
            ax.axvline(reg[0] - 0.5, ls="--", color=LARANJA, lw=1)
            ax.text(reg[0] - 0.3, 0.06, T[lang]["registro"], transform=ax.get_xaxis_transform(), fontsize=7,
                    color=LARANJA)
        ax.errorbar(x, 100 * s.coef, yerr=[100 * (s.coef - s.ic_inf), 100 * (s.ic_sup - s.coef)], fmt="o-",
                    color=AZUL, ms=3, lw=1, capsize=1.5)
        ax.set_title(tit, loc="left", fontsize=9)
    axes[0].set_ylabel("p.p. (log × 100)")
    fig.suptitle(T[lang]["f6_t"], x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _salvar(fig, pasta, "figura6")


# --------------------------------------------------------------------------------------------- tabelas
def _md(df: pd.DataFrame, titulo, nota=None, esquerda=1):
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * esquerda + ["---:"] * (len(df.columns) - esquerda)) + "|"
    linhas = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    txt = f"**{titulo}**\n\n" + "\n".join([cab, sep] + linhas) + "\n"
    return txt + (f"\n{nota}\n" if nota else "")


def _gravar(df, pasta, nome, titulo, nota=None, esquerda=1):
    df.to_csv(pasta / f"{nome}.csv", index=False)
    (pasta / f"{nome}.md").write_text(_md(df, titulo, nota, esquerda), encoding="utf-8")


def _tabela1_dados(cfg):
    cache = caminho(cfg, "outputs", "artigo", "cache_tabela1.csv")
    if cache.exists():
        return pd.read_csv(cache)
    from .caged import conexao
    con = conexao(cfg)
    t = con.sql("""SELECT comp // 100 AS ano, count(*) AS linhas,
                          sum(CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END) AS adm,
                          sum(CASE WHEN saldo_bruto = -1 THEN sinal ELSE 0 END) AS des
                   FROM mov_ajustado GROUP BY 1 ORDER BY 1""").df()
    con.close()
    t.to_csv(cache, index=False)
    return t


def tabelas(cfg, lang, pasta):
    L = T[lang]
    mil = lambda v: f"{int(v):,}".replace(",", "." if lang == "pt" else ",")
    # Tabela 1
    try:
        t = _tabela1_dados(cfg)
        t = pd.concat([t, pd.DataFrame([{"ano": "Total", "linhas": t.linhas.sum(), "adm": t.adm.sum(),
                                         "des": t.des.sum()}])], ignore_index=True)
        t["saldo"] = t.adm - t.des
        cols = (["Ano", "Movimentações", "Admissões", "Desligamentos", "Saldo"] if lang == "pt"
                else ["Year", "Records", "Hires", "Separations", "Net"])
        ult = t.ano[t.ano != "Total"].astype(int).max()
        anos = t.ano.map(lambda a: (f"{a} ({'jan–set' if lang == 'pt' else 'Jan–Sep'})" if str(a) == str(ult)
                                    else a))
        out = pd.DataFrame({cols[0]: anos, cols[1]: t.linhas.map(mil), cols[2]: t.adm.map(mil),
                            cols[3]: t.des.map(mil),
                            cols[4]: t.saldo.map(lambda v: ("+" if v > 0 else "") + mil(v).replace("-", "−"))})
        _gravar(out, pasta, "tabela1", "Tabela 1 – Movimentações do Novo CAGED, jan/2020–set/2026" if lang == "pt"
                else "Table 1 – Novo CAGED records, Jan 2020–Sep 2026",
                "Fonte: microdados do Novo CAGED (MTE). Elaboração própria." if lang == "pt"
                else "Source: Novo CAGED microdata (Ministry of Labor). Authors' calculations.")
    except Exception as e:
        log.warning("tabela 1: %s", e)
    # Tabela 2: validações
    v = pd.read_csv(caminho(cfg, "outputs", "validacao", "validacao_totais.csv")) if \
        caminho(cfg, "outputs", "validacao", "validacao_totais.csv").exists() else None
    arq_pn = caminho(cfg, "outputs", "validacao", "pnad_taxas_brasil.csv")
    pn = pd.read_csv(arq_pn) if arq_pn.exists() else None
    oficial = {20243: (6.4, 38.8), 20253: (5.6, 37.8), 20262: (5.4, 37.4), 20263: (5.2, 37.5)}
    linhas = []
    if v is not None:
        for serie, rot in (("mte_tabela5_sem_ajustes", "CAGED × MTE Tabela 5" if lang == "pt" else "CAGED × MTE Table 5"),
                           ("mte_tabela5_1_com_ajustes", "CAGED × MTE Tabela 5.1" if lang == "pt" else "CAGED × MTE Table 5.1")):
            s = v[v.serie == serie]
            if len(s):
                linhas.append([rot, ("maior diferença mensal nas admissões" if lang == "pt" else "largest monthly gap in hires"),
                               _num(s.dif_adm_pct.abs().max(), lang, 3) + "%", "—"])
    if pn is not None:
        p = pn.set_index("t")
        for t_, (dz, inf) in oficial.items():
            if t_ in p.index:
                linhas.append([f"PNAD {_rot_t(t_, lang)}",
                               "desocupação / informalidade" if lang == "pt" else "unemployment / informality",
                               f"{_num(p.loc[t_, 'taxa_desocupacao_%'], lang, 2)}% / {_num(p.loc[t_, 'taxa_informalidade_%'], lang, 2)}%",
                               f"{_num(dz, lang)}% / {_num(inf, lang)}%"])
    if linhas:
        cols = (["Base", "Indicador", "Microdados", "Oficial"] if lang == "pt"
                else ["Source", "Indicator", "Microdata", "Official"])
        _gravar(pd.DataFrame(linhas, columns=cols), pasta, "tabela2",
                "Tabela 2 – Validação dos microdados" if lang == "pt" else "Table 2 – Microdata validation",
                "Fontes: MTE (Tabelas 5 e 5.1 do Novo CAGED); IBGE (PNAD Contínua, divulgações trimestrais)."
                if lang == "pt" else "Sources: Ministry of Labor (Novo CAGED Tables 5 and 5.1); IBGE (PNAD Contínua releases).",
                esquerda=2)
    # Tabela 3: resultados principais
    r = _tab(cfg, "modelos_principais.csv")
    if r is not None:
        linhas = []
        for y in ("adm", "adm_sup", "adm_base", "des"):
            row = [L["desf"][y]]
            for mod in ("tripla_pooled_com_controles", "tripla_25_29_pooled_com_controles"):
                s = r[(r.modelo == mod) & (r.desfecho == y) & (r.termo == "treat_post")]
                row.append(_cel(s.coef.iloc[0], s.ep.iloc[0], s.p.iloc[0], lang) if len(s) else "—")
            linhas.append(row)
        cols = ["", "18–24", "25–29"] if lang == "pt" else ["", "Ages 18–24", "Ages 25–29"]
        _gravar(pd.DataFrame(linhas, columns=cols), pasta, "tabela3",
                "Tabela 3 – Tripla diferença: jovens × adultos (30+) nas ocupações expostas à IA" if lang == "pt"
                else "Table 3 – Triple difference: young × adult (30+) workers in AI-exposed occupations", L["nota"])
    # Tabela 4: PNAD
    p = _tab(cfg, "pnad_modelos.csv")
    if p is not None:
        linhas = []
        for y in ("ocup", "formal", "informal", "recente", "recente_formal", "recente_informal"):
            row = [L["desf"][y]]
            for g, mod in (("25_29", "pnad_pooled"), ("25_29", "pnad_pooled_cod3"), ("25_29", "pnad_pooled_fase_2020"),
                           ("18_24", "pnad_pooled")):
                s = p[(p.grupo == g) & (p.modelo == mod) & (p.desfecho == y) & (p.termo == "treat_post")]
                row.append(_cel(s.coef.iloc[0], s.ep.iloc[0], s.p.iloc[0], lang) if len(s) else "—")
            linhas.append(row)
        cols = (["", "25–29", "25–29 (COD 3 díg.)", "25–29 (2020+, fases)", "18–24"] if lang == "pt"
                else ["", "Ages 25–29", "25–29 (3-digit COD)", "25–29 (2020+, phases)", "Ages 18–24"])
        _gravar(pd.DataFrame(linhas, columns=cols), pasta, "tabela4",
                "Tabela 4 – PNAD Contínua: ocupação, formalidade e entrada" if lang == "pt"
                else "Table 4 – PNAD Contínua: employment, formality and entry", L["nota"])
    # Tabela 5: robustez
    rb = _tab(cfg, "robustez.csv")
    if rb is not None:
        nomes = {"pt": {"principal": "Principal", "ate_2026T2": "Até 2026T2", "sem_controles_pandemia": "Sem controles da pandemia",
                        "serie_completa_fe_fase": "Série 2020+ com fases", "incluir_2021_fe_fase": "Desde 2021 com fases",
                        "base_2023": "Base 2023 (aceleração)", "so_1o_semestre": "Só 1º semestres",
                        "jovem_18_29": "Jovem = 18–29", "expo_quartil_topo": "Quartil mais exposto",
                        "expo_gradientes_3_4": "Gradientes OIT 3–4", "sem_cbo_imputadas": "Sem CBOs imputadas",
                        "sem_adm_publica_e_TI": "Sem adm. pública e TI", "sem_aprendiz_intermitente": "Sem aprendiz/intermitente",
                        "sem_ajustes_so_MOV": "Só declarações no prazo (MOV)"},
                 "en": {"principal": "Baseline", "ate_2026T2": "Through 2026Q2", "sem_controles_pandemia": "No pandemic controls",
                        "serie_completa_fe_fase": "2020+ with phases", "incluir_2021_fe_fase": "From 2021 with phases",
                        "base_2023": "2023 base (acceleration)", "so_1o_semestre": "First halves only",
                        "jovem_18_29": "Young = 18–29", "expo_quartil_topo": "Top exposure quartile",
                        "expo_gradientes_3_4": "ILO gradients 3–4", "sem_cbo_imputadas": "No imputed occupations",
                        "sem_adm_publica_e_TI": "Excl. public admin. and IT", "sem_aprendiz_intermitente": "Excl. apprentices/intermittent",
                        "sem_ajustes_so_MOV": "On-time records only (MOV)"}}[lang]
        linhas = []
        for var, rot in nomes.items():
            row = [rot]
            for g in ("18_24", "25_29"):
                s = rb[(rb.grupo == g) & (rb.variante == var) & (rb.termo == "treat_post")]
                row.append(_cel(s.coef.iloc[0], s.ep.iloc[0], s.p.iloc[0], lang) if len(s) else "—")
            linhas.append(row)
        cols = ["Variante", "18–24", "25–29"] if lang == "pt" else ["Specification", "Ages 18–24", "Ages 25–29"]
        _gravar(pd.DataFrame(linhas, columns=cols), pasta, "tabela5",
                "Tabela 5 – Robustez: admissões, jovens × adultos" if lang == "pt"
                else "Table 5 – Robustness: hires, young × adult workers", L["nota"])


def rodar(cfg):
    for lang in ("pt", "en"):
        pasta = Path(caminho(cfg, "outputs", "artigo", lang, "x")).parent
        pasta.mkdir(parents=True, exist_ok=True)
        for nome, f in (("figura1", figura1), ("figura2", figura2), ("figura3", figura3), ("figura4", figura4),
                        ("figura5", figura5), ("figura6", figura6), ("tabelas", tabelas)):
            try:
                f(cfg, lang, pasta)
            except Exception as e:
                log.warning("%s (%s) falhou: %s", nome, lang, e)
        log.info("artigo (%s): %s", lang, sorted(p.name for p in pasta.iterdir()))
