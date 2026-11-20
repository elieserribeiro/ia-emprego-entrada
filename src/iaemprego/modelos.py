"""Etapas 6, 7 e 9 — Modelos (pyfixest), em trimestres.

Modelo principal (diferença tripla, PPML), célula = família CBO c × faixa a × UF u × trimestre t, amostra 2022T1+:
  E[Y] = exp( Σ_{t≠2022T3} β_t·Expo_c·Jovem_a·1[t]            ← efeito diferencial da IA sobre a entrada
              + Σ_t φ_t·Choque_c·Jovem_a·1[t] + Σ_t ψ_t·Boom_c·Jovem_a·1[t]   ← controles da pandemia
              + α_{c,a} + γ_{c,t} + δ_{a,u,t} )
  γ_{c,t}: qualquer choque da ocupação comum a todas as idades (inclusive pandemia e sazonalidade da ocupação)
  δ_{a,u,t}: ciclo e sazonalidade por idade e UF (juros, salário mínimo, políticas para jovens)
  No pooled soma-se σ_{c,a,trimestre-do-ano}: sazonalidade própria de ocupação × idade (ex.: jovens no
  comércio em dezembro). No event study ela é colinear com as dummies de trimestre e fica de fora.
  β_{2022T1}, β_{2022T2}: pré-tendência.  "pooled": um só β para 2023T1+ contra 2022T1–T3 (2022T4 fora).
Erros clusterizados por família CBO.
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

from .util import caminho, log

DESFECHOS = {
    "adm": "Admissões (todas)",
    # "adm_1emp" (primeiro emprego) fica FORA: a partir de 2022 ~99,9% das admissões vêm com tipomov = 97
    # ("tipo ignorado", eSocial), então a medida cairia a zero por mudança de registro, não por efeito real.
    # Ver scripts/diagnostico_tipomov.py e outputs/validacao/tipomov_por_ano.csv.
    "adm_sup": "Admissões com superior completo",
    "adm_base": "Admissões na base salarial (≤ p25 da família no trimestre)",
    # "adm_aprendiz" sai da tripla diferença: aprendiz tem 14–24 anos por lei, quase não há adultos 30+
    # para comparar. Vai para um modelo próprio só entre jovens (ver aprendizes()).
    "des": "Desligamentos",
}
FE = "cbo4:faixa + cbo4:t + faixa:uf:t"
FE_SEM_UF = "cbo4:faixa + cbo4:t + faixa:t"
SAZ = " + cbo4:faixa:tri"  # sazonalidade ocupação × idade: só no pooled (no event study é colinear com as
#                            dummies de trimestre — cada trimestre do ano perderia um coeficiente)


def _per(cfg):
    p = cfg["periodo"]
    return p["trimestre_referencia"], p["inicio_amostra_principal"], p["inicio_pos"], p["trimestre_chatgpt"]


def preparar(cfg, painel: pd.DataFrame, expo="expo_z", jovem_chave="jovem", sem_imputados=False,
             inicio_pos=None) -> pd.DataFrame:
    ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"))
    if sem_imputados:
        ex = ex[ex.pct_imputado < 0.5]
    df = painel.merge(ex[["cbo4", "expo", "expo_z", "expo_g34", "expo_quartil"]], on="cbo4", how="inner")
    try:
        pand = pd.read_parquet(caminho(cfg, "processed", "pandemia_cbo4.parquet"))[["cbo4", "choque_z", "boom_z"]]
        df = df.merge(pand, on="cbo4", how="left")
    except FileNotFoundError:
        df["choque_z"] = np.nan
        df["boom_z"] = np.nan
    df["tem_pandemia"] = df.choque_z.notna()
    df[["choque_z", "boom_z"]] = df[["choque_z", "boom_z"]].fillna(0.0)  # sem medida = média
    if expo == "expo_top":
        df["expo_top"] = (df.expo_quartil == 4).astype(float)
    extra_grupos = {"jovem_25_29": ["25_29"]}  # grupos tratados alternativos (sem mexer no config.yaml)
    jov = cfg["idade"].get(jovem_chave) or extra_grupos[jovem_chave]
    adu = cfg["idade"]["adulto"]
    df = df[df.faixa.isin(jov + adu)].copy()
    df["jovem"] = df.faixa.isin(jov).astype(int)
    df["treat"] = df[expo] * df.jovem
    df["chq_j"] = df.choque_z * df.jovem
    df["boom_j"] = df.boom_z * df.jovem
    df["expo_v"] = df[expo]
    pos = inicio_pos or cfg["periodo"]["inicio_pos"]
    df["post"] = (df.t >= pos).astype(int)
    df["treat_post"] = df.treat * df.post
    a, b = cfg["periodo"]["pandemia"]
    df["fase"] = np.where(df.t < a, "pre", np.where(df.t <= b, "pandemia", "pos"))
    if "uf" in df:
        df["uf"] = df.uf.astype(str)
    return df


def _tidy(fit, modelo, desfecho, extra=None):
    t = fit.tidy().reset_index().rename(columns={"Coefficient": "termo", "Estimate": "coef", "Std. Error": "ep",
                                                 "Pr(>|t|)": "p", "2.5%": "ic_inf", "97.5%": "ic_sup"})
    t["periodo"] = t.termo.map(lambda s: int(m.group(1)) if (m := re.search(r"::(\d+)", str(s))) else np.nan)
    t["var"] = t.termo.map(lambda s: "treat" if "treat" in str(s) else ("expo" if "expo" in str(s) else "controle"))
    t["modelo"], t["desfecho"], t["n_obs"] = modelo, desfecho, fit._N
    for k, v in (extra or {}).items():
        t[k] = v
    return t


def _controles(df, ref, var_t="t"):
    if (df.chq_j != 0).any() or (df.boom_j != 0).any():
        return f" + i({var_t}, chq_j, ref={ref}) + i({var_t}, boom_j, ref={ref})"
    return ""


def _controles_anuais(df, ref):
    if (df.chq_j != 0).any() or (df.boom_j != 0).any():
        a = int(ref) // 10 if int(ref) > 9999 else int(ref)
        return f" + i(ano, chq_j, ref={a}) + i(ano, boom_j, ref={a})"
    return ""


def triple(cfg, df, y="adm", ref=None, inicio=None, controles=True, pooled=False, efeitos=FE, rotulo=None,
           fase_fe=False, sazonal=True):
    ref0, ini0, _, chatgpt = _per(cfg)
    ref = ref or ref0
    inicio = inicio if inicio is not None else ini0
    d = df[(df.t >= inicio) & df[y].notna() & (df[y] >= 0)].copy()
    if pooled:
        d = d[d.t != chatgpt]
        if sazonal and "tri" in d and d.tri.nunique() > 1:
            efeitos = efeitos + SAZ
    if d[y].sum() <= 0:
        return None
    if fase_fe:  # amostras que incluem a pandemia: um nível próprio por ocupação × idade em cada fase
        efeitos = efeitos + " + cbo4:faixa:fase"
        d["t_ev"] = np.where(d.fase == "pos", d.t, 0)
        alvo = "treat_post" if pooled else f"i(t_ev, treat, ref={ref})"
    else:
        alvo = "treat_post" if pooled else f"i(t, treat, ref={ref})"
    usa_saz = SAZ.strip(" +") in efeitos
    if controles and usa_saz and "ano" in d:
        # com a sazonalidade ocupação × idade × trimestre, controles por TRIMESTRE seriam colineares
        # (a soma deles em cada trimestre do ano repete o efeito fixo sazonal): no pooled, variam por ANO
        rhs = alvo + _controles_anuais(d, ref)
    else:
        rhs = alvo + (_controles(d, ref) if controles else "")
    fit = pf.fepois(f"{y} ~ {rhs} | {efeitos}", data=d, vcov={"CRV1": "cbo4"})
    out = _tidy(fit, rotulo or ("tripla_pooled" if pooled else "tripla_event"), y)
    return out[out["var"] == "treat"] if not pooled else out[out.termo == "treat_post"]


def so_primeiro_semestre(cfg, df_trimestral: pd.DataFrame) -> pd.DataFrame:
    """Agrega T1+T2 de cada ano: robustez equivalente ao desenho por 1º semestres."""
    d = df_trimestral[df_trimestral.tri.isin([1, 2])]
    chaves = ["ano", "cbo4", "faixa", "uf"]
    soma = [c for c in ["adm", "des", "saldo", "adm_1emp", "adm_sup", "adm_base", "adm_aprendiz",
                        "sal_soma", "sal_log_soma", "sal_n"] if c in d]
    g = d.groupby(chaves, as_index=False)[soma].sum()
    g["t"] = g.ano
    g["tri"] = 1
    return g


def _dupla_sazonal(cfg, d, y, rotulo, extra=None):
    """Dupla diferença entre ocupações (mais × menos expostas) com referência no MESMO trimestre de 2022.

    y ~ Σ_{t ≥ 2023T1} β_t·Expo·1[t] + Σ_q γ_q·Expo·1[trimestre q] + controles da pandemia | cbo4 + uf:t
    γ_q absorve a sazonalidade própria das ocupações expostas (estimada com 2022); assim cada β_t compara o
    trimestre t com o mesmo trimestre de 2022 (T1 com T1, ..., T4 com T4)."""
    ref, ini, pos, _ = _per(cfg)
    d = d[(d.t >= ini) & (d[y] >= 0)].copy()
    d["t_pos"] = np.where(d.t >= pos, d.t, 0)
    ctl = ""
    if (d.choque_z != 0).any():
        ctl = f" + i(t, choque_z, ref={ref}) + i(t, boom_z, ref={ref})"
    fit = pf.fepois(f"{y} ~ i(t_pos, expo_v, ref=0) + i(tri, expo_v, ref=3){ctl} | cbo4 + uf:t",
                    data=d, vcov={"CRV1": "cbo4"})
    t = _tidy(fit, rotulo, y, extra)
    t = t[(t["var"] == "expo") & (t.periodo > 9999)]
    # pooled: média pós-ChatGPT contra o mesmo trimestre de 2022
    d["expo_post"] = d.expo_v * (d.t >= pos)
    ctl_a = ""
    if (d.choque_z != 0).any():
        ctl_a = " + i(ano, choque_z, ref=2022) + i(ano, boom_z, ref=2022)"
    fit2 = pf.fepois(f"{y} ~ expo_post + i(tri, expo_v, ref=3){ctl_a} | cbo4 + uf:t", data=d, vcov={"CRV1": "cbo4"})
    t2 = _tidy(fit2, rotulo + "_pooled", y, extra)
    return pd.concat([t, t2[t2.termo == "expo_post"]])


def por_faixa(cfg, df, y="adm"):
    res = []
    for faixa, d in df.groupby("faixa"):
        res.append(_dupla_sazonal(cfg, d, y, "dupla_por_faixa", {"faixa": faixa}))
    return pd.concat(res)


def aprendizes(cfg, painel):
    """Aprendizes: só entre jovens de 14–24 anos (idade legal), ocupações mais × menos expostas."""
    ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"))[["cbo4", "expo_z"]]
    d = painel[painel.faixa.isin(["ate17", "18_24"])]
    d = d.groupby(["t", "ano", "tri", "cbo4", "uf"], as_index=False).adm_aprendiz.sum()
    d = d.merge(ex, on="cbo4").rename(columns={"expo_z": "expo_v"})
    try:
        pand = pd.read_parquet(caminho(cfg, "processed", "pandemia_cbo4.parquet"))[["cbo4", "choque_z", "boom_z"]]
        d = d.merge(pand, on="cbo4", how="left")
    except FileNotFoundError:
        d["choque_z"], d["boom_z"] = np.nan, np.nan
    d[["choque_z", "boom_z"]] = d[["choque_z", "boom_z"]].fillna(0.0)
    d["uf"] = d.uf.astype(str)
    return _dupla_sazonal(cfg, d, "adm_aprendiz", "aprendizes_14_24", {"faixa": "14_24"})


def grafico_dupla(cfg, tab, titulo, arquivo):
    """Uma linha por faixa etária; 2022 = referência (zero) em cada trimestre."""
    t = tab[tab.periodo.notna()]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="0.5", lw=0.8)
    ts = sorted(t.periodo.unique())
    pos = {v: i for i, v in enumerate(ts)}
    grupos = list(t.groupby("faixa"))
    for faixa, g in grupos:
        g = g.sort_values("periodo")
        x = g.periodo.map(pos).to_numpy()
        rot = faixa.replace("_", "–").replace("50mais", "50+")
        if len(grupos) == 1:  # uma série: barras de erro
            ax.errorbar(x, g.coef, yerr=[g.coef - g.ic_inf, g.ic_sup - g.coef], fmt="o-", ms=4, lw=1.2,
                        capsize=2, color="#1f5fbf", label=rot)
        else:  # várias: faixa sombreada do IC 95%
            linha, = ax.plot(x, g.coef, marker="o", ms=3, lw=2.2 if faixa in ("18_24", "25_29") else 1.1, label=rot)
            ax.fill_between(x, g.ic_inf, g.ic_sup, color=linha.get_color(), alpha=0.10, lw=0)
    marc = [v for v in ts if v % 10 == 1]
    ax.set_xticks([pos[v] for v in marc], [f"{int(v) // 10}T1" for v in marc])
    ax.set_ylabel("coeficiente (log, PPML)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def salarios(cfg, df):
    ref, ini, _, chatgpt = _per(cfg)
    d = df[(df.sal_n > 0) & (df.t >= ini)].copy()
    d["lsal"] = d.sal_log_soma / d.sal_n
    ctl = _controles(d, ref)
    fit = pf.feols(f"lsal ~ i(t, treat, ref={ref}){ctl} | {FE}", data=d, weights="sal_n", vcov={"CRV1": "cbo4"})
    d2 = d[d.t != chatgpt]
    fit2 = pf.feols(f"lsal ~ treat_post{_controles_anuais(d2, ref)} | {FE}{SAZ}", data=d2, weights="sal_n", vcov={"CRV1": "cbo4"})
    a = _tidy(fit, "salario_event", "log_salario_real")
    b = _tidy(fit2, "salario_pooled", "log_salario_real")
    return pd.concat([a[a["var"] == "treat"], b[b.termo == "treat_post"]])


def _rotulo_t(t):
    t = int(t)
    return f"{t // 10}T{t % 10}" if t > 9999 else f"1S{t}"


def grafico_event(cfg, tab: pd.DataFrame, titulo: str, arquivo: str, ref=None):
    ref = ref or cfg["periodo"]["trimestre_referencia"]
    t = tab[tab.periodo.notna() & (tab.periodo != 0)][["periodo", "coef", "ic_inf", "ic_sup"]]
    t = pd.concat([t, pd.DataFrame([{"periodo": ref, "coef": 0.0, "ic_inf": 0.0, "ic_sup": 0.0}])])
    t = t.drop_duplicates("periodo").sort_values("periodo").reset_index(drop=True)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="0.5", lw=0.8)
    trimestral = t.periodo.max() > 9999
    if trimestral:
        a, b = cfg["periodo"]["pandemia"]
        dentro = x[(t.periodo >= a) & (t.periodo <= b)]
        if len(dentro):
            ax.axvspan(dentro.min() - 0.5, dentro.max() + 0.5, color="0.92", zorder=0)
            ax.text(dentro.mean(), 0, "pandemia", ha="center", va="bottom", fontsize=8, color="0.45")
        cg = x[t.periodo == cfg["periodo"]["trimestre_chatgpt"]]
    else:
        cg = x[t.periodo == cfg["periodo"]["base_2023"]["inicio"] // 10] - 0.5
    if len(cg):
        ax.axvline(cg[0], ls="--", color="0.55", lw=0.9)
        ax.text(cg[0] + 0.1, 1, "ChatGPT (nov/2022)", fontsize=8, color="0.4", va="top",
                transform=ax.get_xaxis_transform())
    ax.errorbar(x, t.coef, yerr=[t.coef - t.ic_inf, t.ic_sup - t.coef], fmt="o-", color="#1f5fbf", capsize=2,
                ms=4, lw=1.2)
    passo = 4 if trimestral and len(t) > 12 else 1
    ax.set_xticks(x[::passo], [_rotulo_t(v) for v in t.periodo[::passo]])
    ax.set_ylabel("coeficiente (log, PPML)")
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def rodar_principal(cfg):
    painel = pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet"))
    df = preparar(cfg, painel)
    tabs = []
    for y in DESFECHOS:
        if y not in df or df[y].sum() <= 0:
            log.warning("desfecho %s sem dados (código não encontrado?) — pulado", y)
            continue
        for controles in (True, False):
            for pooled in (False, True):
                rot = f"tripla_{'pooled' if pooled else 'event'}_{'com' if controles else 'sem'}_controles"
                try:
                    t = triple(cfg, df, y, controles=controles, pooled=pooled, rotulo=rot)
                except Exception as e:  # um modelo que falha não derruba os demais
                    log.warning("modelo %s / %s falhou: %s", y, rot, e)
                    continue
                if t is None:
                    continue
                tabs.append(t)
                if not pooled and controles:
                    grafico_event(cfg, t, f"Jovens 18–24 × exposição à IA: {DESFECHOS[y]}", f"event_{y}.png")
        log.info("modelo principal: %s ok", y)
    # tripla diferença com 25–29 como grupo tratado (recém-formados entram no mercado formal nessa idade)
    df2529 = preparar(cfg, painel, jovem_chave="jovem_25_29")
    for y in ("adm", "adm_sup", "adm_base", "des"):
        for pooled in (False, True):
            rot = f"tripla_25_29_{'pooled' if pooled else 'event'}_com_controles"
            try:
                t = triple(cfg, df2529, y, pooled=pooled, rotulo=rot)
            except Exception as e:
                log.warning("modelo 25–29 %s / %s falhou: %s", y, rot, e)
                continue
            if t is None:
                continue
            tabs.append(t)
            if not pooled:
                grafico_event(cfg, t, f"Jovens 25–29 × exposição à IA: {DESFECHOS[y]}", f"event_25_29_{y}.png")
        log.info("tripla 25–29: %s ok", y)
    # event study descritivo com toda a série (2020T1+), sem controles: mostra a pandemia no gráfico
    ini = int(cfg["periodo"]["inicio"].replace("-", "")[:4]) * 10 + 1
    try:
        t = triple(cfg, df, "adm", inicio=ini, controles=False, rotulo="tripla_event_serie_completa")
        tabs.append(t)
        grafico_event(cfg, t, "Admissões de jovens 18–24 × exposição à IA, série completa (sem controles)",
                      "event_adm_serie_completa.png")
    except Exception as e:
        log.warning("série completa falhou: %s", e)
    def _faixa():
        t = por_faixa(cfg, preparar(cfg, painel, jovem_chave="jovem_amplo"), "adm")
        grafico_dupla(cfg, t[t.modelo == "dupla_por_faixa"],
                      "Admissões por faixa etária × exposição (cada trimestre vs. o mesmo de 2022)",
                      "dupla_por_faixa.png")
        return t

    def _aprendiz():
        t = aprendizes(cfg, painel)
        ev = t[t.modelo == "aprendizes_14_24"]
        grafico_dupla(cfg, ev, "Aprendizes 14–24 × exposição (cada trimestre vs. o mesmo de 2022)",
                      "event_aprendizes_14_24.png")
        return t

    for nome, f in (("dupla por faixa", _faixa), ("aprendizes 14–24", _aprendiz), ("salários", lambda: salarios(cfg, df))):
        try:
            tabs.append(f())
            log.info("%s ok", nome)
        except Exception as e:
            log.warning("%s falhou: %s", nome, e)
    # grava o que saiu a cada etapa, mesmo que algo tenha falhado
    res = pd.concat(tabs)
    res.to_csv(caminho(cfg, "outputs", "tabelas", "modelos_principais.csv"), index=False)
    return res


GRUPOS = {"jovem": "18_24", "jovem_25_29": "25_29"}  # grupos tratados nas robustezes e na heterogeneidade


def rodar_heterogeneidade(cfg):
    het = pd.read_parquet(caminho(cfg, "processed", "painel_heterogeneidade.parquet"))
    res = []
    for chave, grupo in GRUPOS.items():
        for (dim, valor), d in het.groupby(["dim", "valor"]):
            df = preparar(cfg, d, jovem_chave=chave)
            for y in ("adm", "adm_sup", "adm_base"):
                try:
                    t = triple(cfg, df, y, efeitos=FE_SEM_UF, pooled=True)
                except Exception as e:  # subamostra pequena demais
                    log.warning("heterogeneidade %s %s=%s %s: %s", grupo, dim, valor, y, e)
                    continue
                if t is not None:
                    t["dim"], t["valor"], t["grupo"] = dim, valor, grupo
                    res.append(t)
            log.info("heterogeneidade %s: %s=%s ok", grupo, dim, valor)
    out = pd.concat(res)
    out.to_csv(caminho(cfg, "outputs", "tabelas", "heterogeneidade.csv"), index=False)
    return out


def rodar_robustez(cfg):
    from . import painel as P
    base_p = pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet"))
    b23 = cfg["periodo"]["base_2023"]
    ini_serie = int(cfg["periodo"]["inicio"].replace("-", "")[:4]) * 10 + 1
    so1s = so_primeiro_semestre(cfg, base_p)
    E = "cbo4:faixa + cbo4:t + faixa:uf:t"  # sem sazonalidade trimestral (só um período por ano)
    var = [  # (nome, painel, kwargs de preparar, kwargs de triple)
        ("principal", base_p, {}, {}),
        ("ate_2026T2", base_p[base_p.t <= 20262], {}, {}),   # sem o último trimestre (ajustes FOR/EXC incompletos)
        ("sem_controles_pandemia", base_p, {}, {"controles": False}),
        ("serie_completa_fe_fase", base_p, {}, {"inicio": ini_serie, "fase_fe": True}),
        # event study 2020T1+ sem controles: coeficientes de 2020–2022 = teste de pré-tendência longo
        ("serie_completa_event", base_p, {}, {"inicio": ini_serie, "controles": False}),
        ("incluir_2021_fe_fase", base_p, {}, {"inicio": 20211, "fase_fe": True}),
        ("base_2023", base_p, {"inicio_pos": b23["inicio_pos"]}, {"inicio": b23["inicio"], "ref": b23["referencia"]}),
        ("so_1o_semestre", so1s, {"inicio_pos": 2023}, {"inicio": 2022, "ref": 2022, "efeitos": E}),
        ("jovem_18_29", base_p, {"jovem_chave": "jovem_amplo"}, {}),
        ("expo_quartil_topo", base_p, {"expo": "expo_top"}, {}),
        ("expo_gradientes_3_4", base_p, {"expo": "expo_g34"}, {}),
        ("sem_cbo_imputadas", base_p, {"sem_imputados": True}, {}),
    ]
    for nome, filtros in (("sem_adm_publica_e_TI", dict(excluir_secoes=("O", "J"))),
                          ("sem_aprendiz_intermitente", dict(excluir_aprendiz=True, excluir_intermitente=True)),
                          ("sem_ajustes_so_MOV", dict(somente_mov=True))):
        var.append((nome, P.construir(cfg, nome=nome, **filtros), {}, {}))
    res = []
    for chave, grupo in GRUPOS.items():
        for nome, painel, kp, kt in var:
            kp = dict(kp)
            if "jovem_chave" in kp:
                if chave != "jovem":  # "jovem_18_29" só faz sentido como variante do grupo 18–24
                    continue
            else:
                kp["jovem_chave"] = chave
            df = preparar(cfg, painel, **kp)
            if nome == "so_1o_semestre":
                df["post"] = (df.t >= 2023).astype(int)
                df["treat_post"] = df.treat * df.post
                df["fase"] = "pos"
            for pooled in (True, False):
                try:
                    t = triple(cfg, df, "adm", pooled=pooled, **kt)
                except Exception as e:
                    log.warning("robustez %s %s (pooled=%s): %s", grupo, nome, pooled, e)
                    continue
                if t is not None:
                    t["variante"], t["grupo"] = nome, grupo
                    res.append(t)
                    if not pooled and nome in ("base_2023", "so_1o_semestre", "serie_completa_fe_fase",
                                               "serie_completa_event"):
                        rot = grupo.replace("_", "–")
                        grafico_event(cfg, t, f"Robustez ({rot}): {nome}", f"event_robustez_{grupo}_{nome}.png",
                                      ref=kt.get("ref"))
            log.info("robustez %s %s ok", grupo, nome)
    out = pd.concat(res)
    out.to_csv(caminho(cfg, "outputs", "tabelas", "robustez.csv"), index=False)
    return out


# ---------------------------------------------------------------------------------------------------------
# Pré-tendência: a queda pós-ChatGPT é maior do que a tendência que já vinha antes?
# ---------------------------------------------------------------------------------------------------------
def _idx(t):
    """20201 -> 0, 20202 -> 1, ... (trimestres corridos desde 2020T1)."""
    t = np.asarray(t, dtype=int)
    return (t // 10 - 2020) * 4 + (t % 10) - 1


def _tendencia_linear(cfg, df, ini):
    """(a) Tripla diferença 2020T1+ com tendência linear própria de Expo × Jovem.

    A tendência é identificada só no pré-ChatGPT (as dummies pós saturam 2023T1+), com nível próprio por
    ocupação × idade em cada fase (pré, pandemia, pós-pandemia). O efeito pós é o desvio da tendência anterior."""
    _, _, pos, chatgpt = _per(cfg)
    d = df[df.t >= ini].copy()
    d["treat_trend"] = d.treat * _idx(d.t)
    d["t_ev"] = np.where(d.t >= pos, d.t, 0)
    ef = FE + " + cbo4:faixa:fase"
    fit = pf.fepois(f"adm ~ i(t_ev, treat, ref=0) + treat_trend | {ef}", data=d, vcov={"CRV1": "cbo4"})
    ev = _tidy(fit, "tendencia_linear_event", "adm")
    d2 = d[d.t != chatgpt]
    fit2 = pf.fepois(f"adm ~ treat_post + treat_trend | {ef}{SAZ}", data=d2, vcov={"CRV1": "cbo4"})
    po = _tidy(fit2, "tendencia_linear_pooled", "adm")
    return pd.concat([ev[ev.termo.str.contains("treat")], po[po.termo.isin(["treat_post", "treat_trend"])]])


def _extrapolacao(cfg, df, ini, sem_pandemia):
    """(b) Extrapolação linear da pré-tendência (Rambachan & Roth, restrição de suavidade com M = 0).

    Ajusta uma reta aos coeficientes pré-ChatGPT do event study 2020T1+ e mede, com a matriz de covariância
    completa, quanto os coeficientes pós ficam abaixo dessa reta."""
    ref, _, pos, chatgpt = _per(cfg)
    d = df[df.t >= ini]
    fit = pf.fepois(f"adm ~ i(t, treat, ref={ref}) | {FE}", data=d, vcov={"CRV1": "cbo4"})
    nomes = list(fit._coefnames)
    V = np.asarray(fit._vcov)
    b = np.asarray(fit.coef().values)
    per = np.array([int(re.search(r"::(\d+)", n).group(1)) for n in nomes])
    a, z = cfg["periodo"]["pandemia"]
    pre_m = (per < chatgpt) & (~((per >= a) & (per <= z)) if sem_pandemia else True)
    pos_m = per >= pos
    # pontos do ajuste: coeficientes pré + a referência (= 0, sem variância)
    S = np.vstack([np.eye(len(b))[pre_m], np.zeros(len(b))])
    xp = np.r_[_idx(per[pre_m]), _idx([ref])[0]]
    X = np.c_[np.ones_like(xp), xp]
    H = np.linalg.solve(X.T @ X, X.T) @ S                     # δ = H b
    Xq = np.c_[np.ones(pos_m.sum()), _idx(per[pos_m])]
    A = np.eye(len(b))[pos_m] - Xq @ H                        # desvios = A b
    dev, Vd = A @ b, A @ V @ A.T
    w = np.full(len(dev), 1 / len(dev))
    media, ep = float(w @ dev), float(np.sqrt(w @ Vd @ w))
    from scipy import stats
    rot = "extrapolacao_sem_pandemia" if sem_pandemia else "extrapolacao_todos_pre"
    linhas = [dict(termo=f"desvio::{p}", coef=dv, ep=float(np.sqrt(Vd[i, i])), periodo=p, var="treat", modelo=rot,
                   desfecho="adm") for i, (p, dv) in enumerate(zip(per[pos_m], dev))]
    linhas.append(dict(termo="desvio_medio_pos", coef=media, ep=ep, p=float(2 * stats.norm.sf(abs(media / ep))),
                       ic_inf=media - 1.96 * ep, ic_sup=media + 1.96 * ep, var="treat", modelo=rot, desfecho="adm"))
    linhas.append(dict(termo="inclinacao_pre_por_trimestre", coef=float((H @ b)[1]), var="treat", modelo=rot,
                       desfecho="adm"))
    out = pd.DataFrame(linhas)
    out["ic_inf"] = out.ic_inf.fillna(out.coef - 1.96 * out.ep)
    out["ic_sup"] = out.ic_sup.fillna(out.coef + 1.96 * out.ep)
    # dados para o gráfico: coeficientes e reta extrapolada
    graf = pd.DataFrame({"periodo": np.r_[per, ref], "coef": np.r_[b, 0.0],
                         "ep": np.r_[np.sqrt(np.diag(V)), 0.0]}).sort_values("periodo")
    delta = H @ b
    graf["reta"] = delta[0] + delta[1] * _idx(graf.periodo)
    return out, graf


def _grafico_tendencia(cfg, graf, titulo, arquivo):
    g = graf.reset_index(drop=True)
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="0.5", lw=0.8)
    a, z = cfg["periodo"]["pandemia"]
    dentro = x[(g.periodo >= a) & (g.periodo <= z)]
    if len(dentro):
        ax.axvspan(dentro.min() - 0.5, dentro.max() + 0.5, color="0.92", zorder=0)
    cg = x[g.periodo == cfg["periodo"]["trimestre_chatgpt"]]
    if len(cg):
        ax.axvline(cg[0], ls="--", color="0.55", lw=0.9)
    ax.errorbar(x, g.coef, yerr=1.96 * g.ep, fmt="o", color="#1f5fbf", capsize=2, ms=4, label="coeficiente")
    ax.plot(x, g.reta, color="#d0612a", lw=1.5, label="tendência pré-ChatGPT extrapolada")
    marc = [i for i, v in enumerate(g.periodo) if int(v) % 10 == 1]
    ax.set_xticks(marc, [f"{int(g.periodo[i]) // 10}T1" for i in marc])
    ax.set_ylabel("coeficiente (log, PPML)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def rodar_pretendencia(cfg):
    painel = pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet"))
    ini = int(cfg["periodo"]["inicio"].replace("-", "")[:4]) * 10 + 1
    res = []
    for chave, grupo in GRUPOS.items():
        df = preparar(cfg, painel, jovem_chave=chave)
        tarefas = [("tendência linear", lambda: _tendencia_linear(cfg, df, ini))]
        for sem_p in (False, True):
            tarefas.append((f"extrapolação (sem pandemia={sem_p})", lambda s=sem_p: _extrapolacao(cfg, df, ini, s)))
        for nome, f in tarefas:
            try:
                r = f()
                if isinstance(r, tuple):
                    r, graf = r
                    if "sem_pandemia" in r.modelo.iloc[0]:
                        rot = grupo.replace("_", "–")
                        _grafico_tendencia(cfg, graf, f"Jovens {rot}: coeficientes vs. tendência pré-ChatGPT "
                                                      f"(sem trimestres da pandemia no ajuste)",
                                           f"pretendencia_{grupo}.png")
                r["grupo"] = grupo
                res.append(r)
                log.info("pré-tendência %s: %s ok", grupo, nome)
            except Exception as e:
                log.warning("pré-tendência %s: %s falhou: %s", grupo, nome, e)
    out = pd.concat(res)
    out.to_csv(caminho(cfg, "outputs", "tabelas", "pretendencia.csv"), index=False)
    return out


# ---------------------------------------------------------------------------------------------------------
# Aceleração (opção B): a inclinação da queda mudou depois do ChatGPT?
# ---------------------------------------------------------------------------------------------------------
# Modelo de quebra de tendência (piecewise linear) na tripla diferença, PPML:
#   adm ~ θ_pre·Treat·k + θ_mud·Treat·max(k, 0) | FE + sazonalidade [+ nível por fase]
#   k = trimestres desde 2022T4 (2022T4 = 0, 2023T1 = 1, ...)
#   θ_pre: inclinação que já existia antes do ChatGPT; θ_mud: MUDANÇA de inclinação depois.
#   Combinações reportadas (com a vcov completa): inclinação pós = θ_pre + θ_mud;
#   efeito acumulado no último trimestre = h·θ_mud; efeito médio pós = (h+1)/2·θ_mud (h = trimestres pós).
# Ressalva para o artigo: teste escolhido DEPOIS de ver os dados (efeito gradual) → complementar, não principal.
VARIANTES_ACEL = {
    "fase_sem_controles": "2020T1+, nível próprio por fase (pré/pandemia/pós)",
    "fase_com_controles": "2020T1+, nível por fase + controles da pandemia por ano",
    "sem_pandemia": "2020T1 + 2022T1+, sem os trimestres da pandemia",
    "so_2022": "2022T1+ (pré-tendência só com 2022T1–T4)",
}


def _comb(nome, c, b, V, modelo, n, extra):
    est, ep = float(c @ b), float(np.sqrt(c @ V @ c))
    from scipy import stats
    return dict(termo=nome, coef=est, ep=ep, p=float(2 * stats.norm.sf(abs(est / ep))) if ep > 0 else np.nan,
                ic_inf=est - 1.96 * ep, ic_sup=est + 1.96 * ep, var="treat", modelo=modelo, desfecho="adm",
                n_obs=n, **extra)


def _quebra_inclinacao(cfg, df, ini, variante):
    ref, ini_principal, _, chatgpt = _per(cfg)
    a, z = cfg["periodo"]["pandemia"]
    d = df[df.t >= ini].copy()
    ef, ctrl = FE + SAZ, ""
    if variante.startswith("fase"):
        ef += " + cbo4:faixa:fase"
        if variante == "fase_com_controles":
            # Controles da pandemia só para os anos DEPOIS da pandemia (2023+ vs. 2022). Dentro da pandemia o
            # nível por fase já faz esse papel; i(ano, ...) para 2020–2021 repetiria o efeito fixo de fase
            # (matriz singular em 25–29). O que importa aqui é separar o fim do boom de 2022 do efeito da IA.
            if (d.chq_j != 0).any() or (d.boom_j != 0).any():
                ano_ref = int(ref) // 10
                termos = []
                for y in sorted(int(v) for v in d.ano.unique() if int(v) > ano_ref):
                    for c in ("chq_j", "boom_j"):
                        d[f"{c}_{y}"] = d[c] * (d.ano == y)
                        termos.append(f"{c}_{y}")
                ctrl = "".join(f" + {t}" for t in termos)
    elif variante == "sem_pandemia":
        d = d[~d.t.between(a, z)]
    elif variante == "so_2022":
        d = d[d.t >= ini_principal]
    k = _idx(d.t) - _idx([chatgpt])[0]
    d["treat_trend"] = d.treat * k
    d["treat_trend_pos"] = d.treat * np.maximum(k, 0)
    fit = pf.fepois(f"adm ~ treat_trend + treat_trend_pos{ctrl} | {ef}", data=d, vcov={"CRV1": "cbo4"})
    nomes = list(fit._coefnames)
    b, V = np.asarray(fit.coef().values), np.asarray(fit._vcov)
    e1, e2 = np.zeros(len(b)), np.zeros(len(b))
    e1[nomes.index("treat_trend")] = 1
    e2[nomes.index("treat_trend_pos")] = 1
    h = int(k.max())
    mod = f"aceleracao_{variante}"
    extra = dict(trimestres_pos=h, trimestres_pre=int(len(np.unique(k[k < 0]))))
    linhas = [
        _comb("inclinacao_pre", e1, b, V, mod, fit._N, extra),
        _comb("mudanca_inclinacao", e2, b, V, mod, fit._N, extra),
        _comb("inclinacao_pos", e1 + e2, b, V, mod, fit._N, extra),
        _comb("efeito_acumulado_ultimo_tri", h * e2, b, V, mod, fit._N, extra),
        _comb("efeito_medio_pos", (h + 1) / 2 * e2, b, V, mod, fit._N, extra),
    ]
    return pd.DataFrame(linhas), (float(b @ e1), float(b @ e2))


def _event_coefs(cfg, df, ini):
    """Coeficientes do event study 2020T1+ (ref. 2022T3), só para o gráfico."""
    ref = cfg["periodo"]["trimestre_referencia"]
    fit = pf.fepois(f"adm ~ i(t, treat, ref={ref}) | {FE}", data=df[df.t >= ini], vcov={"CRV1": "cbo4"})
    per = [int(re.search(r"::(\d+)", n).group(1)) for n in fit._coefnames]
    return pd.DataFrame({"periodo": per + [ref], "coef": list(fit.coef().values) + [0.0],
                         "ep": list(fit.se().values) + [0.0]}).sort_values("periodo").reset_index(drop=True)


def _grafico_aceleracao(cfg, g, inclinacoes, titulo, arquivo):
    ref, chatgpt = cfg["periodo"]["trimestre_referencia"], cfg["periodo"]["trimestre_chatgpt"]
    th_pre, th_mud = inclinacoes
    k = _idx(g.periodo) - _idx([chatgpt])[0]
    k_ref = _idx([ref])[0] - _idx([chatgpt])[0]
    reta = th_pre * (k - k_ref) + th_mud * np.maximum(k, 0)       # ancorada em 0 na referência
    contraf = th_pre * (k - k_ref)                                # se a inclinação anterior continuasse
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="0.5", lw=0.8)
    a, z = cfg["periodo"]["pandemia"]
    dentro = x[(g.periodo >= a) & (g.periodo <= z)]
    if len(dentro):
        ax.axvspan(dentro.min() - 0.5, dentro.max() + 0.5, color="0.92", zorder=0)
    cg = x[g.periodo == chatgpt]
    if len(cg):
        ax.axvline(cg[0], ls="--", color="0.55", lw=0.9)
    ax.errorbar(x, g.coef, yerr=1.96 * g.ep, fmt="o", color="#1f5fbf", capsize=2, ms=4, label="coeficiente (event study)")
    pos = k >= 0
    ax.plot(x, reta, color="#d0612a", lw=1.6, label="tendência com quebra em 2022T4")
    ax.plot(x[pos], contraf[pos], color="#d0612a", lw=1.2, ls=":", label="se a inclinação anterior continuasse")
    marc = [i for i, v in enumerate(g.periodo) if int(v) % 10 == 1]
    ax.set_xticks(marc, [f"{int(g.periodo[i]) // 10}T1" for i in marc])
    ax.set_ylabel("coeficiente (log, PPML)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def rodar_aceleracao(cfg, variantes=None):
    """Opção B do teste de pré-tendência: mudança de inclinação depois do ChatGPT, 4 variantes × 2 grupos.

    variantes: lista com parte das VARIANTES_ACEL para rodar só essas; as linhas delas são substituídas
    no aceleracao.csv existente e as demais ficam como estavam."""
    painel = pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet"))
    ini = int(cfg["periodo"]["inicio"].replace("-", "")[:4]) * 10 + 1
    escolhidas = {v: VARIANTES_ACEL[v] for v in (variantes or VARIANTES_ACEL)}
    res = []
    for chave, grupo in GRUPOS.items():
        df = preparar(cfg, painel, jovem_chave=chave)
        incl_graf = None
        for var, desc in escolhidas.items():
            try:
                r, incl = _quebra_inclinacao(cfg, df, ini, var)
                r["grupo"], r["descricao"] = grupo, desc
                res.append(r)
                if var == "sem_pandemia":
                    incl_graf = incl
                m = r.set_index("termo")
                log.info("aceleração %s / %s: inclinação pré %.4f | mudança %.4f (p=%.3f) | efeito médio pós %.3f",
                         grupo, var, m.coef["inclinacao_pre"], m.coef["mudanca_inclinacao"],
                         m.p["mudanca_inclinacao"], m.coef["efeito_medio_pos"])
            except Exception as e:
                log.warning("aceleração %s / %s falhou: %s", grupo, var, e)
        if incl_graf is not None:
            try:
                rot = grupo.replace("_", "–")
                _grafico_aceleracao(cfg, _event_coefs(cfg, df, ini), incl_graf,
                                    f"Jovens {rot}: a queda acelerou depois do ChatGPT? (quebra de tendência, "
                                    f"sem trimestres da pandemia)", f"aceleracao_{grupo}.png")
            except Exception as e:
                log.warning("gráfico de aceleração %s falhou: %s", grupo, e)
    out = pd.concat(res)
    arq = caminho(cfg, "outputs", "tabelas", "aceleracao.csv")
    if variantes and arq.exists():  # rodada parcial: troca só as linhas das variantes refeitas
        antigo = pd.read_csv(arq)
        antigo = antigo[~antigo.modelo.isin([f"aceleracao_{v}" for v in escolhidas])]
        out = pd.concat([antigo, out], ignore_index=True)
    out.to_csv(arq, index=False)
    return out


# ---------------------------------------------------------------------------------------------------------
# Teste formal das diferenças entre grupos (heterogeneidade)
# ---------------------------------------------------------------------------------------------------------
# Modelo empilhado e totalmente interagido: os dois (ou mais) grupos de uma dimensão entram juntos, cada um
# com seus próprios efeitos fixos, controles da pandemia e coeficiente treat_post. As estimativas de cada
# grupo são as MESMAS do rodar_heterogeneidade (que roda um modelo por grupo), mas aqui a covariância entre
# elas é conhecida (mesmas ocupações → erros correlacionados, cluster por cbo4), então dá para testar a
# diferença. Regiões (5 níveis): teste de Wald conjunto de que todos os efeitos são iguais.
# p_holm: correção de Holm para comparações múltiplas, dentro de cada grupo etário × desfecho.
TESTES_GRUPOS = {
    "sexo": ["homem", "mulher"],
    "raca": ["branca", "negra"],
    "instrucao": ["medio", "superior"],
    "nordeste": ["resto", "NE"],
    "rn": ["resto", "RN"],
    "regiao": ["1", "2", "3", "4", "5"],  # Norte, Nordeste, Sudeste, Sul, Centro-Oeste (fora: não identificada)
}
DESFECHOS_GRUPOS = ("adm", "adm_sup", "adm_base")


def _holm(p):
    p = np.asarray(p, dtype=float)
    ordem = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    acum = 0.0
    for i, j in enumerate(ordem):
        acum = max(acum, (m - i) * p[j])
        adj[j] = min(1.0, acum)
    return adj


def _teste_diferenca(cfg, d, y, niveis):
    from scipy import stats
    ref, ini, _, chatgpt = _per(cfg)
    d = d[(d.t >= ini) & (d.t != chatgpt) & d[y].notna() & (d[y] >= 0) & d.valor.isin(niveis)].copy()
    tem_ctrl = (d.chq_j != 0).any() or (d.boom_j != 0).any()
    ano_ref = int(ref) // 10
    termos, ctrl = [], []
    for k, v in enumerate(niveis):
        m = (d.valor == v).astype(float)
        d[f"tp_{k}"] = d.treat_post * m
        termos.append(f"tp_{k}")
        if tem_ctrl:
            d[f"cq_{k}"], d[f"bm_{k}"] = d.chq_j * m, d.boom_j * m
            ctrl += [f"i(ano, cq_{k}, ref={ano_ref})", f"i(ano, bm_{k}, ref={ano_ref})"]
    ef = "cbo4:faixa:valor + cbo4:t:valor + faixa:t:valor + cbo4:faixa:tri:valor"
    fit = pf.fepois(f"{y} ~ {' + '.join(termos + ctrl)} | {ef}", data=d, vcov={"CRV1": "cbo4"})
    nomes = list(fit._coefnames)
    b, V = np.asarray(fit.coef().values), np.asarray(fit._vcov)
    sobra = [(t, v) for t, v in zip(termos, niveis) if t in nomes]  # grupo sem dados/colinear sai do teste
    if len(sobra) < len(termos):
        log.warning("diferenças %s: grupos sem estimativa removidos: %s", y,
                    [v for t, v in zip(termos, niveis) if t not in nomes])
    termos, niveis = [t for t, _ in sobra], [v for _, v in sobra]
    idx = [nomes.index(t) for t in termos]
    linhas = []
    for k, v in enumerate(niveis):  # efeito de cada grupo
        c = np.zeros(len(b)); c[idx[k]] = 1
        linhas.append(_comb(f"efeito::{v}", c, b, V, "diferenca_grupos", fit._N, {}))
    for k in range(1, len(niveis)):  # diferença de cada grupo contra o primeiro (base)
        c = np.zeros(len(b)); c[idx[k]] = 1; c[idx[0]] = -1
        linhas.append(_comb(f"diferenca::{niveis[k]}-{niveis[0]}", c, b, V, "diferenca_grupos", fit._N, {}))
    if len(niveis) > 2:  # Wald conjunto: todos iguais
        R = np.zeros((len(niveis) - 1, len(b)))
        for k in range(1, len(niveis)):
            R[k - 1, idx[k]], R[k - 1, idx[0]] = 1, -1
        Rb = R @ b
        W = float(Rb @ np.linalg.solve(R @ V @ R.T, Rb))
        linhas.append(dict(termo="wald_todos_iguais", coef=W, p=float(stats.chi2.sf(W, len(niveis) - 1)),
                           var="treat", modelo="diferenca_grupos", desfecho="adm", n_obs=fit._N))
    out = pd.DataFrame(linhas)
    out["desfecho"] = y
    return out


def _grafico_diferencas(cfg, tab, grupo, arquivo):
    t = tab[(tab.grupo == grupo) & tab.termo.str.startswith("efeito::")].copy()
    if t.empty:
        return
    desfs = [y for y in DESFECHOS_GRUPOS if (t.desfecho == y).any()]
    fig, axes = plt.subplots(1, len(desfs), figsize=(4.2 * len(desfs), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    rot_y = {"adm": "Admissões (todas)", "adm_sup": "Admissões com superior", "adm_base": "Admissões na base salarial"}
    ordem = [(dim, v) for dim, nv in TESTES_GRUPOS.items()
             for v in (nv or sorted(t[t.dim == dim].termo.str.replace("efeito::", "").unique()))]
    pos = {k: i for i, k in enumerate(ordem)}
    for ax, y in zip(axes, desfs):
        s = t[t.desfecho == y].copy()
        s["v"] = s.termo.str.replace("efeito::", "")
        s["y"] = [pos.get((a, b), np.nan) for a, b in zip(s.dim, s.v)]
        s = s.dropna(subset=["y"])
        ax.axvline(0, color="0.5", lw=0.8)
        ax.errorbar(100 * s.coef, s.y, xerr=196 * s.ep, fmt="o", color="#1f5fbf", capsize=2, ms=4)
        dif = tab[(tab.grupo == grupo) & (tab.desfecho == y) & tab.termo.str.match(r"diferenca::|wald_")]
        for dim in s.dim.unique():
            dd = dif[dif.dim == dim]
            if len(dd):
                p = dd.p_holm.min() if dim != "regiao" else dd[dd.termo == "wald_todos_iguais"].p_holm.min()
                yy = s[s.dim == dim].y.max()
                ax.annotate(f"p dif.={p:.3f}", (ax.get_xlim()[1], yy), fontsize=7, ha="right", va="center",
                            color="#d0612a" if p < 0.05 else "0.4")
        ax.set_title(rot_y.get(y, y), fontsize=9)
        ax.set_xlabel("efeito (%)")
    axes[0].set_yticks(range(len(ordem)), [f"{a}: {b}" for a, b in ordem], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"Jovens {grupo.replace('_', '–')}: efeito por grupo e teste da diferença (p ajustado por Holm)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def rodar_diferencas(cfg, grupos=None, dims=None):
    """Teste formal das diferenças entre grupos. grupos: subconjunto de GRUPOS (ex.: ['jovem_25_29']);
    dims: subconjunto de TESTES_GRUPOS. Sem argumentos roda tudo."""
    het = pd.read_parquet(caminho(cfg, "processed", "painel_heterogeneidade.parquet"))
    res = []
    for chave, grupo in GRUPOS.items():
        if grupos and chave not in grupos and grupo not in grupos:
            continue
        for dim, niveis in TESTES_GRUPOS.items():
            if dims and dim not in dims:
                continue
            h = het[het.dim == dim]
            presentes = set(h.valor.astype(str).str.replace(r"\.0$", "", regex=True).unique())
            nv = [v for v in niveis if v in presentes] if niveis else sorted(presentes)
            df = preparar(cfg, h, jovem_chave=chave)
            df["valor"] = df.valor.astype(str).str.replace(r"\.0$", "", regex=True)
            for y in DESFECHOS_GRUPOS:
                if dim == "instrucao" and y == "adm_sup":
                    continue  # "superior" é a própria divisão
                try:
                    r = _teste_diferenca(cfg, df, y, nv)
                except Exception as e:
                    log.warning("diferenças %s %s %s falhou: %s", grupo, dim, y, e)
                    continue
                r["dim"], r["grupo"] = dim, grupo
                res.append(r)
                d = r[r.termo.str.match(r"diferenca::|wald_")]
                log.info("diferenças %s %s %s: %s", grupo, dim, y,
                         "; ".join(f"{a} {c:+.4f} (p={p:.3f})" for a, c, p in zip(d.termo, d.coef, d.p)))
    out = pd.concat(res, ignore_index=True)
    arq = caminho(cfg, "outputs", "tabelas", "diferencas_grupos.csv")
    if (grupos or dims) and arq.exists():  # rodada parcial: troca só as linhas refeitas
        antigo = pd.read_csv(arq)
        chaves = set(zip(out.grupo, out.dim))
        antigo = antigo[[(g, d) not in chaves for g, d in zip(antigo.grupo, antigo.dim)]]
        out = pd.concat([antigo, out], ignore_index=True)
    # Holm sobre a tabela completa (inclusive linhas de rodadas anteriores), por grupo etário × desfecho
    teste = out.termo.str.match(r"diferenca::|wald_")
    out["p_holm"] = np.nan
    for _, idx in out[teste].groupby(["grupo", "desfecho"]).groups.items():
        out.loc[idx, "p_holm"] = _holm(out.loc[idx, "p"])
    out.to_csv(arq, index=False)
    for g in out.grupo.unique():
        try:
            _grafico_diferencas(cfg, out, g, f"diferencas_grupos_{g}.png")
        except Exception as e:
            log.warning("gráfico de diferenças %s falhou: %s", g, e)
    return out
