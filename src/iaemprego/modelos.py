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
    "adm_base": "Admissões na base salarial (≤ p25 da família)",
    "adm_aprendiz": "Admissões de aprendizes",
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
    jov, adu = cfg["idade"][jovem_chave], cfg["idade"]["adulto"]
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


def por_faixa(cfg, df, y="adm"):
    ref, ini, _, _ = _per(cfg)
    res = []
    for faixa, d in df[df.t >= ini].groupby("faixa"):
        d = d[d[y] >= 0]
        fit = pf.fepois(f"{y} ~ i(t, expo_v, ref={ref}) + i(t, choque_z, ref={ref}) + i(t, boom_z, ref={ref})"
                        f" | cbo4 + uf:t", data=d, vcov={"CRV1": "cbo4"})
        t = _tidy(fit, "dupla_por_faixa", y, {"faixa": faixa})
        res.append(t[t["var"] == "expo"])
    return pd.concat(res)


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
    # event study descritivo com toda a série (2020T1+), sem controles: mostra a pandemia no gráfico
    ini = int(cfg["periodo"]["inicio"].replace("-", "")[:4]) * 10 + 1
    try:
        t = triple(cfg, df, "adm", inicio=ini, controles=False, rotulo="tripla_event_serie_completa")
        tabs.append(t)
        grafico_event(cfg, t, "Admissões de jovens 18–24 × exposição à IA, série completa (sem controles)",
                      "event_adm_serie_completa.png")
    except Exception as e:
        log.warning("série completa falhou: %s", e)
    for nome, f in (("dupla por faixa", lambda: por_faixa(cfg, preparar(cfg, painel, jovem_chave="jovem_amplo"), "adm")),
                    ("salários", lambda: salarios(cfg, df))):
        try:
            tabs.append(f())
            log.info("%s ok", nome)
        except Exception as e:
            log.warning("%s falhou: %s", nome, e)
    # grava o que saiu a cada etapa, mesmo que algo tenha falhado
    res = pd.concat(tabs)
    res.to_csv(caminho(cfg, "outputs", "tabelas", "modelos_principais.csv"), index=False)
    return res


def rodar_heterogeneidade(cfg):
    het = pd.read_parquet(caminho(cfg, "processed", "painel_heterogeneidade.parquet"))
    res = []
    for (dim, valor), d in het.groupby(["dim", "valor"]):
        df = preparar(cfg, d)
        for y in ("adm", "adm_sup", "adm_base"):
            try:
                t = triple(cfg, df, y, efeitos=FE_SEM_UF, pooled=True)
            except Exception as e:  # subamostra pequena demais
                log.warning("heterogeneidade %s=%s %s: %s", dim, valor, y, e)
                continue
            if t is not None:
                t["dim"], t["valor"] = dim, valor
                res.append(t)
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
    for nome, painel, kp, kt in var:
        df = preparar(cfg, painel, **kp)
        if nome == "so_1o_semestre":
            df["post"] = (df.t >= 2023).astype(int)
            df["treat_post"] = df.treat * df.post
            df["fase"] = "pos"
        for pooled in (True, False):
            try:
                t = triple(cfg, df, "adm", pooled=pooled, **kt)
            except Exception as e:
                log.warning("robustez %s (pooled=%s): %s", nome, pooled, e)
                continue
            if t is not None:
                t["variante"] = nome
                res.append(t)
                if not pooled and nome in ("base_2023", "so_1o_semestre", "serie_completa_fe_fase"):
                    grafico_event(cfg, t, f"Robustez: {nome}", f"event_robustez_{nome}.png", ref=kt.get("ref"))
        log.info("robustez %s ok", nome)
    out = pd.concat(res)
    out.to_csv(caminho(cfg, "outputs", "tabelas", "robustez.csv"), index=False)
    return out
