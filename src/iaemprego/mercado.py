"""Etapa 10 — O mercado como um todo: as ocupações mais expostas à IA contrataram menos (todas as idades)?

Diferença dupla entre ocupações (mais × menos expostas), TODAS as idades juntas. Complementa a tripla diferença
dos jovens: lá os choques de cada ocupação se cancelam (jovem − adulto na mesma ocupação); aqui não, então o
resultado é mais exposto a choques setoriais e entra como CONTEXTO, com controles e teste de pré-tendência.

Referência = o MESMO trimestre de 2022 (T1 com T1, ..., T4 com T4), o que separa o efeito da sazonalidade
própria de cada ocupação (efeito fixo ocupação × UF × trimestre do ano).

CAGED (célula família CBO × UF × trimestre), PPML:
  y ~ Σ_{t ∉ 2022} β_t·Expo·1[t] + controles da pandemia por trimestre | cbo4:uf:tri + uf:t
  pooled (2022T1+): y ~ Expo·Pós + controles por ano | cbo4:uf:tri + uf:t
  y ∈ {admissões, desligamentos, adm. com superior, adm. na base salarial}; log do salário de admissão (MQO).
  Controles: choque e boom da pandemia medidos na OCUPAÇÃO (todas as idades): variação do log das admissões
  2020T1→2020T2 e 2020T1→2022T1, padronizados com peso nas admissões de 2020T1.
  Robustezes: sem controles; efeito fixo grande grupo CBO (1º dígito) × UF × trimestre (compara ocupações
  do mesmo grande grupo).
Demanda por qualificação: parcela de admitidos com superior (e na base salarial) por ocupação × trimestre,
  MQO ponderado pelas admissões, efeito fixo ocupação × trimestre do ano + trimestre.
PNAD (célula ocupação COD × trimestre, todas as idades): ocupados, formais, informais, recém-entrados.
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

DESF_CAGED = {"adm": "Admissões", "des": "Desligamentos", "adm_sup": "Admissões com superior",
              "adm_base": "Admissões na base salarial"}
DESF_PNAD = {"ocup": "Ocupados", "formal": "Ocupados formais", "informal": "Ocupados informais",
             "recente": "Recém-entrados (< 1 ano)"}


# --------------------------------------------------------------------------------------------- utilidades
def _dummies(d, var, prefixo, ts, refs):
    nomes = []
    for t in ts:
        if t in refs:
            continue
        n = f"{prefixo}_{t}"
        d[n] = d[var] * (d.t == t)
        nomes.append(n)
    return nomes


def _tab(fit, modelo, desfecho, prefixo="e_", extra=None):
    t = fit.tidy().reset_index()
    t.columns = ["termo", "coef", "ep", "t", "p", "ic_inf", "ic_sup"][: len(t.columns)]
    t = t[t.termo.str.startswith(prefixo) | (t.termo == "expo_post")].copy()
    t["periodo"] = t.termo.map(lambda s: int(m.group(1)) if (m := re.search(r"_(\d{5})$", s)) else np.nan)
    t["modelo"], t["desfecho"], t["n_obs"] = modelo, desfecho, fit._N
    for k, v in (extra or {}).items():
        t[k] = v
    return t


def _choques(cfg, nac: pd.DataFrame) -> pd.DataFrame:
    """Choque e boom da pandemia por ocupação, todas as idades (aproximação trimestral das janelas mensais)."""
    piv = nac.pivot_table(index="cbo4", columns="t", values="adm", aggfunc="sum")
    if not {20201, 20202, 20221} <= set(piv.columns):
        log.warning("mercado: sem 2020T1/2020T2/2022T1, sem controles da pandemia")
        return pd.DataFrame(columns=["cbo4", "chq_m", "boom_m"])
    lg = lambda x: np.log(x.clip(lower=0).fillna(0) + 1)
    d = pd.DataFrame({"chq": lg(piv[20202]) - lg(piv[20201]), "boom": lg(piv[20221]) - lg(piv[20201]),
                      "w": piv[20201].fillna(0)})
    d = d[d.w >= 20]
    for v in ("chq", "boom"):
        mu = np.average(d[v], weights=d.w)
        sd = np.sqrt(np.average((d[v] - mu) ** 2, weights=d.w))
        d[f"{v}_m"] = (d[v] - mu) / sd
    return d[["chq_m", "boom_m"]].reset_index()


def _shift_share(cfg) -> pd.DataFrame:
    """Ciclo setorial previsto por ocupação (shift-share, leave-one-out), a partir dos microdados do CAGED.

    ss_y[c,t] = Σ_s w[c,s] · log(1 + Y[s,t] − Y[c,s,t]),  w[c,s] = parcela das admissões de 2022 da ocupação c
    no setor s (seção CNAE). Mede quanto a ocupação contrataria se acompanhasse o ciclo dos setores em que
    costuma trabalhar, sem usar a própria ocupação. Entra como controle: absorve o boom e a normalização
    setorial (TI, finanças, serviços às empresas) que se confundem com a exposição à IA."""
    from .caged import conexao
    from .util import intervalo_comp
    arq = caminho(cfg, "processed", "shift_share_cbo4.parquet")
    if arq.exists():
        return pd.read_parquet(arq)
    con = conexao(cfg)
    ini, fim = intervalo_comp(cfg, con)
    sup = ",".join(str(int(x)) for x in cfg["codigos"]["superior"])
    m = con.sql(f"""
        SELECT substr(cbo6, 1, 4) AS cbo4, coalesce(nullif(trim(secao), ''), '?') AS secao,
               (comp // 100) * 10 + ((comp % 100) - 1) // 3 + 1 AS t,
               sum(CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END) AS adm,
               sum(CASE WHEN saldo_bruto = -1 THEN sinal ELSE 0 END) AS des,
               sum(CASE WHEN saldo_bruto = 1 AND instr IN ({sup}) THEN sinal ELSE 0 END) AS adm_sup
        FROM mov_ajustado WHERE comp BETWEEN {ini} AND {fim} AND idade >= 14
        GROUP BY ALL""").df()
    con.close()
    w = m[m.t // 10 == 2022].groupby(["cbo4", "secao"]).adm.sum()
    w = (w / w.groupby(level=0).transform("sum")).rename("w").reset_index()
    w = w[w.w > 0]
    out = None
    for y in ("adm", "des", "adm_sup"):
        setor = m.groupby(["secao", "t"])[y].sum().rename("S").reset_index()
        x = w.merge(setor, on="secao").merge(m[["cbo4", "secao", "t", y]], on=["cbo4", "secao", "t"], how="left")
        x["loo"] = (x.S - x[y].fillna(0)).clip(lower=0)
        x["v"] = x.w * np.log1p(x.loo)
        r = x.groupby(["cbo4", "t"]).v.sum().rename(f"ss_{y}").reset_index()
        out = r if out is None else out.merge(r, on=["cbo4", "t"], how="outer")
    out.to_parquet(arq, index=False)
    log.info("shift-share: %d ocupações × %d trimestres", out.cbo4.nunique(), out.t.nunique())
    return out


# --------------------------------------------------------------------------------------------- CAGED
def _caged(cfg):
    p = pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet"))
    cols = [c for c in ["adm", "des", "adm_sup", "adm_base", "sal_n", "sal_log_soma"] if c in p]
    d = p.groupby(["t", "ano", "tri", "cbo4", "uf"], as_index=False)[cols].sum()
    ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"))[["cbo4", "expo_z"]]
    d = d.merge(ex, on="cbo4", how="inner")
    ch = _choques(cfg, d.groupby(["cbo4", "t"], as_index=False).adm.sum())
    d = d.merge(ch, on="cbo4", how="left")
    d[["chq_m", "boom_m"]] = d[["chq_m", "boom_m"]].fillna(0.0)
    try:
        ss = _shift_share(cfg)
        d = d.merge(ss, on=["cbo4", "t"], how="left")
        for c in ("ss_adm", "ss_des", "ss_adm_sup"):
            d[c] = d[c].fillna(d[c].mean())
    except Exception as e:
        log.warning("mercado: shift-share indisponível (%s)", e)
    d["uf"] = d.uf.astype(str)
    d["cbo1"] = d.cbo4.str[:1]
    d["expo_post"] = d.expo_z * (d.t >= cfg["periodo"]["inicio_pos"])
    return d


def _modelos_caged(cfg, d):
    ini = cfg["periodo"]["inicio_amostra_principal"]
    tem_ctl = (d.chq_m != 0).any()
    res = []
    ts = sorted(d.t.unique())
    refs = [t for t in ts if t // 10 == 2022]
    ev = d.copy()
    e_cols = _dummies(ev, "expo_z", "e", ts, refs)
    c_cols = (_dummies(ev, "chq_m", "c", ts, refs) + _dummies(ev, "boom_m", "b", ts, refs)) if tem_ctl else []
    pooled = d[d.t >= ini]
    ctl_a = " + i(ano, chq_m, ref=2022) + i(ano, boom_m, ref=2022)" if tem_ctl else ""
    for y in DESF_CAGED:
        if y not in d or d[y].sum() <= 0:
            continue
        tarefas = [
            ("mercado_event", lambda: pf.fepois(f"{y} ~ {' + '.join(e_cols + c_cols)} | cbo4:uf:tri + uf:t",
                                                data=ev[ev[y] >= 0], vcov={"CRV1": "cbo4"})),
            ("mercado_pooled", lambda: pf.fepois(f"{y} ~ expo_post{ctl_a} | cbo4:uf:tri + uf:t",
                                                 data=pooled[pooled[y] >= 0], vcov={"CRV1": "cbo4"})),
            ("mercado_pooled_sem_controles", lambda: pf.fepois(f"{y} ~ expo_post | cbo4:uf:tri + uf:t",
                                                               data=pooled[pooled[y] >= 0], vcov={"CRV1": "cbo4"})),
            ("mercado_pooled_grande_grupo", lambda: pf.fepois(f"{y} ~ expo_post{ctl_a} | cbo4:uf:tri + cbo1:uf:t",
                                                              data=pooled[pooled[y] >= 0], vcov={"CRV1": "cbo4"})),
        ]
        ssv = {"adm": "ss_adm", "des": "ss_des", "adm_sup": "ss_adm_sup", "adm_base": "ss_adm"}.get(y)
        if ssv in d:
            tarefas += [
                ("mercado_event_shift_share", lambda: pf.fepois(
                    f"{y} ~ {' + '.join(e_cols + c_cols)} + {ssv} | cbo4:uf:tri + uf:t",
                    data=ev[ev[y] >= 0], vcov={"CRV1": "cbo4"})),
                ("mercado_pooled_shift_share", lambda: pf.fepois(
                    f"{y} ~ expo_post{ctl_a} + {ssv} | cbo4:uf:tri + uf:t",
                    data=pooled[pooled[y] >= 0], vcov={"CRV1": "cbo4"})),
            ]
        for nome, f in tarefas:
            try:
                t = _tab(f(), nome, y, extra={"fonte": "CAGED"})
                res.append(t)
                if "pooled" in nome:
                    r = t.iloc[0]
                    log.info("mercado CAGED %-9s %-30s %+.4f (p=%.3f)", y, nome, r.coef, r.p)
            except Exception as e:
                log.warning("mercado CAGED %s %s falhou: %s", y, nome, e)
    # salário de admissão (log, MQO ponderado pelo nº de salários)
    if "sal_n" in d:
        s = d[d.sal_n > 0].copy()
        s["lsal"] = s.sal_log_soma / s.sal_n
        ev_s = s.copy()
        e_s = _dummies(ev_s, "expo_z", "e", ts, refs)
        c_s = (_dummies(ev_s, "chq_m", "c", ts, refs) + _dummies(ev_s, "boom_m", "b", ts, refs)) if tem_ctl else []
        for nome, f in (
            ("mercado_event", lambda: pf.feols(f"lsal ~ {' + '.join(e_s + c_s)} | cbo4:uf:tri + uf:t", data=ev_s,
                                               weights="sal_n", vcov={"CRV1": "cbo4"})),
            ("mercado_pooled", lambda: pf.feols(f"lsal ~ expo_post{ctl_a} | cbo4:uf:tri + uf:t",
                                                data=s[s.t >= ini], weights="sal_n", vcov={"CRV1": "cbo4"})),
        ) + ((
            ("mercado_event_shift_share", lambda: pf.feols(
                f"lsal ~ {' + '.join(e_s + c_s)} + ss_adm + ss_adm_sup | cbo4:uf:tri + uf:t", data=ev_s,
                weights="sal_n", vcov={"CRV1": "cbo4"})),
            ("mercado_pooled_shift_share", lambda: pf.feols(
                f"lsal ~ expo_post{ctl_a} + ss_adm + ss_adm_sup | cbo4:uf:tri + uf:t", data=s[s.t >= ini],
                weights="sal_n", vcov={"CRV1": "cbo4"})),
        ) if "ss_adm" in s else ()):
            try:
                t = _tab(f(), nome, "log_salario_admissao", extra={"fonte": "CAGED"})
                res.append(t)
                if "pooled" in nome:
                    log.info("mercado CAGED salário %+.4f (p=%.3f)", t.iloc[0].coef, t.iloc[0].p)
            except Exception as e:
                log.warning("mercado CAGED salário %s falhou: %s", nome, e)
    # demanda por qualificação: parcela de admitidos com superior / na base salarial (ocupação × trimestre)
    nac = d.groupby(["cbo4", "t", "tri"], as_index=False)[["adm", "adm_sup", "adm_base"]].sum()
    nac = nac[nac.adm > 0].merge(d.drop_duplicates("cbo4")[["cbo4", "expo_z", "chq_m", "boom_m"]], on="cbo4")
    tem_ss = "ss_adm" in d
    if tem_ss:
        nac = nac.merge(d.drop_duplicates(["cbo4", "t"])[["cbo4", "t", "ss_adm", "ss_adm_sup"]], on=["cbo4", "t"],
                        how="left")
    nac["expo_post"] = nac.expo_z * (nac.t >= cfg["periodo"]["inicio_pos"])
    nac["ano"] = nac.t // 10
    for num, rot in (("adm_sup", "parcela_superior_pp"), ("adm_base", "parcela_base_salarial_pp")):
        nac[rot] = 100 * nac[num] / nac.adm
        ev_n = nac.copy()
        e_n = _dummies(ev_n, "expo_z", "e", ts, refs)
        for nome, f in (
            ("mercado_event", lambda: pf.feols(f"{rot} ~ {' + '.join(e_n)} | cbo4:tri + t", data=ev_n,
                                               weights="adm", vcov={"CRV1": "cbo4"})),
            ("mercado_pooled", lambda: pf.feols(f"{rot} ~ expo_post{ctl_a} | cbo4:tri + t", data=nac[nac.t >= ini],
                                                weights="adm", vcov={"CRV1": "cbo4"})),
        ) + ((
            ("mercado_event_shift_share", lambda: pf.feols(
                f"{rot} ~ {' + '.join(e_n)} + ss_adm + ss_adm_sup | cbo4:tri + t", data=ev_n, weights="adm",
                vcov={"CRV1": "cbo4"})),
            ("mercado_pooled_shift_share", lambda: pf.feols(
                f"{rot} ~ expo_post{ctl_a} + ss_adm + ss_adm_sup | cbo4:tri + t", data=nac[nac.t >= ini],
                weights="adm", vcov={"CRV1": "cbo4"})),
        ) if tem_ss else ()):
            try:
                t = _tab(f(), nome, rot, extra={"fonte": "CAGED"})
                res.append(t)
                if "pooled" in nome:
                    log.info("mercado CAGED %s %+.3f p.p. (p=%.3f)", rot, t.iloc[0].coef, t.iloc[0].p)
            except Exception as e:
                log.warning("mercado CAGED %s %s falhou: %s", rot, nome, e)
    return res


# --------------------------------------------------------------------------------------------- PNAD
def _modelos_pnad(cfg):
    arq = caminho(cfg, "processed", "painel_pnad.parquet")
    if not arq.exists():
        log.warning("mercado: rode a etapa pnad antes (painel_pnad.parquet não existe)")
        return []
    p = pd.read_parquet(arq)
    medidas = [c for c in DESF_PNAD if c in p]
    d = p.groupby(["t", "ano", "tri", "cod"], as_index=False).agg({**{c: "sum" for c in medidas}, "expo_z": "first"})
    d["expo_post"] = d.expo_z * (d.t >= cfg["periodo"]["inicio_pos"])
    ini = cfg["periodo"]["inicio_amostra_principal"]
    ts = sorted(d.t.unique())
    refs = [t for t in ts if t // 10 == 2022]
    ev = d.copy()
    e_cols = _dummies(ev, "expo_z", "e", ts, refs)
    res = []
    for y in medidas:
        for nome, f in (
            ("mercado_event", lambda: pf.fepois(f"{y} ~ {' + '.join(e_cols)} | cod:tri + t", data=ev,
                                                vcov={"CRV1": "cod"})),
            ("mercado_pooled", lambda: pf.fepois(f"{y} ~ expo_post | cod:tri + t", data=d[d.t >= ini],
                                                 vcov={"CRV1": "cod"})),
        ):
            try:
                t = _tab(f(), nome, y, extra={"fonte": "PNAD"})
                res.append(t)
                if "pooled" in nome:
                    log.info("mercado PNAD %-9s %+.4f (p=%.3f)", y, t.iloc[0].coef, t.iloc[0].p)
            except Exception as e:
                log.warning("mercado PNAD %s %s falhou: %s", y, nome, e)
    return res


# --------------------------------------------------------------------------------------------- gráfico
def _grafico(cfg, tab, painel_desf, arquivo, titulo):
    itens = [(f, y, r) for f, y, r in painel_desf if ((tab.fonte == f) & (tab.desfecho == y)
                                                      & (tab.modelo == "mercado_event")).any()]
    if not itens:
        return
    n = len(itens)
    cols = 2
    linhas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(linhas, cols, figsize=(11, 3.2 * linhas), squeeze=False)
    a, b = cfg["periodo"]["pandemia"]
    for ax, (fonte, y, rot) in zip(axes.flat, itens):
        s = tab[(tab.fonte == fonte) & (tab.desfecho == y) & (tab.modelo == "mercado_event")]
        s = s[["periodo", "coef", "ic_inf", "ic_sup"]].copy()
        zeros = pd.DataFrame({"periodo": [20221, 20222, 20223, 20224], "coef": 0.0, "ic_inf": 0.0, "ic_sup": 0.0})
        s = pd.concat([s, zeros]).drop_duplicates("periodo").sort_values("periodo").reset_index(drop=True)
        x = np.arange(len(s))
        esc = 1 if "pp" in y or "salario" in y else 100
        dentro = x[(s.periodo >= a) & (s.periodo <= b)]
        if len(dentro):
            ax.axvspan(dentro.min() - 0.5, dentro.max() + 0.5, color="0.93", zorder=0)
        ref = x[s.periodo // 10 == 2022]
        ax.axvspan(ref.min() - 0.5, ref.max() + 0.5, color="#fde9d9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.errorbar(x, esc * s.coef, yerr=[esc * (s.coef - s.ic_inf), esc * (s.ic_sup - s.coef)], fmt="o-",
                    color="#1f5fbf", capsize=2, ms=3, lw=1)
        marc = [i for i, v in enumerate(s.periodo) if int(v) % 10 == 1]
        ax.set_xticks(marc, [f"{int(s.periodo[i]) // 10}" for i in marc], fontsize=8)
        ax.set_title(f"{fonte} · {rot}", fontsize=9)
        ax.set_ylabel("p.p." if esc == 1 and "pp" in y else ("log" if esc == 1 else "% (log × 100)"), fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def _grafico_comparacao(cfg, tab, itens, arquivo, titulo):
    """Event study sem (azul) e com (laranja) o controle de ciclo setorial."""
    itens = [(y, r) for y, r in itens if ((tab.desfecho == y) & (tab.modelo == "mercado_event_shift_share")).any()]
    if not itens:
        return
    fig, axes = plt.subplots(int(np.ceil(len(itens) / 2)), 2, figsize=(11, 3.3 * int(np.ceil(len(itens) / 2))),
                             squeeze=False)
    a, b = cfg["periodo"]["pandemia"]
    for ax, (y, rot) in zip(axes.flat, itens):
        esc = 1 if ("pp" in y or "salario" in y) else 100
        for k, (mod, cor, nome) in enumerate((("mercado_event", "#1f5fbf", "sem controle setorial"),
                                               ("mercado_event_shift_share", "#d0612a", "com controle setorial"))):
            s = tab[(tab.fonte == "CAGED") & (tab.desfecho == y) & (tab.modelo == mod)][["periodo", "coef", "ic_inf", "ic_sup"]]
            z = pd.DataFrame({"periodo": [20221, 20222, 20223, 20224], "coef": 0.0, "ic_inf": 0.0, "ic_sup": 0.0})
            s = pd.concat([s, z]).drop_duplicates("periodo").sort_values("periodo").reset_index(drop=True)
            x = np.arange(len(s)) + (k - 0.5) * 0.25
            ax.errorbar(x, esc * s.coef, yerr=[esc * (s.coef - s.ic_inf), esc * (s.ic_sup - s.coef)], fmt="o",
                        color=cor, capsize=1.5, ms=3, lw=0.9, label=nome)
        xs = np.arange(len(s))
        dentro = xs[(s.periodo >= a) & (s.periodo <= b)]
        if len(dentro):
            ax.axvspan(dentro.min() - 0.5, dentro.max() + 0.5, color="0.93", zorder=0)
        ref = xs[s.periodo // 10 == 2022]
        ax.axvspan(ref.min() - 0.5, ref.max() + 0.5, color="#fde9d9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.8)
        marc = [i for i, v in enumerate(s.periodo) if int(v) % 10 == 1]
        ax.set_xticks(marc, [f"{int(s.periodo[i]) // 10}" for i in marc], fontsize=8)
        ax.set_title(f"CAGED · {rot}", fontsize=9)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(itens):]:
        ax.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho(cfg, "outputs", "figuras", arquivo), dpi=200)
    plt.close(fig)


def rodar(cfg):
    d = _caged(cfg)
    res = _modelos_caged(cfg, d) + _modelos_pnad(cfg)
    out = pd.concat(res, ignore_index=True)
    out.to_csv(caminho(cfg, "outputs", "tabelas", "mercado.csv"), index=False)
    titulo = ("Mercado como um todo (todas as idades): ocupações mais × menos expostas à IA, "
              "por DP de exposição, vs. o mesmo trimestre de 2022 (faixa rosa)")
    _grafico(cfg, out, [("CAGED", "adm", "admissões"), ("CAGED", "des", "desligamentos"),
                        ("CAGED", "adm_sup", "admissões com superior"),
                        ("CAGED", "adm_base", "admissões na base salarial")], "mercado_caged.png", titulo)
    _grafico(cfg, out, [("CAGED", "log_salario_admissao", "log do salário de admissão"),
                        ("CAGED", "parcela_superior_pp", "parcela de admitidos com superior (p.p.)"),
                        ("PNAD", "ocup", "ocupados"), ("PNAD", "formal", "ocupados formais"),
                        ("PNAD", "informal", "ocupados informais"), ("PNAD", "recente", "recém-entrados")],
             "mercado_qualificacao_pnad.png", titulo)
    _grafico_comparacao(cfg, out, [("adm", "admissões (%)"), ("des", "desligamentos (%)"),
                                   ("adm_sup", "admissões com superior (%)"),
                                   ("parcela_superior_pp", "parcela de admitidos com superior (p.p.)"),
                                   ("log_salario_admissao", "log do salário de admissão")],
                        "mercado_shift_share.png",
                        "Mercado como um todo: sem e com controle do ciclo setorial (shift-share), vs. mesmo trimestre de 2022")
    log.info("mercado ok")
    return out
