"""Etapa 3 — Exposição à IA generativa por ocupação (CBO 2002).

Ponte: CBO 2002 -> CIUO/ISCO-88 (tábua oficial do MTE) -> ISCO-08 (Ganzeboom & Treiman)
       -> escore médio das tarefas da ocupação no índice OIT 2025 (Gmyrek et al., WP 140).
Arquivo: data/ref/ponte_cbo2002_isco08_ilo2025.csv (gerado por scripts de referência; ver README).

Imputação: CBO sem escore recebe a média (ponderada por admissões de jan–set/2022) das CBOs com escore
da mesma família (4 dígitos), depois subgrupo (3) e subgrupo principal (2). Fica marcado em `imputado`.
Gradientes (para a robustez "gradientes 3–4"): quando a ponte só chega a um grupo ISCO-08 de 3 ou 2 dígitos,
a OIT não publica gradiente; ele é calculado pela regra oficial (WP 140, Tabela 5) com a média e o
desvio-padrão das tarefas do grupo (agregando as ocupações de 4 dígitos). A regra reproduz 99% dos gradientes
oficiais a 4 dígitos (100% dos gradientes 1–4). CBOs sem ISCO recebem a parcela de G3–4 da família, como a
exposição. Coluna `gradiente_origem`: oit | regra_oit_grupo | imputado.
Saídas: data/processed/exposicao_cbo6.parquet, exposicao_cbo4.parquet e outputs/validacao/cobertura_exposicao.csv
"""
from __future__ import annotations

import pandas as pd

import numpy as np

from .caged import conexao
from .util import caminho, log


def _pesos_referencia(cfg) -> pd.DataFrame:
    con = conexao(cfg)
    a, b = cfg["periodo"]["meses_pesos"]
    return con.sql(f"""
        SELECT cbo6, sum(CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END) AS adm_ref
        FROM mov_ajustado WHERE comp BETWEEN {int(a)} AND {int(b)}
        GROUP BY 1""").df()


ROTULO = {"g4": "Exposed: Gradient 4", "g3": "Exposed: Gradient 3", "g2": "Exposed: Gradient 2",
          "g1": "Exposed: Gradient 1", "minima": "Minimal Exposure", "nao_exposta": "Not Exposed"}


def gradiente_oit(mu: float, sd: float) -> str:
    """Regra oficial da OIT (Gmyrek et al. 2025, WP 140, Tabela 5)."""
    sd = 0.0 if pd.isna(sd) else sd
    if mu >= 0.6 and mu - sd >= 0.5:
        g = "g4"
    elif 0.5 <= mu < 0.6 and mu + sd >= 0.5:
        g = "g3"
    elif 0.4 <= mu < 0.5 and mu + sd >= 0.5:
        g = "g2"
    elif mu < 0.4 and mu + sd >= 0.5:
        g = "g1"
    elif mu < 0.5 and mu + sd > 0.4:
        g = "minima"
    else:
        g = "nao_exposta"
    return ROTULO[g]


def _gradientes_grupo(cfg) -> dict[str, str]:
    """Gradiente de cada grupo ISCO-08 de 3 e 2 dígitos, com média e DP agregados das tarefas."""
    o = pd.read_csv(caminho(cfg, "ref", "ilo2025_isco08_ocupacao.csv"), dtype={"ISCO_08": str})
    o["ISCO_08"] = o.ISCO_08.str.zfill(4)
    res = {}
    for k in (3, 2):
        for g, d in o.groupby(o.ISCO_08.str[:k]):
            n, m, s = d.n_tarefas.to_numpy(float), d.mean_score_2025.to_numpy(float), d.sd_2025.fillna(0).to_numpy(float)
            N = n.sum()
            M = (n * m).sum() / N
            ss = ((n - 1) * s ** 2).sum() + (n * (m - M) ** 2).sum()  # dentro + entre ocupações
            res[g] = gradiente_oit(M, np.sqrt(ss / (N - 1)) if N > 1 else 0.0)
    return res


def construir(cfg):
    ponte = pd.read_csv(caminho(cfg, "ref", "ponte_cbo2002_isco08_ilo2025.csv"),
                        dtype={"cbo2002": str, "isco88": str, "isco08": str, "familia": str})
    ponte = ponte.rename(columns={"cbo2002": "cbo6"})
    pesos = _pesos_referencia(cfg)
    # CBOs que aparecem nos microdados mas não estão no domínio oficial (ex.: códigos novos)
    extra = pesos[~pesos.cbo6.isin(ponte.cbo6)][["cbo6"]]
    df = pd.concat([ponte, extra], ignore_index=True).merge(pesos, on="cbo6", how="left")
    df["adm_ref"] = df.adm_ref.fillna(0).clip(lower=0)
    df["expo"] = df.ilo_mean_2025
    df["imputado"] = df.expo.isna()
    df["nivel_imputacao"] = None
    for k in (4, 3, 2):
        g = df[~df.expo.isna()].assign(p=lambda d: d.cbo6.str[:k], w=lambda d: d.adm_ref + 1e-9)
        media = g.groupby("p").apply(lambda x: (x.expo * x.w).sum() / x.w.sum(), include_groups=False)
        falta = df.expo.isna()
        if not falta.any():
            break
        df.loc[falta, "expo"] = df.loc[falta, "cbo6"].str[:k].map(media).astype(float)
        df.loc[falta & df.expo.notna(), "nivel_imputacao"] = f"cbo{k}"
    # padronização (z) ponderada pelas admissões de referência
    v = df.dropna(subset=["expo"])
    mu = (v.expo * v.adm_ref).sum() / v.adm_ref.sum()
    sd = (((v.expo - mu) ** 2 * v.adm_ref).sum() / v.adm_ref.sum()) ** 0.5
    df["expo_z"] = (df.expo - mu) / sd
    # quartis ponderados por admissões (1 = menos exposto)
    v = df.dropna(subset=["expo"]).sort_values("expo")
    acum = v.adm_ref.cumsum() / v.adm_ref.sum()
    df.loc[v.index, "expo_quartil"] = (acum.clip(upper=0.999999) * 4).astype(int) + 1
    # gradientes: oficiais a 4 dígitos; regra da OIT nos grupos de 3/2 dígitos; família CBO para os sem ISCO
    grupos = _gradientes_grupo(cfg)
    df["gradiente_origem"] = np.where(df.ilo_gradiente.notna(), "oit", None)
    for nivel, k in (("isco3", 3), ("isco2", 2)):
        f = df.ilo_gradiente.isna() & (df.ilo_nivel == nivel) & df.isco08.notna()
        if f.any():
            df.loc[f, "ilo_gradiente"] = df.loc[f, "isco08"].str.zfill(4).str[:k].map(grupos)
            df.loc[f & df.ilo_gradiente.notna(), "gradiente_origem"] = "regra_oit_grupo"
    grad = df.ilo_gradiente.fillna("")
    df["expo_g34"] = np.where(df.ilo_gradiente.notna(),
                              grad.str.contains("Gradient 3|Gradient 4").astype(float), np.nan)
    for k in (4, 3, 2):
        falta = df.expo_g34.isna()
        if not falta.any():
            break
        g = df[~falta].assign(p=lambda d: d.cbo6.str[:k], w=lambda d: d.adm_ref + 1e-9)
        parc = g.groupby("p").apply(lambda x: (x.expo_g34 * x.w).sum() / x.w.sum(), include_groups=False)
        df.loc[falta, "expo_g34"] = df.loc[falta, "cbo6"].str[:k].map(parc).astype(float)
        df.loc[falta & df.expo_g34.notna(), "gradiente_origem"] = "imputado"
    df["expo_g34"] = df.expo_g34.fillna(0.0)
    df.to_parquet(caminho(cfg, "processed", "exposicao_cbo6.parquet"), index=False)

    # família (4 dígitos): média ponderada pelas admissões de referência
    f = df.dropna(subset=["expo"]).assign(cbo4=lambda d: d.cbo6.str[:4], w=lambda d: d.adm_ref + 1e-9)
    fam = f.groupby("cbo4").apply(lambda x: pd.Series({
        "expo": (x.expo * x.w).sum() / x.w.sum(),
        "expo_z": (x.expo_z * x.w).sum() / x.w.sum(),
        "expo_g34": (x.expo_g34 * x.w).sum() / x.w.sum(),
        "pct_imputado": (x.imputado * x.w).sum() / x.w.sum(),
        "adm_ref": x.adm_ref.sum()}), include_groups=False).reset_index()
    fam = fam.sort_values("expo")
    acum = fam.adm_ref.cumsum() / fam.adm_ref.sum()
    fam["expo_quartil"] = (acum.clip(upper=0.999999) * 4).astype(int) + 1
    fam.to_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"), index=False)

    # cobertura ponderada pelas admissões (o número que importa para o artigo)
    tot = df.adm_ref.sum()
    cob = pd.DataFrame([
        ("admissões de referência (jan–set/2022)", tot),
        ("% com escore direto (ponte por ocupação/família/prefixo)", 100 * df.loc[~df.imputado, "adm_ref"].sum() / tot),
        ("% com escore imputado pela hierarquia CBO", 100 * df.loc[df.imputado & df.expo.notna(), "adm_ref"].sum() / tot),
        ("% sem escore", 100 * df.loc[df.expo.isna(), "adm_ref"].sum() / tot),
        ("% via ponte por ocupação (tábua MTE)", 100 * df.loc[df.nivel_ponte == "ocupacao", "adm_ref"].sum() / tot),
        ("% com ISCO-88->08 ambíguo (>1 destino)", 100 * df.loc[df.isco88_isco08_alternativas.fillna(1) > 1, "adm_ref"].sum() / tot),
        ("% gradiente oficial OIT (4 dígitos)", 100 * df.loc[df.gradiente_origem == "oit", "adm_ref"].sum() / tot),
        ("% gradiente pela regra OIT no grupo (3/2 dígitos)", 100 * df.loc[df.gradiente_origem == "regra_oit_grupo", "adm_ref"].sum() / tot),
        ("% gradiente imputado pela família CBO", 100 * df.loc[df.gradiente_origem == "imputado", "adm_ref"].sum() / tot),
        ("% em gradientes 3–4", 100 * (df.expo_g34 * df.adm_ref).sum() / tot),
    ], columns=["indicador", "valor"])
    out = caminho(cfg, "outputs", "validacao", "cobertura_exposicao.csv")
    cob.to_csv(out, index=False)
    top = df.sort_values("adm_ref", ascending=False).head(40)[
        ["cbo6", "adm_ref", "isco08", "expo", "ilo_gradiente", "imputado"]]
    top.to_csv(caminho(cfg, "outputs", "validacao", "exposicao_top40_cbo.csv"), index=False)
    log.info("cobertura da exposição:\n%s", cob.round(2).to_string(index=False))
    return df
