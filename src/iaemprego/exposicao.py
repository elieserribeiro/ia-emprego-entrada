"""Etapa 3 — Exposição à IA generativa por ocupação (CBO 2002).

Ponte: CBO 2002 -> CIUO/ISCO-88 (tábua oficial do MTE) -> ISCO-08 (Ganzeboom & Treiman)
       -> escore médio das tarefas da ocupação no índice OIT 2025 (Gmyrek et al., WP 140).
Arquivo: data/ref/ponte_cbo2002_isco08_ilo2025.csv (gerado por scripts de referência; ver README).

Imputação: CBO sem escore recebe a média (ponderada por admissões de jan–set/2022) das CBOs com escore
da mesma família (4 dígitos), depois subgrupo (3) e subgrupo principal (2). Fica marcado em `imputado`.
Saídas: data/processed/exposicao_cbo6.parquet, exposicao_cbo4.parquet e outputs/validacao/cobertura_exposicao.csv
"""
from __future__ import annotations

import pandas as pd

from .caged import conexao
from .util import caminho, log


def _pesos_referencia(cfg) -> pd.DataFrame:
    con = conexao(cfg)
    a, b = cfg["periodo"]["meses_pesos"]
    return con.sql(f"""
        SELECT cbo6, sum(CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END) AS adm_ref
        FROM mov_ajustado WHERE comp BETWEEN {int(a)} AND {int(b)}
        GROUP BY 1""").df()


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
    grad = df.ilo_gradiente.fillna("")
    df["expo_g34"] = grad.str.contains("Gradient 3|Gradient 4").astype(int)
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
        ("admissões 1S de referência", tot),
        ("% com escore direto (ponte por ocupação/família/prefixo)", 100 * df.loc[~df.imputado, "adm_ref"].sum() / tot),
        ("% com escore imputado pela hierarquia CBO", 100 * df.loc[df.imputado & df.expo.notna(), "adm_ref"].sum() / tot),
        ("% sem escore", 100 * df.loc[df.expo.isna(), "adm_ref"].sum() / tot),
        ("% via ponte por ocupação (tábua MTE)", 100 * df.loc[df.nivel_ponte == "ocupacao", "adm_ref"].sum() / tot),
        ("% com ISCO-88->08 ambíguo (>1 destino)", 100 * df.loc[df.isco88_isco08_alternativas.fillna(1) > 1, "adm_ref"].sum() / tot),
    ], columns=["indicador", "valor"])
    out = caminho(cfg, "outputs", "validacao", "cobertura_exposicao.csv")
    cob.to_csv(out, index=False)
    top = df.sort_values("adm_ref", ascending=False).head(40)[
        ["cbo6", "adm_ref", "isco08", "expo", "ilo_gradiente", "imputado"]]
    top.to_csv(caminho(cfg, "outputs", "validacao", "exposicao_top40_cbo.csv"), index=False)
    log.info("cobertura da exposição:\n%s", cob.round(2).to_string(index=False))
    return df
