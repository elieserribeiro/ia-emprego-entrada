"""Etapa 8 — PNAD Contínua: o outro lado da porta de entrada.

O CAGED mostra quem foi contratado; a PNAD mostra também quem procurou e não conseguiu.
1) Taxa de desocupação por faixa etária e escolaridade (superior completo), 1º semestre de 2021 a 2026.
2) Painel ocupação (COD 4 dígitos ≈ ISCO-08) × faixa × trimestre com ocupados e "recém-entrados"
   (no trabalho há menos de 1 ano, V4040), e o mesmo modelo de diferença tripla do CAGED.
Códigos usados (confirmar no dicionário baixado): VD4001=1 força de trabalho; VD4002 1 ocupado / 2 desocupado;
VD3004=7 superior completo; V4040 1 (<1 mês) e 2 (1 mês a <1 ano); peso V1028.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from .util import caminho, log

VARS = ["Ano", "Trimestre", "UF", "V1028", "V2007", "V2009", "V2010", "VD3004", "VD4001", "VD4002",
        "V4010", "VD4009", "V4040"]


def _layout(cfg) -> dict[str, tuple[int, int]]:
    zarq = caminho(cfg, "raw", "pnad", "Dicionario_e_input.zip")
    with zipfile.ZipFile(zarq) as z:
        nome = next(n for n in z.namelist() if Path(n).name.lower().startswith("input") and n.lower().endswith(".txt"))
        txt = z.read(nome).decode("latin-1")
    pos = {}
    for m in re.finditer(r"@(\d+)\s+(\w+)\s+\$?(\d+)\.", txt):
        ini, var, larg = int(m.group(1)), m.group(2), int(m.group(3))
        pos[var] = (ini - 1, ini - 1 + larg)
    return pos


def ler_trimestres(cfg):
    pos = _layout(cfg)
    usar = [v for v in VARS if v in pos]
    faltam = set(VARS) - set(usar)
    if faltam:
        log.warning("PNAD: variáveis ausentes no layout: %s", faltam)
    for zarq in sorted(Path(cfg["_caminhos"]["raw"], "pnad").glob("PNADC_0*.zip")):
        saida = caminho(cfg, "interim", "pnad", zarq.stem + ".parquet")
        if saida.exists():
            continue
        with zipfile.ZipFile(zarq) as z:
            nome = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
            with z.open(nome) as fh:
                partes = pd.read_fwf(io.TextIOWrapper(fh, encoding="latin-1"), colspecs=[pos[v] for v in usar],
                                     names=usar, dtype=str, chunksize=200_000, header=None)
                df = pd.concat(partes)
        df["V1028"] = pd.to_numeric(df.V1028, errors="coerce")
        df["V2009"] = pd.to_numeric(df.V2009, errors="coerce")
        df.to_parquet(saida, index=False)
        log.info("%s -> %d pessoas", zarq.name, len(df))


def _faixa(cfg, idade):
    for a, b, n in cfg["idade"]["faixas"]:
        if a <= idade <= b:
            return n
    return None


def rodar(cfg):
    import pyfixest as pf
    from .modelos import _tidy, grafico_event

    ler_trimestres(cfg)
    arqs = sorted(Path(cfg["_caminhos"]["interim"], "pnad").glob("*.parquet"))
    if not arqs:
        log.warning("PNAD: nenhum trimestre lido")
        return
    df = pd.concat(pd.read_parquet(a) for a in arqs)
    df = df[df.V2009 >= 14].copy()
    df["ano"] = df.Ano.astype(int)
    df["tri"] = df.Trimestre.astype(int)
    df["t"] = df.ano * 10 + df.tri
    df["faixa"] = df.V2009.map(lambda i: _faixa(cfg, i))
    df["superior"] = (df.VD3004.str.strip() == "7").astype(int)
    df["forca"] = (df.VD4001.str.strip() == "1") * df.V1028
    df["ocup"] = (df.VD4002.str.strip() == "1") * df.V1028
    df["desoc"] = (df.VD4002.str.strip() == "2") * df.V1028
    df["recente"] = df.ocup * df.get("V4040", pd.Series("", index=df.index)).fillna("").str.strip().isin(["1", "2"])

    # 1) desocupação por faixa × superior
    t = df.groupby(["t", "faixa", "superior"])[["forca", "desoc"]].sum().reset_index()
    t["taxa_desocupacao_%"] = 100 * t.desoc / t.forca
    t.to_csv(caminho(cfg, "outputs", "tabelas", "pnad_desocupacao_faixa_superior.csv"), index=False)

    # 2) painel ocupação × faixa × trimestre (cada trimestre é uma amostra própria: não há média)
    oc = df[df.ocup > 0].copy()
    oc["cod4"] = oc.V4010.str.strip().str.zfill(4)
    p = oc.groupby(["t", "tri", "cod4", "faixa"])[["ocup", "recente"]].sum().reset_index()
    ilo = pd.read_csv(caminho(cfg, "ref", "ilo2025_isco08_ocupacao.csv"), dtype={"ISCO_08": str})
    m4 = ilo.set_index("ISCO_08").mean_score_2025
    p["expo"] = p.cod4.map(m4)
    for k in (3, 2):
        mk = ilo.assign(k=ilo.ISCO_08.str[:k]).groupby("k").mean_score_2025.mean()
        f = p.expo.isna()
        if f.any():  # pandas 3: seleção vazia vira texto e não entra em coluna numérica
            p.loc[f, "expo"] = p.loc[f, "cod4"].str[:k].map(mk).astype(float)
    p = p.dropna(subset=["expo", "faixa"])
    p["expo_z"] = (p.expo - p.expo.mean()) / p.expo.std()
    jov, adu = cfg["idade"]["jovem"], cfg["idade"]["adulto"]
    p = p[p.faixa.isin(jov + adu)].copy()
    p["treat"] = p.expo_z * p.faixa.isin(jov)
    per = cfg["periodo"]
    p["treat_post"] = p.treat * (p.t >= per["inicio_pos"])
    p.to_parquet(caminho(cfg, "processed", "painel_pnad.parquet"), index=False)
    ref = per["trimestre_referencia"]
    p = p[(p.t >= per["inicio_amostra_principal"])]
    pooled = p[p.t != per["trimestre_chatgpt"]]
    res = []
    for y in ("ocup", "recente"):
        if p[y].sum() <= 0:
            continue
        for rhs, nome, d, saz in ((f"i(t, treat, ref={ref})", "pnad_tripla_event", p, ""),
                                  ("treat_post", "pnad_tripla_pooled", pooled, " + cod4:faixa:tri")):
            fit = pf.fepois(f"{y} ~ {rhs} | cod4:faixa + cod4:t + faixa:t{saz}", data=d,
                            vcov={"CRV1": "cod4"})
            tt = _tidy(fit, nome, y)
            res.append(tt)
            if nome.endswith("event"):
                grafico_event(cfg, tt, f"PNAD: jovens 18–24 × exposição, {y}", f"pnad_event_{y}.png")
    if res:
        pd.concat(res).to_csv(caminho(cfg, "outputs", "tabelas", "pnad_modelos.csv"), index=False)
    log.info("PNAD ok")
