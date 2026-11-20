"""Etapa 8 — PNAD Contínua: o outro lado da porta de entrada.

O CAGED mostra quem foi contratado com carteira; a PNAD mostra também os informais e quem procurou e não
conseguiu. Mesmo desenho do CAGED, adaptado:

1) Validação: taxa de desocupação (14+) por trimestre, para comparar com a taxa oficial do IBGE.
2) Desocupação por faixa etária × superior completo (descritiva).
3) Painel ocupação (COD 4 dígitos ≈ ISCO-08) × faixa × trimestre, contagens expandidas pelo peso V1028:
     ocup              ocupados
     formal / informal ocupados formais e informais (definição do IBGE, abaixo)
     recente           no trabalho há menos de 1 ano (V4040 = 1 ou 2) → proxy de "entrada"
     recente_formal / recente_informal
   Tripla diferença PPML (Expo × Jovem × trimestre), jovens 18–24 e 25–29 contra adultos 30+,
   EF cod4:faixa + cod4:t + faixa:t (+ cod4:faixa:tri no pooled), amostra 2022T1+, ref. 2022T3.
   Robustezes: ocupações agregadas em 3 dígitos; série 2020T1+ com nível por fase (pré/pandemia/pós).

Informalidade (IBGE): empregado do setor privado sem carteira (VD4009=02), doméstico sem carteira (04),
trabalhador familiar auxiliar (10), e empregador/conta própria (08, 09) sem CNPJ (V4019=2). Se V4019 não
existir no arquivo, empregador e conta própria contam como informais.
Outros códigos: VD4001=1 força de trabalho; VD4002 1 ocupado / 2 desocupado; VD3004=7 superior completo.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .util import caminho, log

VARS = ["Ano", "Trimestre", "UF", "V1028", "V2007", "V2009", "V2010", "VD3004", "VD4001", "VD4002",
        "V4010", "VD4009", "V4019", "V4040"]
TEXTO = {"V4010"}  # o resto vira número
DESFECHOS_PNAD = {
    "ocup": "Ocupados",
    "formal": "Ocupados formais",
    "informal": "Ocupados informais",
    "recente": "Recém-entrados (< 1 ano no trabalho)",
    "recente_formal": "Recém-entrados formais",
    "recente_informal": "Recém-entrados informais",
}


def _layout(cfg) -> dict[str, tuple[int, int]]:
    import re
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
    """Cada zip → parquet com as variáveis de VARS. O texto de largura fixa é extraído para uma pasta
    temporária (um trimestre por vez, ~2 GB) e lido pelo DuckDB, muito mais rápido que pandas.read_fwf."""
    pos = _layout(cfg)
    usar = [v for v in VARS if v in pos]
    faltam = set(VARS) - set(usar)
    if faltam:
        log.warning("PNAD: variáveis ausentes no layout: %s", sorted(faltam))
    cols = []
    for v in usar:
        a, b = pos[v]
        expr = f"substr(l, {a + 1}, {b - a})"
        if v in TEXTO:
            cols.append(f"trim({expr}) AS {v}")
        elif v == "V1028":
            cols.append(f"TRY_CAST(trim({expr}) AS DOUBLE) AS {v}")
        else:
            cols.append(f"TRY_CAST(trim({expr}) AS INTEGER) AS {v}")
    for zarq in sorted(Path(cfg["_caminhos"]["raw"], "pnad").glob("PNADC_0*.zip")):
        saida = caminho(cfg, "interim", "pnad", zarq.stem + ".parquet")
        if saida.exists():
            existentes = set(pd.read_parquet(saida, columns=None).columns) if saida.stat().st_size < 1 else None
            import pyarrow.parquet as pq
            if set(usar) <= set(pq.read_schema(saida).names):
                continue  # já lido com todas as variáveis
        tmp = Path(tempfile.mkdtemp(prefix="pnad_"))
        try:
            with zipfile.ZipFile(zarq) as z:
                nome = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
                txt = Path(z.extract(nome, tmp))
            con = duckdb.connect()
            con.sql(f"""COPY (SELECT {', '.join(cols)}
                              FROM read_csv('{txt.as_posix()}', columns={{'l': 'VARCHAR'}}, delim='|',
                                            header=false, quote='', escape='', auto_detect=false))
                        TO '{saida.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
            n = con.sql(f"SELECT count(*) FROM read_parquet('{saida.as_posix()}')").fetchone()[0]
            con.close()
            log.info("%s -> %d pessoas", zarq.name, n)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _faixa(cfg, idade: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=idade.index, dtype="object")
    for a, b, n in cfg["idade"]["faixas"]:
        out[(idade >= a) & (idade <= b)] = n
    return out


def _carregar(cfg) -> pd.DataFrame:
    arqs = sorted(Path(cfg["_caminhos"]["interim"], "pnad").glob("*.parquet"))
    if not arqs:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(a) for a in arqs), ignore_index=True)
    df = df[df.V2009 >= 14].copy()
    df["t"] = df.Ano * 10 + df.Trimestre
    df["tri"] = df.Trimestre
    df["ano"] = df.Ano
    df["faixa"] = _faixa(cfg, df.V2009)
    df["superior"] = (df.VD3004 == 7).astype(int)
    w = df.V1028.fillna(0.0)
    oc = df.VD4002 == 1
    df["forca"] = (df.VD4001 == 1) * w
    df["ocup"] = oc * w
    df["desoc"] = (df.VD4002 == 2) * w
    pos = df.VD4009
    sem_cnpj = (df.V4019 == 2) if "V4019" in df else pos.isin([8, 9])
    informal = pos.isin([2, 4, 10]) | (pos.isin([8, 9]) & sem_cnpj)
    df["informal"] = (oc & informal) * w
    df["formal"] = (oc & ~informal) * w
    rec = df.V4040.isin([1, 2]) if "V4040" in df else pd.Series(False, index=df.index)
    df["recente"] = (oc & rec) * w
    df["recente_formal"] = (oc & rec & ~informal) * w
    df["recente_informal"] = (oc & rec & informal) * w
    return df


def _validacao(cfg, df):
    v = df.groupby("t")[["forca", "desoc", "ocup", "informal"]].sum()
    v["taxa_desocupacao_%"] = 100 * v.desoc / v.forca
    v["taxa_informalidade_%"] = 100 * v.informal / v.ocup
    v[["ocup", "desoc", "forca"]] = v[["ocup", "desoc", "forca"]] / 1e6
    v = v.rename(columns={"ocup": "ocupados_milhoes", "desoc": "desocupados_milhoes", "forca": "forca_milhoes"})
    v = v.drop(columns="informal").round(2).reset_index()
    v.to_csv(caminho(cfg, "outputs", "validacao", "pnad_taxas_brasil.csv"), index=False)
    log.info("PNAD validação (compare com o IBGE):\n%s", v.to_string(index=False))
    t = df.groupby(["t", "faixa", "superior"])[["forca", "desoc", "ocup", "informal"]].sum().reset_index()
    t["taxa_desocupacao_%"] = 100 * t.desoc / t.forca
    t["taxa_informalidade_%"] = 100 * t.informal / t.ocup
    t.to_csv(caminho(cfg, "outputs", "tabelas", "pnad_desocupacao_faixa_superior.csv"), index=False)


def _exposicao(cfg, p: pd.DataFrame, nivel: int = 4) -> pd.DataFrame:
    """Escore OIT pelo COD (≈ ISCO-08), com recuo para 3 e 2 dígitos; padronizado com pesos = ocupados de
    2022T1–T3 (todas as idades), como no CAGED (admissões de jan–set/2022)."""
    ilo = pd.read_csv(caminho(cfg, "ref", "ilo2025_isco08_ocupacao.csv"), dtype={"ISCO_08": str})
    if nivel == 4:
        p["expo"] = p.cod.map(ilo.set_index("ISCO_08").mean_score_2025).astype(float)
        recuos = (3, 2)
    else:
        p["expo"] = np.nan
        recuos = (nivel, 2)
    for k in recuos:
        mk = ilo.assign(k=ilo.ISCO_08.str[:k]).groupby("k").mean_score_2025.mean()
        f = p.expo.isna()
        if f.any():
            p.loc[f, "expo"] = p.loc[f, "cod"].str[:k].map(mk).astype(float)
    p = p.dropna(subset=["expo"]).copy()
    a, b = cfg["periodo"]["inicio_amostra_principal"], cfg["periodo"]["trimestre_referencia"]
    pesos = p[(p.t >= a) & (p.t <= b)].groupby("cod").ocup.sum()
    oc = p.groupby("cod").expo.first().to_frame().join(pesos.rename("w")).fillna({"w": 0.0})
    m = np.average(oc.expo, weights=oc.w)
    s = np.sqrt(np.average((oc.expo - m) ** 2, weights=oc.w))
    p["expo_z"] = (p.expo - m) / s
    return p


def _painel(cfg, df, nivel=4):
    oc = df[df.ocup > 0].copy()
    oc["cod"] = oc.V4010.astype(str).str.strip().str.zfill(4).str[:nivel]
    oc = oc[oc.cod.str.match(r"^\d+$") & (oc.cod.str[:1] != "0")]  # sem forças armadas / código inválido
    medidas = list(DESFECHOS_PNAD)
    p = oc.dropna(subset=["faixa"]).groupby(["t", "ano", "tri", "cod", "faixa"])[medidas].sum().reset_index()
    return _exposicao(cfg, p, nivel)


def _tripla(cfg, p, jov, y, pooled, rotulo, fase=False):
    import pyfixest as pf
    from .modelos import _tidy
    per = cfg["periodo"]
    adu = cfg["idade"]["adulto"]
    d = p[p.faixa.isin(jov + adu)].copy()
    d["treat"] = d.expo_z * d.faixa.isin(jov)
    d["treat_post"] = d.treat * (d.t >= per["inicio_pos"])
    ef = "cod:faixa + cod:t + faixa:t"
    if fase:
        a, b = per["pandemia"]
        d["fase"] = np.where(d.t < a, "pre", np.where(d.t <= b, "pandemia", "pos"))
        ef += " + cod:faixa:fase"
    else:
        d = d[d.t >= per["inicio_amostra_principal"]]
    if pooled:
        d = d[d.t != per["trimestre_chatgpt"]]
        ef += " + cod:faixa:tri"
        rhs = "treat_post"
    else:
        rhs = f"i(t, treat, ref={per['trimestre_referencia']})"
    if d[y].sum() <= 0:
        return None
    fit = pf.fepois(f"{y} ~ {rhs} | {ef}", data=d, vcov={"CRV1": "cod"})
    t = _tidy(fit, rotulo, y)
    return t[t["var"] == "treat"]


def rodar(cfg):
    from .modelos import GRUPOS, grafico_event
    ler_trimestres(cfg)
    df = _carregar(cfg)
    if df.empty:
        log.warning("PNAD: nenhum trimestre lido")
        return
    _validacao(cfg, df)
    p4 = _painel(cfg, df, 4)
    p4.to_parquet(caminho(cfg, "processed", "painel_pnad.parquet"), index=False)
    p3 = _painel(cfg, df, 3)
    extra = {"jovem_25_29": ["25_29"]}
    res = []
    for chave, grupo in GRUPOS.items():
        jov = cfg["idade"].get(chave) or extra[chave]
        for y in DESFECHOS_PNAD:
            tarefas = [("pnad_event", p4, False, False), ("pnad_pooled", p4, True, False),
                       ("pnad_pooled_cod3", p3, True, False), ("pnad_pooled_fase_2020", p4, True, True)]
            for rot, p, pooled, fase in tarefas:
                try:
                    t = _tripla(cfg, p, jov, y, pooled, rot, fase)
                except Exception as e:
                    log.warning("PNAD %s %s %s falhou: %s", grupo, y, rot, e)
                    continue
                if t is None:
                    continue
                t["grupo"] = grupo
                res.append(t)
                if rot == "pnad_event" and y in ("ocup", "formal", "informal", "recente"):
                    grafico_event(cfg, t, f"PNAD · jovens {grupo.replace('_', '–')} × exposição: "
                                          f"{DESFECHOS_PNAD[y].lower()}", f"pnad_event_{grupo}_{y}.png")
                if pooled:
                    r = t[t.termo == "treat_post"].iloc[0]
                    log.info("PNAD %s %-16s %-24s %+.4f (p=%.3f)", grupo, y, rot, r.coef, r.p)
    if res:
        out = pd.concat(res, ignore_index=True)
        out.to_csv(caminho(cfg, "outputs", "tabelas", "pnad_modelos.csv"), index=False)
    log.info("PNAD ok")
