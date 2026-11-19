"""Base de textos das ocupações (para o buscador de CBO).

Fontes, em ordem de preferência:
1. Pacote oficial "Estrutura CBO" do MTE (cbo.mte.gov.br → Downloads), descompactado em dados/cbo_mte/.
   Usa: CBO2002 - Ocupacao.csv (títulos), CBO2002 - Sinonimo.csv (sinônimos), CBO2002 - Familia.csv
   (títulos das famílias) e CBO2002 - PerfilOcupacional.csv (atividades).
2. Sem o pacote: títulos da tábua oficial MTE CBO→CIUO (dados/cbo_titulos_tabua_mte.csv, ~1.400 CBOs).

Saída: tabela longa (cbo6, tipo, texto), tipo ∈ {titulo, sinonimo, familia, atividade}.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def _ler_csv(p: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, sep=None, engine="python", dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"não consegui ler {p}")


def _achar(pasta: Path, chave: str) -> Path | None:
    for p in sorted(pasta.rglob("*")):
        if p.suffix.lower() in (".csv", ".txt") and chave in normalizar(p.stem).replace(" ", ""):
            return p
    return None


def _col(df, *candidatas):
    cols = {normalizar(c).replace(" ", "_"): c for c in df.columns}
    for c in candidatas:
        if c in cols:
            return cols[c]
    for k, c in cols.items():
        if any(x in k for x in candidatas):
            return c
    raise KeyError(f"coluna {candidatas} não encontrada em {list(df.columns)}")


def _cod6(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)


def carregar(pasta_mte: Path | None = None) -> pd.DataFrame:
    pasta = Path(pasta_mte or DADOS / "cbo_mte")
    partes = []
    ocup = _achar(pasta, "ocupacao") if pasta.exists() else None
    if ocup:
        d = _ler_csv(ocup)
        partes.append(pd.DataFrame({"cbo6": _cod6(d[_col(d, "codigo", "cod_ocupacao")]),
                                    "tipo": "titulo", "texto": d[_col(d, "titulo")]}))
        if (p := _achar(pasta, "sinonimo")):
            d = _ler_csv(p)
            partes.append(pd.DataFrame({"cbo6": _cod6(d[_col(d, "codigo", "cod_ocupacao")]),
                                        "tipo": "sinonimo", "texto": d[_col(d, "titulo", "sinonimo")]}))
        if (p := _achar(pasta, "perfilocupacional")):
            d = _ler_csv(p)
            partes.append(pd.DataFrame({"cbo6": _cod6(d[_col(d, "cod_ocupacao")]),
                                        "tipo": "atividade", "texto": d[_col(d, "nome_atividade", "atividade")]}))
        if (p := _achar(pasta, "familia")):
            d = _ler_csv(p)
            fam = dict(zip(d[_col(d, "codigo", "cod_familia")].str.replace(r"\D", "", regex=True).str.zfill(4),
                           d[_col(d, "titulo")]))
            tit = partes[0]
            partes.append(pd.DataFrame({"cbo6": tit.cbo6, "tipo": "familia", "texto": tit.cbo6.str[:4].map(fam)}))
    else:
        d = pd.read_csv(DADOS / "cbo_titulos_tabua_mte.csv", dtype=str)
        partes.append(pd.DataFrame({"cbo6": _cod6(d.cbo2002), "tipo": "titulo", "texto": d.titulo_cbo2002}))
    out = pd.concat(partes).dropna(subset=["texto"])
    out["texto"] = out.texto.str.strip()
    out = out[out.texto.str.len() > 1].drop_duplicates(["cbo6", "tipo", "texto"]).reset_index(drop=True)
    return out


def titulos(base: pd.DataFrame) -> dict[str, str]:
    t = base[base.tipo == "titulo"].drop_duplicates("cbo6")
    return dict(zip(t.cbo6, t.texto))


def documentos(base: pd.DataFrame, prefixo: str = "", max_itens: int = 40) -> pd.DataFrame:
    """Um documento por CBO: título + família + sinônimos + atividades (truncado)."""
    def junta(g):
        t = g[g.tipo == "titulo"].texto.tolist()[:1]
        f = g[g.tipo == "familia"].texto.tolist()[:1]
        s = g[g.tipo == "sinonimo"].texto.tolist()[:max_itens]
        a = g[g.tipo == "atividade"].texto.tolist()[:max_itens]
        partes = t + [f"família: {f[0]}" if f else ""] + ([("também chamado: " + "; ".join(s))] if s else []) + \
            ([("atividades: " + "; ".join(a))] if a else [])
        return prefixo + ". ".join(p for p in partes if p)
    docs = base.groupby("cbo6").apply(junta, include_groups=False).rename("documento").reset_index()
    return docs[docs.cbo6.isin(base[base.tipo == "titulo"].cbo6)]
