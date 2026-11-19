"""Etapa 2 — Ingestão e validação dos microdados do Novo CAGED.

ingerir():  .7z -> Parquet (um por arquivo), colunas com nomes normalizados, tudo texto.
conexao():  DuckDB com a view `mov` (movimentos tipados) e `mov_ajustado` (regra de ajustes do MTE).
validar():  compara os totais mensais com as Tabelas 5 (sem ajustes) e 5.1 (com ajustes) do MTE
            e grava as frequências dos códigos (sexo, raça, instrução, tipo de movimentação...).
"""
from __future__ import annotations

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from .util import caminho, intervalo_comp, log, normalizar_nome

COLUNAS_ESSENCIAIS = ["competenciamov", "saldomovimentacao", "cbo2002ocupacao", "idade", "uf"]


def _extrair_7z(arq: Path, destino: Path) -> Path:
    exe = shutil.which("7z") or shutil.which("7za") or shutil.which(r"C:\Program Files\7-Zip\7z.exe")
    if exe:  # 7-Zip nativo é bem mais rápido que py7zr
        subprocess.run([exe, "e", "-y", f"-o{destino}", str(arq)], check=True, capture_output=True)
    else:
        import py7zr
        with py7zr.SevenZipFile(arq, "r") as z:
            z.extractall(path=destino)
    arquivos = [Path(f) for f in glob.glob(str(destino / "**" / "*"), recursive=True) if Path(f).is_file()]
    return max(arquivos, key=lambda f: f.stat().st_size)


def ingerir(cfg, refazer=False):
    brutos = sorted(Path(cfg["_caminhos"]["raw"], "caged").glob("CAGED*.7z"))
    if not brutos:
        raise FileNotFoundError("nenhum CAGED*.7z em data/raw/caged — rode a etapa de download")
    con = duckdb.connect()
    for arq in brutos:
        tipo, mes_pub = arq.stem[5:8], arq.stem[8:14]
        saida = caminho(cfg, "interim", "caged", f"{tipo}_{mes_pub}.parquet")
        if saida.exists() and not refazer:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            txt = _extrair_7z(arq, Path(tmp))
            rel = con.sql(f"""SELECT * FROM read_csv('{txt.as_posix()}', delim=';', header=true,
                               all_varchar=true, quote='"', ignore_errors=false)""")
            cols = rel.columns
            norm = [normalizar_nome(c) for c in cols]
            faltam = [c for c in COLUNAS_ESSENCIAIS if c not in norm]
            if faltam:
                raise ValueError(f"{arq.name}: colunas ausentes {faltam}; cabeçalho = {cols}")
            sel = ", ".join(f'"{c}" AS {n}' for c, n in zip(cols, norm))
            con.sql(f"""COPY (SELECT {sel}, '{tipo}' AS tipo_arquivo, '{mes_pub}' AS mes_pub
                         FROM read_csv('{txt.as_posix()}', delim=';', header=true, all_varchar=true))
                        TO '{saida.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
        n = con.sql(f"SELECT count(*) FROM '{saida.as_posix()}'").fetchall()[0][0]
        log.info("%s -> %s (%d linhas)", arq.name, saida.name, n)


def conexao(cfg, memoria="8GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql(f"SET memory_limit='{memoria}'")
    padrao = Path(cfg["_caminhos"]["interim"], "caged", "*.parquet").as_posix()
    lim = int(cfg["caged"]["meses_ajuste"])
    con.sql(f"""
    CREATE OR REPLACE VIEW mov AS
    SELECT
        tipo_arquivo,
        CAST(mes_pub AS INTEGER)                                   AS mes_pub,
        CAST(substr(competenciamov, 1, 6) AS INTEGER)              AS comp,
        TRY_CAST(saldomovimentacao AS INTEGER)                     AS saldo_bruto,
        CASE WHEN tipo_arquivo = 'EXC' THEN -1 ELSE 1 END          AS sinal,
        lpad(trim(cbo2002ocupacao), 6, '0')                        AS cbo6,
        TRY_CAST(idade AS INTEGER)                                 AS idade,
        TRY_CAST(uf AS INTEGER)                                    AS uf,
        TRY_CAST(regiao AS INTEGER)                                AS regiao,
        secao,
        TRY_CAST(sexo AS INTEGER)                                  AS sexo,
        TRY_CAST(racacor AS INTEGER)                               AS raca,
        TRY_CAST(graudeinstrucao AS INTEGER)                       AS instr,
        TRY_CAST(tipomovimentacao AS INTEGER)                      AS tipomov,
        TRY_CAST(indicadoraprendiz AS INTEGER)                     AS aprendiz,
        TRY_CAST(indtrabintermitente AS INTEGER)                   AS intermitente,
        TRY_CAST(replace(replace(salario, '.', ''), ',', '.') AS DOUBLE) AS salario_txt_br,
        TRY_CAST(salario AS DOUBLE)                                AS salario_txt_pt
    FROM read_parquet('{padrao}', union_by_name=true)
    """)
    # salário: o MTE usa vírgula decimal ("1500,00"); se vier com ponto, a 2ª forma acerta
    con.sql("""
    CREATE OR REPLACE VIEW mov2 AS
    SELECT *, COALESCE(CASE WHEN salario_txt_pt IS NOT NULL THEN salario_txt_pt END, salario_txt_br) AS salario,
              (comp // 100) * 12 + comp % 100 AS comp_m, (mes_pub // 100) * 12 + mes_pub % 100 AS pub_m
    FROM mov""")
    con.sql(f"""
    CREATE OR REPLACE VIEW mov_ajustado AS
    SELECT * FROM mov2
    WHERE tipo_arquivo = 'MOV'
       OR (tipo_arquivo IN ('FOR', 'EXC') AND pub_m - comp_m BETWEEN 0 AND {lim})""")
    return con


def _totais(con, view):
    return con.sql(f"""
        SELECT comp // 100 AS ano, comp % 100 AS mes,
               sum(CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END)  AS admissoes,
               sum(CASE WHEN saldo_bruto = -1 THEN sinal ELSE 0 END) AS desligamentos,
               sum(saldo_bruto * sinal)                              AS saldo
        FROM {view} GROUP BY 1, 2 ORDER BY 1, 2""").df()


def validar(cfg):
    con = conexao(cfg)
    out = Path(cfg["_caminhos"]["outputs"], "validacao")
    out.mkdir(parents=True, exist_ok=True)
    ini, fim = intervalo_comp(cfg, con)
    res = []
    for view, ref in (("mov2 WHERE tipo_arquivo='MOV'", "mte_tabela5_sem_ajustes.csv"),
                      ("mov_ajustado", "mte_tabela5_1_com_ajustes.csv")):
        micro = _totais(con, f"(SELECT * FROM {view})")
        oficial = pd.read_csv(caminho(cfg, "ref", ref))
        m = micro.merge(oficial, on=["ano", "mes"], suffixes=("_micro", "_mte"))
        comp = m.ano * 100 + m.mes
        m = m[(comp >= ini) & (comp <= fim)]
        for v in ("admissoes", "desligamentos", "saldo"):
            m[f"dif_{v}"] = m[f"{v}_micro"] - m[f"{v}_mte"]
        m["dif_adm_pct"] = 100 * m.dif_admissoes / m.admissoes_mte
        m["serie"] = ref.replace(".csv", "")
        res.append(m)
    res = pd.concat(res)
    res.to_csv(out / "validacao_totais.csv", index=False)
    pior = res.groupby("serie").dif_adm_pct.apply(lambda s: s.abs().max())
    log.info("maior diferença de admissões (%%) por série:\n%s", pior.round(3).to_string())
    # frequências dos códigos — confirmar no layout oficial antes de interpretar
    freqs = []
    for col in ("tipomov", "sexo", "raca", "instr", "aprendiz", "intermitente", "regiao"):
        f = con.sql(f"""SELECT '{col}' AS variavel, {col} AS codigo,
                          sum(CASE WHEN saldo_bruto = 1 THEN 1 ELSE 0 END) AS n_admissoes,
                          sum(CASE WHEN saldo_bruto = -1 THEN 1 ELSE 0 END) AS n_desligamentos
                        FROM mov2 WHERE tipo_arquivo = 'MOV' GROUP BY 1, 2 ORDER BY 2""").df()
        freqs.append(f)
    pd.concat(freqs).to_csv(out / "frequencias_codigos.csv", index=False)
    sal = con.sql("""SELECT comp // 100 AS ano, quantile_cont(salario, [0.01, 0.5, 0.99]) AS q,
                            avg(CASE WHEN salario IS NULL THEN 1.0 ELSE 0 END) AS pct_nulo
                     FROM mov2 WHERE tipo_arquivo='MOV' AND saldo_bruto = 1 GROUP BY 1 ORDER BY 1""").df()
    sal.to_csv(out / "salario_quantis.csv", index=False)
    log.info("validação gravada em %s", out)
    return res
