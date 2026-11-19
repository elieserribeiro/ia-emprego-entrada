"""Etapa 4 — Painéis trimestrais e medidas de choque da pandemia.

painel principal:       t (trimestre) × cbo4 × faixa etária × UF
painel heterogeneidade: t × cbo4 × faixa × (dimensão, valor) para sexo, raça, instrução, região, NE, RN
pandemia_cbo4:          choque (abr–jun/2020) e boom (jan–fev/2022) por família, como JOVEM − ADULTO

Medidas de entrada (contagens de admissões, líquidas de exclusões):
  adm, adm_1emp (primeiro emprego), adm_sup (superior completo+), adm_base (salário real ≤ p25 da família
  em jan–set/2022), adm_aprendiz; e des (desligamentos). Salários: soma e soma do log do salário real.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .caged import conexao
from .util import caminho, intervalo_comp, log


def _lista(xs):
    return ",".join(str(int(x)) for x in xs)


def _faixa_sql(cfg):
    casos = " ".join(f"WHEN idade BETWEEN {a} AND {b} THEN '{n}'" for a, b, n in cfg["idade"]["faixas"])
    return f"CASE {casos} END"


def _base(cfg, con: duckdb.DuckDBPyConnection, somente_mov=False, excluir_secoes=(), excluir_aprendiz=False,
          excluir_intermitente=False):
    ipca = pd.read_csv(caminho(cfg, "ref", "ipca.csv"))
    base = ipca.loc[ipca.ano_mes == cfg["ipca"]["base"], "indice"]
    if base.empty:
        base = ipca.loc[ipca.ano_mes == ipca.ano_mes.max(), "indice"]
        log.warning("IPCA sem o mês-base %s; usando %s", cfg["ipca"]["base"], ipca.ano_mes.max())
    ipca["deflator"] = float(base.iloc[0]) / ipca.indice
    con.register("ipca_df", ipca[["ano_mes", "deflator"]])
    c = cfg["codigos"]
    ini, fim = intervalo_comp(cfg, con)
    filtros = [f"comp BETWEEN {ini} AND {fim}", "idade >= 14"]
    if somente_mov:
        filtros.append("tipo_arquivo = 'MOV'")
    if excluir_secoes:
        filtros.append("coalesce(secao,'') NOT IN (" + ",".join(f"'{s}'" for s in excluir_secoes) + ")")
    if excluir_aprendiz:
        filtros.append(f"coalesce(aprendiz,0) NOT IN ({_lista(c['aprendiz'])})")
    if excluir_intermitente:
        filtros.append(f"coalesce(intermitente,0) NOT IN ({_lista(c['intermitente'])})")
    con.sql(f"""
    CREATE OR REPLACE TEMP VIEW base AS
    SELECT m.*, comp // 100 AS ano, ((comp % 100) - 1) // 3 + 1 AS tri,
           (comp // 100) * 10 + ((comp % 100) - 1) // 3 + 1 AS t,
           substr(cbo6, 1, 4) AS cbo4, {_faixa_sql(cfg)} AS faixa,
           CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END  AS e_adm,
           CASE WHEN saldo_bruto = -1 THEN sinal ELSE 0 END AS e_des,
           CASE WHEN saldo_bruto = 1 AND tipo_arquivo <> 'EXC' AND salario > 0 AND salario < 500000
                THEN salario * i.deflator END AS sal_real
    FROM mov_ajustado m LEFT JOIN ipca_df i ON i.ano_mes = m.comp
    WHERE {' AND '.join(filtros)}""")
    a, b = cfg["periodo"]["meses_pesos"]
    con.sql(f"""CREATE OR REPLACE TEMP TABLE limiar AS
                SELECT cbo4, quantile_cont(sal_real, 0.25) AS p25 FROM base
                WHERE comp BETWEEN {int(a)} AND {int(b)} AND sal_real IS NOT NULL
                GROUP BY 1 HAVING count(*) >= 30""")
    return c


def _agregados(c):
    return f"""
        sum(e_adm) AS adm, sum(e_des) AS des, sum(e_adm) - sum(e_des) AS saldo,
        sum(CASE WHEN tipomov IN ({_lista(c['primeiro_emprego'])}) THEN e_adm ELSE 0 END) AS adm_1emp,
        sum(CASE WHEN instr IN ({_lista(c['superior'])}) THEN e_adm ELSE 0 END) AS adm_sup,
        sum(CASE WHEN sal_real IS NOT NULL AND sal_real <= l.p25 THEN e_adm ELSE 0 END) AS adm_base,
        sum(CASE WHEN aprendiz IN ({_lista(c['aprendiz'])}) THEN e_adm ELSE 0 END) AS adm_aprendiz,
        sum(sal_real) AS sal_soma, sum(ln(sal_real)) AS sal_log_soma, count(sal_real) AS sal_n"""


def _pandemia(cfg, con) -> pd.DataFrame | None:
    """Choque e boom da pandemia por família CBO: [ln jovens(janela) − ln jovens(pré)] − [o mesmo p/ adultos]."""
    p = cfg["pandemia"]
    jov = ",".join(f"'{x}'" for x in cfg["idade"]["jovem"])
    adu = ",".join(f"'{x}'" for x in cfg["idade"]["adulto"])

    def janela(nome, a, b):
        return f"""sum(CASE WHEN comp BETWEEN {a} AND {b} AND faixa IN ({jov}) THEN e_adm ELSE 0 END) AS j_{nome},
                   sum(CASE WHEN comp BETWEEN {a} AND {b} AND faixa IN ({adu}) THEN e_adm ELSE 0 END) AS a_{nome}"""
    d = con.sql(f"""SELECT cbo4, {janela('pre', *p['pre'])}, {janela('chq', *p['choque'])},
                           {janela('boom', *p['boom'])}
                    FROM base GROUP BY 1""").df()
    if d[["j_pre", "a_pre"]].sum().sum() == 0:
        log.warning("sem dados de jan–fev/2020: controles da pandemia não serão construídos")
        return None
    # meses por janela (o choque tem 3 meses e o pré 2): a escala some na diferença jovem − adulto
    lg = lambda x: np.log(x.clip(lower=0) + 1)
    d["choque"] = (lg(d.j_chq) - lg(d.j_pre)) - (lg(d.a_chq) - lg(d.a_pre))
    d["boom"] = (lg(d.j_boom) - lg(d.j_pre)) - (lg(d.a_boom) - lg(d.a_pre))
    w = d.j_pre + d.a_pre
    d = d[w >= 20].copy()  # famílias pequenas demais dão medidas só de ruído
    w = d.j_pre + d.a_pre
    for v in ("choque", "boom"):
        mu = np.average(d[v], weights=w)
        sd = np.sqrt(np.average((d[v] - mu) ** 2, weights=w))
        d[f"{v}_z"] = (d[v] - mu) / sd
    d["peso_pre"] = w
    return d


def construir(cfg, nome="principal", **filtros) -> pd.DataFrame:
    con = conexao(cfg)
    c = _base(cfg, con, **filtros)
    ag = _agregados(c)
    df = con.sql(f"""
        SELECT t, ano, tri, b.cbo4, faixa, uf, {ag}
        FROM base b LEFT JOIN limiar l ON l.cbo4 = b.cbo4
        WHERE faixa IS NOT NULL GROUP BY ALL""").df()
    df.to_parquet(caminho(cfg, "processed", f"painel_{nome}.parquet"), index=False)
    log.info("painel %s: %d células, %d admissões, trimestres %s a %s", nome, len(df), int(df.adm.sum()),
             df.t.min(), df.t.max())
    if nome != "principal":
        return df
    pand = _pandemia(cfg, con)
    if pand is not None:
        pand.to_parquet(caminho(cfg, "processed", "pandemia_cbo4.parquet"), index=False)
        try:  # correlação com a exposição: quão separáveis são IA e pandemia?
            ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"))
            m = pand.merge(ex[["cbo4", "expo_z"]], on="cbo4")
            w = m.peso_pre
            corr = {}
            for v in ("choque_z", "boom_z"):
                cov = np.cov(m.expo_z, m[v], aweights=w)
                corr[v] = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            pd.DataFrame([corr]).to_csv(caminho(cfg, "outputs", "validacao", "correlacao_expo_pandemia.csv"), index=False)
            log.info("correlação (ponderada) exposição × pandemia: %s", {k: round(v, 3) for k, v in corr.items()})
        except FileNotFoundError:
            log.warning("rode a etapa exposicao antes do painel para ver a correlação com a pandemia")
    dims = {
        "sexo": "CASE WHEN sexo IN ({}) THEN 'mulher' ELSE 'homem' END".format(_lista(c["sexo_mulher"])),
        "raca": "CASE WHEN raca IN ({}) THEN 'negra' WHEN raca IN ({}) THEN 'branca' ELSE 'outra_ni' END".format(
            _lista(c["raca_negra"]), _lista(c["raca_branca"])),
        "instrucao": "CASE WHEN instr IN ({}) THEN 'superior' WHEN instr IN ({}) THEN 'medio' "
                     "ELSE 'outro' END".format(_lista(c["superior"]), _lista(c["medio_completo"])),
        "regiao": "CAST(regiao AS VARCHAR)",
        "nordeste": f"CASE WHEN regiao = {int(c['regiao_nordeste'])} THEN 'NE' ELSE 'resto' END",
        "rn": f"CASE WHEN uf = {int(c['uf_rn'])} THEN 'RN' ELSE 'resto' END",
    }
    partes = [f"""SELECT t, ano, tri, b.cbo4, faixa, '{d}' AS dim, {expr} AS valor, {ag}
                  FROM base b LEFT JOIN limiar l ON l.cbo4 = b.cbo4
                  WHERE faixa IS NOT NULL GROUP BY ALL""" for d, expr in dims.items()]
    het = con.sql(" UNION ALL ".join(partes)).df()
    het.to_parquet(caminho(cfg, "processed", "painel_heterogeneidade.parquet"), index=False)
    log.info("painel heterogeneidade: %d células", len(het))
    return df
