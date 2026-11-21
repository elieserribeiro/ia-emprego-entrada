"""Vagas de jovens que deixaram de ser abertas nas ocupações mais expostas à IA (contrafactual).

Leitura do modelo PPML: admissões observadas = admissões sem o efeito × exp(beta × expo_z).
Logo, vagas não abertas na célula = observadas × (exp(-beta × expo_z) - 1).

Comparação escolhida: média de exposição. Só entram as ocupações ACIMA da média (expo_z > 0),
e o contrafactual é "se essa ocupação tivesse a exposição média". Os coeficientes vêm do event
study (beta por trimestre) e do pooled (beta único, 2023T1 em diante), ambos com controles da
pandemia. O intervalo usa os limites do IC 95% de cada beta (aproximação).

Atenção na leitura: o efeito é relativo aos adultos (30+) da mesma ocupação. "Vagas não abertas"
quer dizer vagas de jovens que faltam em relação ao que se esperaria pela evolução dos adultos,
e não postos de trabalho destruídos.

Uso: python -m iaemprego vagas   (depois da etapa "modelos")
Saídas: outputs/tabelas/vagas_nao_abertas_trimestre.csv e vagas_nao_abertas_resumo.csv
"""
import logging

import numpy as np
import pandas as pd

from .util import caminho

log = logging.getLogger(__name__)

MODELOS = {
    "25_29": ("tripla_25_29_event_com_controles", "tripla_25_29_pooled_com_controles"),
    "18_24": ("tripla_event_com_controles", "tripla_pooled_com_controles"),
}
INICIO_POS = 20231


def rodar(cfg) -> pd.DataFrame:
    painel = pd.read_parquet(caminho(cfg, "processed", "painel_principal.parquet"), columns=["t", "cbo4", "faixa", "adm"])
    ex = pd.read_parquet(caminho(cfg, "processed", "exposicao_cbo4.parquet"), columns=["cbo4", "expo_z"])
    res = pd.read_csv(caminho(cfg, "outputs", "tabelas", "modelos_principais.csv"))
    res = res[res.desfecho == "adm"]

    trimestrais, resumo = [], []
    for faixa, (m_ev, m_po) in MODELOS.items():
        d = painel[(painel.faixa == faixa) & (painel.t >= INICIO_POS)]
        d = d.groupby(["t", "cbo4"], as_index=False).adm.sum().merge(ex, on="cbo4", how="inner")
        d = d[d.expo_z > 0]  # só ocupações acima da média de exposição

        ev = res[res.modelo == m_ev].dropna(subset=["periodo"]).copy()
        ev["t"] = ev.periodo.astype(int)
        po = res[(res.modelo == m_po) & (res.termo == "treat_post")].iloc[0]

        x = d.merge(ev[["t", "coef", "ic_inf", "ic_sup"]], on="t", how="inner")
        for nome, b in (("central", "coef"), ("ic_menor", "ic_sup"), ("ic_maior", "ic_inf")):
            x[nome] = x.adm * (np.exp(-x[b] * x.expo_z) - 1)
        for nome, b in (("pooled_central", po.coef), ("pooled_ic_menor", po.ic_sup), ("pooled_ic_maior", po.ic_inf)):
            x[nome] = x.adm * (np.exp(-b * x.expo_z) - 1)

        cols = ["adm", "central", "ic_menor", "ic_maior", "pooled_central", "pooled_ic_menor", "pooled_ic_maior"]
        q = x.groupby("t", as_index=False)[cols].sum()
        q.insert(0, "faixa", faixa)
        q["pct_das_adm"] = 100 * q.central / (q.adm + q.central)
        trimestrais.append(q)

        tot = q[cols].sum()
        anos = q.t.nunique() / 4
        resumo.append(dict(
            faixa=faixa, trimestres=q.t.nunique(), primeiro=q.t.min(), ultimo=q.t.max(),
            adm_observadas=tot.adm,
            nao_abertas_event=tot.central, ic_menor_event=tot.ic_menor, ic_maior_event=tot.ic_maior,
            nao_abertas_pooled=tot.pooled_central, ic_menor_pooled=tot.pooled_ic_menor, ic_maior_pooled=tot.pooled_ic_maior,
            media_anual_event=tot.central / anos, pct_das_adm=100 * tot.central / (tot.adm + tot.central),
            ocupacoes_acima_media=d.cbo4.nunique(),
        ))

    tri = pd.concat(trimestrais)
    res_df = pd.DataFrame(resumo)
    tri.to_csv(caminho(cfg, "outputs", "tabelas", "vagas_nao_abertas_trimestre.csv"), index=False)
    res_df.to_csv(caminho(cfg, "outputs", "tabelas", "vagas_nao_abertas_resumo.csv"), index=False)
    for r in res_df.itertuples():
        log.info("vagas não abertas %s: %.0f (IC %.0f a %.0f), %.1f%% das admissões",
                 r.faixa, r.nao_abertas_event, r.ic_menor_event, r.ic_maior_event, r.pct_das_adm)
    return res_df
