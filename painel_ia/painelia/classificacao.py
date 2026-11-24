"""Constrói `artefatos/ocupacoes.json`: o que o painel mostra para cada CBO (exposição, não impacto).

Camada 1 — Exposição (alta / média / baixa), pelos gradientes oficiais da OIT 2025 (Gmyrek et al., WP 140,
           Tabela 5), calculados com média μ e desvio-padrão σ dos escores das tarefas:
             Gradiente 4: μ ≥ 0,6 e μ − σ ≥ 0,5       Gradiente 3: 0,5 ≤ μ < 0,6 e μ + σ ≥ 0,5
             Gradiente 2: 0,4 ≤ μ < 0,5 e μ + σ ≥ 0,5  Gradiente 1: μ < 0,4 e μ + σ ≥ 0,5
             Mínima: μ < 0,5 e μ + σ > 0,4             Não exposta: o resto
           alta = gradientes 3–4; média = gradientes 1–2; baixa = mínima ou não exposta.
           Mais o percentil da ocupação entre as CBOs (ponderado por admissões, quando houver os dados do CAGED).
Camada 2 — Natureza provável: substituição de tarefas × transformação × complementaridade, pela média,
           dispersão e parcela de tarefas com escore ≥ 0,5.
Camada 3 — Evidência brasileira (CAGED): ver evidencia_br.py; anexada aqui se existir.

As tarefas vêm da ISCO-08 correspondente à CBO pela ponte do artigo (data/ref/ponte_cbo2002_isco08_ilo2025.csv).
Quando a ponte só chega a 3 ou 2 dígitos, usa as tarefas de todas as ocupações ISCO daquele grupo; quando a CBO
não tem ISCO, usa as tarefas das outras CBOs da mesma família (4, 3, 2 dígitos da CBO).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import textos

RAIZ = Path(__file__).resolve().parents[1]
REPO = RAIZ.parent
ART = RAIZ / "artefatos"

GRADIENTE_PT = {
    "g4": "Gradiente 4 (exposição mais alta, pouca variação entre tarefas)",
    "g3": "Gradiente 3 (exposição significativa, alta variação entre tarefas)",
    "g2": "Gradiente 2 (exposição moderada, alta variação entre tarefas)",
    "g1": "Gradiente 1 (exposição baixa, mas algumas tarefas muito expostas)",
    "minima": "Exposição mínima",
    "nao_exposta": "Não exposta",
}
NIVEL = {"g4": "alta", "g3": "alta", "g2": "media", "g1": "media", "minima": "baixa", "nao_exposta": "baixa"}


def gradiente(mu: float, sd: float) -> str:
    """Regra oficial da OIT (WP 140, Tabela 5)."""
    sd = 0.0 if pd.isna(sd) else sd
    if mu >= 0.6 and mu - sd >= 0.5:
        return "g4"
    if 0.5 <= mu < 0.6 and mu + sd >= 0.5:
        return "g3"
    if 0.4 <= mu < 0.5 and mu + sd >= 0.5:
        return "g2"
    if mu < 0.4 and mu + sd >= 0.5:
        return "g1"
    if mu < 0.5 and mu + sd > 0.4:
        return "minima"
    return "nao_exposta"


def natureza(mu: float, sd: float, frac_alta: float, grad: str) -> dict:
    if grad == "g4":
        return {"tipo": "substituicao",
                "rotulo": "Tende à substituição de tarefas",
                "texto": "A maior parte das tarefas atuais desta ocupação pode, tecnicamente, ser feita por IA "
                         "generativa, com pouca variação entre elas. É o perfil em que a reorganização do trabalho "
                         "tende a ser maior: menos tempo nas tarefas rotineiras de texto e informação, e mais "
                         "supervisão, revisão e relação com pessoas."}
    if grad == "g3" or (grad == "g2" and frac_alta >= 0.4):
        return {"tipo": "transformacao",
                "rotulo": "Tende à transformação",
                "texto": "Uma parte grande das tarefas é exposta, mas outra parte continua dependendo de julgamento, "
                         "presença ou contato humano. O mais provável é a ocupação mudar de conteúdo: a IA assume "
                         "partes do trabalho e quem ocupa o cargo passa a combinar as duas coisas."}
    if grad in ("g2", "g1") or (grad == "minima" and frac_alta > 0):
        return {"tipo": "complementaridade",
                "rotulo": "Tende à complementaridade",
                "texto": "Apenas algumas tarefas são expostas. A IA tende a funcionar como apoio (redigir, resumir, "
                         "pesquisar, organizar), enquanto o núcleo da ocupação segue humano."}
    return {"tipo": "pouco_afetada",
            "rotulo": "Pouco afetada pela IA generativa",
            "texto": "As tarefas desta ocupação dependem sobretudo de trabalho físico, presencial ou interpessoal, "
                     "que a IA generativa (que trabalha com texto, imagem e informação) faz pouco hoje."}


def _aplicar_ajustes(ponte: pd.DataFrame) -> pd.DataFrame:
    """Ajustes manuais da ponte, só no painel, para casos em que a tábua oficial leva a tarefas que a pessoa não
    reconheceria (ver dados/ajustes_ponte.csv, com o motivo de cada um)."""
    arq = RAIZ / "dados" / "ajustes_ponte.csv"
    if not arq.exists():
        return ponte
    ponte = ponte.copy()
    for a in pd.read_csv(arq, dtype=str).itertuples():
        m = ponte.cbo2002.str.startswith(a.cbo_prefixo)
        ponte.loc[m, "isco08"], ponte.loc[m, "ilo_nivel"] = a.isco08, "isco4"
    return ponte


def _traducoes() -> dict:
    """Tarefas da OIT traduzidas (dados/tarefas_pt.csv, gerado uma vez por painelia.traduzir_tarefas)."""
    arq = RAIZ / "dados" / "tarefas_pt.csv"
    if not arq.exists():
        return {}
    t = pd.read_csv(arq, dtype=str)
    return {(r.isco08, str(r.tarefa_id)): r.tarefa_pt for r in t.itertuples()}


def _listas_tarefas(g: pd.DataFrame, pt: dict) -> tuple[list, list]:
    """Mais e menos expostas, sem repetir tarefas quando a ocupação tem poucas."""
    g = g.sort_values("escore")
    n = len(g)
    k = min(3, max(1, n // 2))
    def fmt(df):
        return [dict(tarefa=t.tarefa, tarefa_pt=pt.get((t.isco08, str(t.tarefa_id))), escore=round(t.escore, 2))
                for t in df.itertuples()]
    return fmt(g.tail(k)[::-1]), fmt(g.head(min(k, n - k)))


def _tarefas_por_cbo(ponte: pd.DataFrame, tarefas: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    por4 = {k: g for k, g in tarefas.groupby("isco08")}
    res: dict[str, tuple[pd.DataFrame, str]] = {}
    for r in ponte.itertuples():
        cod = str(r.isco08) if pd.notna(r.isco08) else ""
        nivel = r.ilo_nivel if isinstance(r.ilo_nivel, str) else ""
        if cod and nivel == "isco4" and cod in por4:
            res[r.cbo2002] = (por4[cod], f"ISCO-08 {cod}")
        elif cod and nivel in ("isco3", "isco2"):
            k = 3 if nivel == "isco3" else 2
            g = tarefas[tarefas.isco08.str[:k] == cod[:k]]
            if len(g):
                res[r.cbo2002] = (g, f"grupo ISCO-08 {cod[:k]}")
    # sem ISCO: tarefas das CBOs vizinhas na hierarquia
    for r in ponte.itertuples():
        if r.cbo2002 in res:
            continue
        for k in (4, 3, 2):
            viz = [res[c][0] for c in ponte.cbo2002 if c[:k] == r.cbo2002[:k] and c in res]
            if viz:
                res[r.cbo2002] = (pd.concat(viz).drop_duplicates(["isco08", "tarefa_id"]), f"família CBO {r.cbo2002[:k]} (imputado)")
                break
    return res


def _pesos_admissoes() -> pd.Series | None:
    p = REPO / "data" / "processed" / "exposicao_cbo6.parquet"
    if p.exists():
        e = pd.read_parquet(p)
        return e.set_index("cbo6").adm_ref
    return None


def construir(pasta_mte: Path | None = None, saida: Path | None = None) -> dict:
    ponte = pd.read_csv(REPO / "data" / "ref" / "ponte_cbo2002_isco08_ilo2025.csv", dtype={"cbo2002": str, "isco08": str})
    ponte["cbo2002"] = ponte.cbo2002.str.zfill(6)
    ponte = _aplicar_ajustes(ponte)
    pt = _traducoes()
    tarefas = pd.read_csv(RAIZ / "dados" / "ilo2025_tarefas_isco08.csv", dtype={"isco08": str})
    base = textos.carregar(pasta_mte)
    tit = textos.titulos(base)
    por_cbo = _tarefas_por_cbo(ponte, tarefas)
    linhas = []
    for cbo, (g, fonte) in por_cbo.items():
        s = g.escore.to_numpy()
        mu, sd = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0
        grad = gradiente(mu, sd)
        mais, menos = _listas_tarefas(g, pt)
        linhas.append(dict(
            cbo=cbo, titulo=tit.get(cbo), fonte_tarefas=fonte,
            tarefas_de_ocupacoes_vizinhas=not fonte.startswith("ISCO-08 "), media=round(mu, 3), dp=round(sd, 3),
            frac_tarefas_alta=round(float((s >= 0.5).mean()), 3), n_tarefas=len(s), gradiente=grad,
            gradiente_rotulo=GRADIENTE_PT[grad], exposicao=NIVEL[grad],
            natureza=natureza(mu, sd, float((s >= 0.5).mean()), grad),
            tarefas_mais_expostas=mais, tarefas_menos_expostas=menos,
        ))
    df = pd.DataFrame(linhas)
    w = _pesos_admissoes()
    df["peso"] = df.cbo.map(w).fillna(0) if w is not None else 1.0
    if df.peso.sum() == 0:
        df["peso"] = 1.0
    o = df.sort_values("media")
    acum = o.peso.cumsum() / o.peso.sum()
    df.loc[o.index, "percentil"] = (100 * (acum - o.peso / o.peso.sum() / 2)).round().astype(int)
    df["percentil_base"] = "admissões no CAGED (jan–set/2022)" if w is not None else "CBOs (sem ponderação)"

    # distribuição das classes: para conferir antes de fixar os cortes
    dist = (df.groupby(["exposicao", "gradiente"]).agg(n_cbos=("cbo", "size"), peso=("peso", "sum")).reset_index())
    dist["pct_cbos"] = (100 * dist.n_cbos / dist.n_cbos.sum()).round(1)
    dist["pct_ponderado"] = (100 * dist.peso / dist.peso.sum()).round(1)
    ART.mkdir(exist_ok=True)
    dist.drop(columns="peso").to_csv(ART / "distribuicao_exposicao.csv", index=False)

    try:
        from . import evidencia_br
        ev = evidencia_br.carregar()
    except FileNotFoundError:
        ev = None
    out = {"versao": 1, "fonte": "OIT 2025 (Gmyrek et al., WP 140, CC BY 4.0); ponte CBO→ISCO-08 do artigo",
           "ocupacoes": {}}
    for r in df.drop(columns="peso").to_dict("records"):
        if ev is not None:
            r["evidencia_br"] = ev.get(r["cbo"][:4])
        out["ocupacoes"][r["cbo"]] = r
    if ev is not None:
        out["evidencia_geral"] = ev.get("_geral")
    (saida or ART / "ocupacoes.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = construir()
    print(f"{len(o['ocupacoes'])} ocupações; distribuição em {ART / 'distribuicao_exposicao.csv'}")
    print(pd.read_csv(ART / "distribuicao_exposicao.csv").to_string(index=False))
