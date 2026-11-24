"""Camada 3 — o que os dados brasileiros mostram (Novo CAGED), por família CBO e perfil.

Roda DEPOIS do pipeline do artigo (usa data/processed/painel_*.parquet e outputs/tabelas/modelos_principais.csv).
Para cada família CBO (4 dígitos) × grupo de idade (18–24, 25–29, 30+), e também × escolaridade:
  variação das admissões entre os 4 trimestres até 2022T3 (antes do ChatGPT) e os 4 trimestres mais recentes,
  descontada a variação do mercado inteiro para o mesmo grupo de idade.
  Classe: "queda" | "estável" | "alta" | "dados insuficientes", com nível de confiança.
É descritivo (não causal): mostra como as contratações daquele perfil andaram, não o efeito da IA. O efeito
estimado no artigo (diferença tripla) vai no campo `_geral`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
REPO = RAIZ.parent
SAIDA = RAIZ / "artefatos" / "evidencia_br.json"
MIN_ADM = 100        # admissões mínimas em cada janela para mostrar a evidência
LIMIAR = 0.10        # |variação relativa| abaixo de 10% = estável


def _grupo(faixa: str) -> str | None:
    return {"18_24": "18-24", "25_29": "25-29", "30_39": "30+", "40_49": "30+", "50mais": "30+"}.get(faixa)


def _janelas(ts: list[int], ref: int = 20223) -> tuple[list[int], list[int]]:
    ts = sorted(ts)
    i = ts.index(ref)
    return ts[max(0, i - 3): i + 1], ts[-4:]


def _classe(a0, a1, r_mkt):
    if a0 < MIN_ADM or a1 < MIN_ADM:
        return {"classe": "dados insuficientes"}
    r = np.log(a1 / a0) - r_mkt
    ep = np.sqrt(1 / a0 + 1 / a1) * 2.0  # ×2: sobredispersão (células de contratação variam mais que Poisson)
    var = float(np.expm1(r))
    if abs(var) < LIMIAR or abs(r) < 1.96 * ep:
        cl = "estável"
    else:
        cl = "queda" if r < 0 else "alta"
    conf = "alta" if (abs(r) / ep > 4 and min(a0, a1) >= 1000) else ("moderada" if abs(r) / ep > 2 else "baixa")
    return {"classe": cl, "variacao_relativa_pct": round(100 * var, 1), "confianca": conf,
            "adm_antes": int(a0), "adm_recente": int(a1)}


def construir() -> dict:
    pp = REPO / "data" / "processed" / "painel_principal.parquet"
    ph = REPO / "data" / "processed" / "painel_heterogeneidade.parquet"
    p = pd.read_parquet(pp, columns=["t", "cbo4", "faixa", "adm"])
    p["grupo"] = p.faixa.map(_grupo)
    p = p.dropna(subset=["grupo"])
    antes, recente = _janelas(p.t.unique().tolist())
    rot = {"antes": f"{antes[0] // 10}T{antes[0] % 10}–{antes[-1] // 10}T{antes[-1] % 10}",
           "recente": f"{recente[0] // 10}T{recente[0] % 10}–{recente[-1] // 10}T{recente[-1] % 10}"}

    def resumo(df, chaves):
        df = df.assign(j=np.where(df.t.isin(antes), "a0", np.where(df.t.isin(recente), "a1", None))).dropna(subset=["j"])
        g = df.groupby(chaves + ["j"]).adm.sum().unstack("j").fillna(0)
        mk = df.groupby(["grupo", "j"]).adm.sum().unstack("j")
        r_mkt = np.log(mk.a1 / mk.a0)
        return g, r_mkt

    out: dict = {}
    g, r_mkt = resumo(p, ["cbo4", "grupo"])
    for (c, gr), v in g.iterrows():
        out.setdefault(c, {"janelas": rot, "por_idade": {}, "por_idade_escolaridade": {}})
        out[c]["por_idade"][gr] = _classe(v.a0, v.a1, r_mkt[gr])
    if ph.exists():
        h = pd.read_parquet(ph, columns=["t", "cbo4", "faixa", "dim", "valor", "adm"])
        h = h[h.dim == "instrucao"].assign(grupo=lambda d: d.faixa.map(_grupo)).dropna(subset=["grupo"])
        g, r_mkt = resumo(h, ["cbo4", "grupo", "valor"])
        for (c, gr, esc), v in g.iterrows():
            if c in out:
                out[c]["por_idade_escolaridade"].setdefault(gr, {})[esc] = _classe(v.a0, v.a1, r_mkt[gr])
    tab = REPO / "outputs" / "tabelas"
    mp = tab / "modelos_principais.csv"
    if mp.exists():
        # efeitos do artigo (tripla diferença, pooled, com controles), na convenção do artigo: 100 × coeficiente
        m = pd.read_csv(mp)
        geral: dict = {"por_faixa": {}}
        for faixa, modelo in (("18-24", "tripla_pooled_com_controles"), ("25-29", "tripla_25_29_pooled_com_controles")):
            f = m[(m.modelo == modelo) & (m.termo == "treat_post")].set_index("desfecho")
            geral["por_faixa"][faixa] = {y: {"coef": round(float(r.coef), 4), "ep": round(float(r.ep), 4),
                                             "p": round(float(r.p), 4), "efeito_pct_por_dp": round(100 * float(r.coef), 1)}
                                         for y, r in f.iterrows()}
        vg = tab / "vagas_nao_abertas_resumo.csv"
        if vg.exists():
            v = pd.read_csv(vg).set_index("faixa")
            if "25_29" in v.index:
                r = v.loc["25_29"]
                geral["vagas_25_29"] = {"total": int(round(r.nao_abertas_event, -3)), "ic_menor": int(round(r.ic_menor_event, -3)),
                                        "ic_maior": int(round(r.ic_maior_event, -3)), "ate": str(int(r.ultimo)),
                                        "pct_das_adm": round(float(r.pct_das_adm), 1)}
        out["_geral"] = geral
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def carregar() -> dict:
    if not SAIDA.exists():
        raise FileNotFoundError(SAIDA)
    return json.loads(SAIDA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    o = construir()
    print(f"evidência para {len(o) - ('_geral' in o)} famílias CBO -> {SAIDA}")
