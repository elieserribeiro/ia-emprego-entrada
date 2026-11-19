"""Treina e avalia o buscador de CBO.

    python -m painelia.treinar                      # TF-IDF + ajuste fino de embeddings (precisa de requirements-treino)
    python -m painelia.treinar --so-tfidf           # só a linha de base (rápido, sem GPU nem download)
    python -m painelia.treinar --modelo-base intfloat/multilingual-e5-small --epocas 2

Desenho:
- Documento de cada CBO: título oficial + família + sinônimos + atividades (textos.documentos).
- Pares de treino (consulta → CBO): sinônimos e atividades oficiais. 20% dos sinônimos de cada CBO com ≥ 2
  sinônimos ficam FORA do treino e dos documentos, e são a validação ("nome popular do cargo").
- Segunda validação, mais realista: descrições escritas como um usuário escreveria (dados/consultas_teste.csv,
  cargo + tarefas), com a resposta identificada pelo título oficial. Amplie com exemplos seus.
- Métricas: acerto no top-1 e no top-3, por ocupação (6 dígitos) e por família (4 dígitos).
- Ajuste fino: MultipleNegativesRankingLoss (contraste dentro do lote), sentence-transformers.
- Ao final, o índice é refeito com TODOS os sinônimos e salvo em artefatos/indice_<motor>/.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import busca, textos

RAIZ = Path(__file__).resolve().parents[1]
ART = RAIZ / "artefatos"


def dividir(base: pd.DataFrame, frac=0.2, semente=0):
    rng = random.Random(semente)
    sin = base[base.tipo == "sinonimo"]
    val_idx = []
    for _, g in sin.groupby("cbo6"):
        if len(g) >= 2:
            n = max(1, int(round(frac * len(g))))
            val_idx += rng.sample(list(g.index), n)
    val = base.loc[val_idx]
    return base.drop(index=val_idx), val


def consultas_teste(base: pd.DataFrame) -> pd.DataFrame:
    p = RAIZ / "dados" / "consultas_teste.csv"
    if not p.exists():
        return pd.DataFrame(columns=["consulta", "cbo6"])
    q = pd.read_csv(p)
    idx = {}
    for r in base[base.tipo.isin(["titulo", "sinonimo"])].itertuples():
        idx.setdefault(textos.normalizar(r.texto), r.cbo6)
    q["cbo6"] = q.titulo_esperado.map(lambda t: idx.get(textos.normalizar(t)))
    falt = q[q.cbo6.isna()]
    if len(falt):
        print(f"[aviso] {len(falt)} consultas de teste sem título correspondente na base (ignoradas): "
              f"{falt.titulo_esperado.tolist()[:5]}")
    return q.dropna(subset=["cbo6"])


def avaliar(buscador, consultas: pd.DataFrame, nome: str) -> dict:
    if consultas.empty:
        return {}
    a1 = a3 = f1 = f3 = 0
    for r in consultas.itertuples():
        top = [c for c, _ in buscador.buscar(r.consulta, k=3)]
        a1 += top[0] == r.cbo6
        a3 += r.cbo6 in top
        f1 += top[0][:4] == r.cbo6[:4]
        f3 += r.cbo6[:4] in [c[:4] for c in top]
    n = len(consultas)
    res = {"conjunto": nome, "n": n, "top1": a1 / n, "top3": a3 / n, "top1_familia": f1 / n, "top3_familia": f3 / n}
    print(f"  {buscador.nome:5s} | {nome:22s} n={n:5d}  top1={res['top1']:.3f}  top3={res['top3']:.3f}  "
          f"top3(família)={res['top3_familia']:.3f}")
    return res


def _tfidf(base):
    docs = textos.documentos(base)
    return busca.BuscadorTfidf.treinar(docs.cbo6.tolist(), docs.documento.tolist())


def _st(base_treino, modelo_base, epocas, lote, pd_prefix, pq_prefix):
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ajuste fino de {modelo_base} em {disp} ({epocas} época(s))")
    modelo = SentenceTransformer(modelo_base, device=disp)
    docs = textos.documentos(base_treino, prefixo=pd_prefix)
    doc = dict(zip(docs.cbo6, docs.documento))
    pares = base_treino[base_treino.tipo.isin(["sinonimo", "atividade", "titulo"]) & base_treino.cbo6.isin(doc)]
    ex = [InputExample(texts=[pq_prefix + r.texto, doc[r.cbo6]]) for r in pares.itertuples()]
    print(f"  {len(ex)} pares de treino")
    dl = DataLoader(ex, shuffle=True, batch_size=lote)
    perda = losses.MultipleNegativesRankingLoss(modelo)
    modelo.fit(train_objectives=[(dl, perda)], epochs=epocas, warmup_steps=int(0.1 * len(dl) * epocas),
               show_progress_bar=True)
    return modelo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so-tfidf", action="store_true")
    ap.add_argument("--modelo-base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--pasta-mte", default=None)
    a = ap.parse_args(argv)
    e5 = "e5" in a.modelo_base
    pq, pdoc = ("query: ", "passage: ") if e5 else ("", "")

    base = textos.carregar(a.pasta_mte)
    print(f"base: {base.cbo6.nunique()} CBOs; textos por tipo: {base.tipo.value_counts().to_dict()}")
    treino, val = dividir(base)
    q_val = val.rename(columns={"texto": "consulta"})[["consulta", "cbo6"]]
    q_teste = consultas_teste(base)
    relatorio = []

    print("Avaliação (índice sem os sinônimos de validação):")
    tf = _tfidf(treino)
    relatorio += [avaliar(tf, q_val, "sinônimos retidos"), avaliar(tf, q_teste, "consultas de usuário")]

    if not a.so_tfidf:
        from sentence_transformers import SentenceTransformer
        docs = textos.documentos(treino, prefixo=pdoc)
        zero = busca.BuscadorST.treinar(SentenceTransformer(a.modelo_base), docs.cbo6.tolist(),
                                        docs.documento.tolist(), pq)
        zero.nome = "st-0"
        relatorio += [avaliar(zero, q_val, "sinônimos retidos"), avaliar(zero, q_teste, "consultas de usuário")]
        modelo = _st(treino, a.modelo_base, a.epocas, a.lote, pdoc, pq)
        ajust = busca.BuscadorST.treinar(modelo, docs.cbo6.tolist(), docs.documento.tolist(), pq)
        relatorio += [avaliar(ajust, q_val, "sinônimos retidos"), avaliar(ajust, q_teste, "consultas de usuário")]

    # índices finais com a base completa
    ART.mkdir(exist_ok=True)
    _tfidf(base).salvar(ART / "indice_tfidf")
    if not a.so_tfidf:
        docs = textos.documentos(base, prefixo=pdoc)
        busca.BuscadorST.treinar(modelo, docs.cbo6.tolist(), docs.documento.tolist(), pq).salvar(ART / "indice_st")
    titulos = textos.titulos(base)
    (ART / "titulos_cbo.json").write_text(json.dumps(titulos, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([r for r in relatorio if r]).to_csv(ART / "avaliacao_buscador.csv", index=False)
    print(f"índices e avaliação em {ART}")


if __name__ == "__main__":
    main()
