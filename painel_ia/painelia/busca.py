"""Buscador de CBO a partir de texto livre (cargo + tarefas).

Dois motores com a mesma interface:
  "st"    — sentence-transformers (embeddings multilíngues ajustados no treino); o principal.
  "tfidf" — n-gramas de caracteres; linha de base da avaliação e reserva se o modelo não carregar.
O índice (um vetor por CBO) fica em artefatos/indice_<motor>/.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .textos import normalizar

RAIZ = Path(__file__).resolve().parents[1]


class BuscadorTfidf:
    nome = "tfidf"

    def __init__(self, vetorizador=None, matriz=None, cbos=None):
        self.vet, self.mat, self.cbos = vetorizador, matriz, cbos

    @classmethod
    def treinar(cls, docs_cbo: list[str], textos: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vet = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
                              preprocessor=normalizar)
        mat = vet.fit_transform(textos)
        return cls(vet, mat, np.asarray(docs_cbo))

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[str, float]]:
        q = self.vet.transform([consulta])
        s = (self.mat @ q.T).toarray().ravel()
        idx = np.argsort(-s)[:k]
        return [(str(self.cbos[i]), float(s[i])) for i in idx]

    def salvar(self, pasta: Path):
        pasta.mkdir(parents=True, exist_ok=True)
        with open(pasta / "tfidf.pkl", "wb") as f:
            pickle.dump((self.vet, self.mat, self.cbos), f)

    @classmethod
    def carregar(cls, pasta: Path):
        with open(pasta / "tfidf.pkl", "rb") as f:
            return cls(*pickle.load(f))


class BuscadorST:
    nome = "st"

    def __init__(self, modelo, emb, cbos, prefixo_consulta="query: "):
        self.modelo, self.emb, self.cbos, self.pq = modelo, emb, np.asarray(cbos), prefixo_consulta

    @classmethod
    def treinar(cls, modelo, docs_cbo, textos, prefixo_consulta="query: "):
        emb = modelo.encode(textos, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return cls(modelo, emb, docs_cbo, prefixo_consulta)

    def buscar(self, consulta: str, k: int = 3):
        q = self.modelo.encode([self.pq + consulta], normalize_embeddings=True)[0]
        s = self.emb @ q
        idx = np.argsort(-s)[:k]
        return [(str(self.cbos[i]), float(s[i])) for i in idx]

    def salvar(self, pasta: Path):
        pasta.mkdir(parents=True, exist_ok=True)
        self.modelo.save(str(pasta / "modelo"))
        np.save(pasta / "emb.npy", self.emb.astype(np.float32))
        (pasta / "meta.json").write_text(json.dumps({"cbos": list(map(str, self.cbos)), "prefixo_consulta": self.pq}))

    @classmethod
    def carregar(cls, pasta: Path):
        from sentence_transformers import SentenceTransformer
        meta = json.loads((pasta / "meta.json").read_text())
        return cls(SentenceTransformer(str(pasta / "modelo")), np.load(pasta / "emb.npy"), meta["cbos"],
                   meta["prefixo_consulta"])


def carregar(pasta_artefatos: Path | None = None, motor: str | None = None):
    """Carrega o melhor motor disponível: st, se houver o índice e a biblioteca; senão tfidf."""
    a = Path(pasta_artefatos or RAIZ / "artefatos")
    ordem = [motor] if motor else ["st", "tfidf"]
    erros = []
    for m in ordem:
        try:
            return (BuscadorST if m == "st" else BuscadorTfidf).carregar(a / f"indice_{m}")
        except Exception as e:  # biblioteca ausente ou índice não treinado
            erros.append(f"{m}: {e}")
    raise RuntimeError("nenhum índice de busca disponível; rode `python -m painelia.treinar`. " + " | ".join(erros))
