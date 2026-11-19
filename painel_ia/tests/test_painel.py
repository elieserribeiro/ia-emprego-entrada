"""Testes do painel (sem rede, sem modelo de embeddings: usa o índice TF-IDF)."""
from __future__ import annotations

import importlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from painelia import classificacao


def test_regra_gradientes_oit():
    g = classificacao.gradiente
    assert g(0.64, 0.07) == "g4"
    assert g(0.55, 0.10) == "g3"
    assert g(0.45, 0.10) == "g2"
    assert g(0.35, 0.16) == "g1"
    assert g(0.34, 0.10) == "minima"
    assert g(0.19, 0.07) == "nao_exposta"
    assert classificacao.NIVEL["g3"] == "alta" and classificacao.NIVEL["g1"] == "media"


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    from painelia import treinar
    treinar.main(["--so-tfidf"])
    classificacao.construir()
    db = tmp_path / "r.db"
    monkeypatch.setenv("PAINEL_SQLITE", str(db))
    monkeypatch.setenv("ADMIN_TOKEN", "segredo")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from painelia import api
    importlib.reload(api)
    return TestClient(api.app), db


PERFIL = {"faixa_etaria": "18-24", "escolaridade": "superior", "uf": "RN", "tempo_funcao": "menos_1"}


def test_fluxo_completo_e_privacidade(cliente):
    c, db = cliente
    assert c.get("/saude").json()["ok"]
    texto_secreto = "sou assistente administrativo e cuido de planilhas e notas fiscais XYZSEGREDO"
    s = c.post("/sugerir-cbo", json={"texto": texto_secreto}).json()["sugestoes"]
    assert 1 <= len(s) <= 3 and all(len(x["cbo"]) == 6 for x in s)
    # sem consentimento: nada gravado
    r = c.post("/classificar", json={"cbo": "411010", **PERFIL, "percepcao": "muito"}).json()
    assert r["exposicao"]["nivel"] == "alta" and r["natureza"]["tipo"] == "substituicao"
    assert r["explicacao_origem"] == "reserva" and r["gravado"] is False
    assert not db.exists() or sqlite3.connect(db).execute(
        "SELECT count(*) FROM sqlite_master WHERE name='respostas'").fetchone()[0] == 0 or \
        sqlite3.connect(db).execute("SELECT count(*) FROM respostas").fetchone()[0] == 0
    # com consentimento: só campos categóricos
    r = c.post("/classificar", json={"cbo": "411010", **PERFIL, "percepcao": "muito", "consentimento": True}).json()
    assert r["gravado"] is True
    con = sqlite3.connect(db)
    cols = [d[1] for d in con.execute("PRAGMA table_info(respostas)")]
    linhas = con.execute("SELECT * FROM respostas").fetchall()
    assert len(linhas) == 1
    assert set(cols) == set(__import__("painelia.armazenamento", fromlist=["CAMPOS"]).CAMPOS)
    assert "XYZSEGREDO" not in json.dumps(linhas) and len(linhas[0][0]) == 7  # mês AAAA-MM, sem dia


def test_validacao_e_relatorio_k10(cliente):
    c, _ = cliente
    assert c.post("/classificar", json={"cbo": "abc", **PERFIL}).status_code == 422
    assert c.post("/classificar", json={"cbo": "411010", **PERFIL, "uf": "XX"}).status_code == 422
    for _ in range(12):
        c.post("/classificar", json={"cbo": "411010", **PERFIL, "consentimento": True})
    for _ in range(11):
        c.post("/classificar", json={"cbo": "411010", **{**PERFIL, "faixa_etaria": "25-29"}, "consentimento": True})
    for _ in range(3):
        c.post("/classificar", json={"cbo": "715210", **{**PERFIL, "faixa_etaria": "40-49"}, "consentimento": True})
    assert c.get("/relatorio").status_code == 403
    rel = c.get("/relatorio?por=exposicao,faixa_etaria", headers={"X-Admin-Token": "segredo"}).json()["linhas"]
    assert {"exposicao": "alta", "faixa_etaria": "18-24", "n": 12} in rel
    # uma única célula < 10 (n=3): a menor visível (n=11) também é suprimida, para não reconstruir por diferença
    assert {"exposicao": "(suprimido)", "faixa_etaria": "(suprimido)", "n": 14} in rel
    assert all(l["n"] >= 10 or l["exposicao"] == "(suprimido)" for l in rel)
