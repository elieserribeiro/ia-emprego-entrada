"""Armazenamento CONSENTIDO e mínimo das classificações (LGPD), e relatório com supressão k ≥ 10.

Só é gravado quando a pessoa marca o consentimento (caixa desmarcada por padrão). Campos gravados, todos
categóricos: CBO confirmada, classes do resultado, faixa etária, escolaridade, UF, faixa de tempo na função,
resposta da pergunta de percepção e MÊS (sem dia/hora). NUNCA gravamos: o texto livre digitado, IP, e-mail,
identificadores de navegador/dispositivo, cookies.

Destinos:
  - Supabase (produção): variáveis SUPABASE_URL e SUPABASE_SERVICE_KEY (chave só no servidor); tabela do
    arquivo supabase/schema.sql, com RLS ligada e nenhuma política para o público.
  - SQLite local (desenvolvimento/testes): PAINEL_SQLITE=caminho.db.
  - Nenhum configurado: nada é gravado.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date

import pandas as pd

CAMPOS = ["mes", "cbo", "cbo_familia", "exposicao", "natureza", "sinal_br", "faixa_etaria", "escolaridade", "uf",
          "tempo_funcao", "percepcao", "uso_ia", "sexo", "raca"]
PERFIL = ("faixa_etaria", "escolaridade", "uf", "tempo_funcao", "percepcao", "uso_ia", "sexo", "raca")
K_MIN = 10


def registro(cbo: str, resultado: dict, perfil: dict, sensivel: bool = False) -> dict:
    """Monta o registro permitido. Qualquer campo fora de CAMPOS é descartado aqui.

    Raça/cor é dado sensível (LGPD, art. 11): só é gravada com o consentimento específico (`sensivel=True`)."""
    r = {
        "mes": date.today().strftime("%Y-%m"),
        "cbo": cbo, "cbo_familia": cbo[:4],
        "exposicao": resultado["exposicao"]["nivel"],
        "natureza": resultado["natureza"]["tipo"],
        "sinal_br": (resultado.get("evidencia_br") or {}).get("classe"),
        **{k: perfil.get(k) for k in PERFIL},
    }
    if not sensivel:
        r["raca"] = None
    return {k: r.get(k) for k in CAMPOS}


def destino() -> str:
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
        return "supabase"
    if os.environ.get("PAINEL_SQLITE"):
        return "sqlite"
    return "nenhum"


def _sqlite():
    con = sqlite3.connect(os.environ["PAINEL_SQLITE"])
    con.execute(f"CREATE TABLE IF NOT EXISTS respostas ({', '.join(c + ' TEXT' for c in CAMPOS)})")
    return con


def gravar(reg: dict) -> bool:
    d = destino()
    if d == "nenhum":
        return False
    if d == "sqlite":
        with _sqlite() as con:
            con.execute(f"INSERT INTO respostas VALUES ({','.join('?' * len(CAMPOS))})", [reg[c] for c in CAMPOS])
        return True
    import httpx
    url = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/respostas"
    chave = os.environ["SUPABASE_SERVICE_KEY"]
    r = httpx.post(url, json=reg, timeout=10, headers=_cab_supabase(chave, {"Prefer": "return=minimal"}))
    r.raise_for_status()
    return True


def _cab_supabase(chave: str, extra: dict | None = None) -> dict:
    """Cabeçalhos do PostgREST. Chaves antigas (service_role, JWT 'eyJ...') vão também em Authorization;
    as chaves novas ('sb_secret_...') não são JWT e vão só em apikey."""
    h = {"apikey": chave}
    if chave.startswith("eyJ"):
        h["Authorization"] = f"Bearer {chave}"
    return {**h, **(extra or {})}


def _todas() -> pd.DataFrame:
    d = destino()
    if d == "sqlite":
        with _sqlite() as con:
            return pd.read_sql("SELECT * FROM respostas", con)
    if d == "supabase":
        import httpx
        chave = os.environ["SUPABASE_SERVICE_KEY"]
        url = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/respostas?select=" + ",".join(CAMPOS)
        linhas, ini = [], 0
        while True:  # paginação do PostgREST
            r = httpx.get(url, timeout=30, headers=_cab_supabase(chave, {"Range": f"{ini}-{ini + 9999}"}))
            r.raise_for_status()
            lote = r.json()
            linhas += lote
            if len(lote) < 10000:
                break
            ini += 10000
        return pd.DataFrame(linhas, columns=CAMPOS)
    return pd.DataFrame(columns=CAMPOS)


def relatorio(por: list[str], k: int = K_MIN) -> pd.DataFrame:
    """Contagens agregadas por `por`; células com menos de k respostas são suprimidas (não aparecem).

    O total suprimido é informado numa linha à parte, para não permitir reconstruir as células por diferença
    quando há só uma célula suprimida (nesse caso a menor célula visível também é suprimida)."""
    por = [c for c in por if c in CAMPOS]
    df = _todas()
    if df.empty or not por:
        return pd.DataFrame(columns=por + ["n"])
    g = df.groupby(por, dropna=False).size().rename("n").reset_index()
    sup = g.n < k
    if sup.sum() == 1 and (~sup).any():  # supressão secundária
        sup.loc[g[~sup].n.idxmin()] = True
    vis = g[~sup].copy()
    if sup.any():
        vis = pd.concat([vis, pd.DataFrame([{**{c: "(suprimido)" for c in por}, "n": int(g[sup].n.sum())}])])
    return vis.reset_index(drop=True)
