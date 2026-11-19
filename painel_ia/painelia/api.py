"""API do painel "Exposição da sua ocupação à IA" (FastAPI).

Rodar localmente:   uvicorn painelia.api:app --reload --no-access-log
Variáveis de ambiente:
  OPENAI_API_KEY        chave da OpenAI (só no servidor). Sem ela, o texto usa a reserva determinística.
  OPENAI_MODEL          modelo (padrão gpt-5-mini)
  ORIGENS_PERMITIDAS    domínios do site, separados por vírgula (ex.: https://meusite.lovable.app)
  SUPABASE_URL, SUPABASE_SERVICE_KEY   gravação consentida (ou PAINEL_SQLITE=arquivo.db para testes)
  ADMIN_TOKEN           protege /relatorio
  PAINEL_ARTEFATOS      pasta dos artefatos (padrão painel_ia/artefatos)

Privacidade: o texto livre chega só em /sugerir-cbo, é usado para a busca e descartado. Não há log de acesso
(rode o uvicorn com --no-access-log), e nada é gravado sem consentimento.
"""
from __future__ import annotations

import json
import os
import threading
import time
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import armazenamento, busca, explicacao

RAIZ = Path(__file__).resolve().parents[1]
VERSAO = "1.0"


class Faixa(str, Enum):
    a14_17 = "14-17"
    a18_24 = "18-24"
    a25_29 = "25-29"
    a30_39 = "30-39"
    a40_49 = "40-49"
    a50 = "50+"


class Escolaridade(str, Enum):
    fundamental = "fundamental"   # até fundamental completo
    medio = "medio"               # médio completo ou superior incompleto
    superior = "superior"         # superior completo
    pos = "pos"                   # pós-graduação


class Tempo(str, Enum):
    menos_1 = "menos_1"
    t1_3 = "1_3"
    t3_10 = "3_10"
    mais_10 = "mais_10"


class Percepcao(str, Enum):
    nada = "nada"
    pouco = "pouco"
    moderado = "moderado"
    muito = "muito"
    nao_sei = "nao_sei"


UFS = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
       "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
UF = Enum("UF", {u: u for u in UFS}, type=str)


class PedidoSugestao(BaseModel):
    texto: str = Field(..., min_length=3, max_length=1000, description="cargo e principais tarefas")


class PedidoClassificacao(BaseModel):
    cbo: str = Field(..., pattern=r"^\d{6}$")
    faixa_etaria: Faixa
    escolaridade: Escolaridade
    uf: UF
    tempo_funcao: Tempo
    percepcao: Percepcao | None = None
    consentimento: bool = False


def _carregar():
    art = Path(os.environ.get("PAINEL_ARTEFATOS", RAIZ / "artefatos"))
    dados = json.loads((art / "ocupacoes.json").read_text(encoding="utf-8"))
    tit_p = art / "titulos_cbo.json"
    titulos = json.loads(tit_p.read_text(encoding="utf-8")) if tit_p.exists() else {}
    return dados, titulos, busca.carregar(art)


DADOS, TITULOS, BUSCADOR = _carregar()
OCUP = DADOS["ocupacoes"]

app = FastAPI(title="Exposição da ocupação à IA generativa", version=VERSAO,
              description="Classifica a EXPOSIÇÃO (não o impacto) de ocupações brasileiras à IA generativa.")
app.add_middleware(CORSMiddleware,
                   allow_origins=[o.strip() for o in os.environ.get("ORIGENS_PERMITIDAS", "*").split(",") if o.strip()],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Admin-Token"])

# limite global simples (sem IP: não identificamos quem chama)
_lock, _janela, _cont = threading.Lock(), [time.time()], [0]
LIMITE_MIN = int(os.environ.get("LIMITE_POR_MINUTO", "600"))


def _limitar():
    with _lock:
        if time.time() - _janela[0] > 60:
            _janela[0], _cont[0] = time.time(), 0
        _cont[0] += 1
        if _cont[0] > LIMITE_MIN:
            raise HTTPException(429, "muitas requisições; tente em um minuto")


@app.get("/saude")
def saude():
    return {"ok": True, "versao": VERSAO, "motor_busca": BUSCADOR.nome, "ocupacoes": len(OCUP),
            "texto": "gpt" if os.environ.get("OPENAI_API_KEY") else "reserva",
            "armazenamento": armazenamento.destino()}


@app.post("/sugerir-cbo")
def sugerir(p: PedidoSugestao):
    _limitar()
    res = []
    for cbo, s in BUSCADOR.buscar(p.texto, k=8):
        if cbo in OCUP and cbo not in [r["cbo"] for r in res]:
            res.append({"cbo": cbo, "titulo": TITULOS.get(cbo) or OCUP[cbo].get("titulo"), "similaridade": round(s, 3)})
        if len(res) == 3:
            break
    return {"sugestoes": res}  # o texto não é guardado nem registrado


def _evidencia(ocup: dict, faixa: str, esc: str) -> dict | None:
    ev = ocup.get("evidencia_br")
    if not ev:
        return None
    grupo = {"18-24": "18-24", "25-29": "25-29", "14-17": None}.get(faixa, "30+")
    if grupo is None:
        return None
    instr = {"superior": "superior", "pos": "superior", "medio": "medio"}.get(esc, "outro")
    fino = (ev.get("por_idade_escolaridade", {}).get(grupo) or {}).get(instr)
    base = ev.get("por_idade", {}).get(grupo)
    escolhido = fino if fino and fino.get("classe") != "dados insuficientes" else base
    if not escolhido:
        return None
    return {**escolhido, "grupo_idade": grupo, "escolaridade": instr if escolhido is fino else "todas",
            "janelas": ev.get("janelas"), "familia_cbo": ocup["cbo"][:4]}


def _mensagem_perfil(nivel: str, faixa: str, tempo: str) -> str:
    inicio = tempo in ("menos_1", "1_3") or faixa in ("18-24", "25-29")
    if nivel == "baixa":
        return ("O núcleo do seu trabalho depende pouco de tarefas que a IA generativa faz hoje. Ainda assim, "
                "ferramentas de IA podem ajudar em partes administrativas, como orçamentos, agendas e mensagens.")
    if inicio:
        return ("Em início de carreira, as tarefas de entrada (redigir, organizar, pesquisar, responder) costumam "
                "ser as mais expostas. Ganhar cedo experiência nas tarefas menos expostas, como julgamento, contato "
                "com clientes e coordenação, e usar a IA como apoio ajuda a se diferenciar.")
    return ("Sua experiência é um ativo: as tarefas menos expostas costumam exigir conhecimento acumulado, "
            "julgamento e relacionamento. Dominar as ferramentas de IA nas tarefas mais expostas tende a "
            "aumentar sua produtividade, não a substituí-la.")


@app.post("/classificar")
def classificar(p: PedidoClassificacao):
    _limitar()
    o = OCUP.get(p.cbo)
    if not o:
        raise HTTPException(404, "CBO sem dados de exposição")
    perfil = {"faixa_etaria": p.faixa_etaria.value, "escolaridade": p.escolaridade.value, "uf": p.uf.value,
              "tempo_funcao": p.tempo_funcao.value, "percepcao": p.percepcao.value if p.percepcao else None}
    ev = _evidencia(o, perfil["faixa_etaria"], perfil["escolaridade"])
    texto, origem = explicacao.explicar(o, {k: v for k, v in perfil.items() if k != "percepcao"}, ev)
    res = {
        "cbo": p.cbo, "titulo": TITULOS.get(p.cbo) or o.get("titulo"),
        "exposicao": {"nivel": o["exposicao"], "gradiente": o["gradiente"], "gradiente_rotulo": o["gradiente_rotulo"],
                      "percentil": o.get("percentil"), "percentil_base": o.get("percentil_base"),
                      "media_tarefas": o["media"], "dp_tarefas": o["dp"]},
        "natureza": o["natureza"],
        "tarefas_mais_expostas": o["tarefas_mais_expostas"], "tarefas_menos_expostas": o["tarefas_menos_expostas"],
        "tarefas_de_ocupacoes_vizinhas": o.get("tarefas_de_ocupacoes_vizinhas", False),
        "evidencia_br": ev, "evidencia_geral": DADOS.get("evidencia_geral"),
        "mensagem_perfil": _mensagem_perfil(o["exposicao"], perfil["faixa_etaria"], perfil["tempo_funcao"]),
        "explicacao": texto, "explicacao_origem": origem,
        "aviso": ("Exposição não é previsão de perda de emprego: mede quanto das tarefas a IA generativa consegue "
                  "fazer tecnicamente (índice da OIT, 2025). O efeito real depende de adoção, custos e decisões "
                  "das empresas."),
        "fontes": ["OIT — Gmyrek et al. (2025), WP 140, CC BY 4.0", "MTE — Novo CAGED (microdados)",
                   "MTE — Classificação Brasileira de Ocupações (CBO 2002)"],
    }
    gravado = False
    if p.consentimento:
        try:
            gravado = armazenamento.gravar(armazenamento.registro(p.cbo, res, perfil))
        except Exception:
            gravado = False  # a falha de gravação não impede a pessoa de ver o resultado
    res["gravado"] = gravado
    return res


@app.get("/relatorio")
def relatorio(por: str = Query("exposicao,faixa_etaria", description="campos separados por vírgula"),
              x_admin_token: str | None = Header(None)):
    if not os.environ.get("ADMIN_TOKEN") or x_admin_token != os.environ["ADMIN_TOKEN"]:
        raise HTTPException(403, "proibido")
    t = armazenamento.relatorio(por.split(","))
    return {"k_minimo": armazenamento.K_MIN, "linhas": t.to_dict("records")}
