"""Texto explicativo do resultado: OpenAI (GPT) ancorado SÓ nos dados do painel, com reserva determinística.

- A chave fica no servidor (variável de ambiente OPENAI_API_KEY); o site nunca a vê.
- Não enviamos texto livre do usuário ao GPT: só a CBO confirmada, os dados da OIT/CAGED e o perfil categórico.
- `store=False`: pede à OpenAI que não guarde a resposta para uso posterior.
- Respostas são guardadas em cache na memória por (CBO, perfil): a mesma combinação não gera nova chamada.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

INSTRUCOES = """Você explica, em português do Brasil, a EXPOSIÇÃO de uma ocupação à IA generativa para uma pessoa
que trabalha nela. Regras:
1. Use APENAS os dados do JSON fornecido. Não invente números, estudos, salários nem previsões.
2. Exposição não é impacto: diga que o índice mede quanto das tarefas a IA consegue fazer tecnicamente,
   não se a pessoa vai perder o emprego. Isso depende de adoção, custos, regulação e decisões das empresas.
3. Traduza para o português e cite 2 tarefas mais expostas e 1 ou 2 menos expostas. Se
   "tarefas_de_ocupacoes_vizinhas" for true, as tarefas vêm de ocupações parecidas: use só as que fazem
   sentido para o título da ocupação e não diga que são exatamente as dela.
4. Se houver "evidencia_br", diga em uma frase o que aconteceu com as contratações do perfil da pessoa no
   Brasil (o JSON traz a classe e a variação), deixando claro que é um dado descritivo e não prova que a causa
   foi a IA. Se for "dados insuficientes" ou ausente, não mencione números.
5. Personalize pelo perfil: pessoas em início de carreira (pouco tempo na função, 18–29 anos) dependem mais das
   tarefas de entrada, que costumam ser as mais expostas; experiência, rede de contatos, formação e
   familiaridade com as ferramentas aumentam a capacidade de adaptação. Sugira 2 caminhos concretos e realistas
   ligados às tarefas MENOS expostas ou ao uso da IA como apoio.
6. Tom sereno e respeitoso, sem alarmismo nem otimismo exagerado. No máximo 170 palavras, em 2 ou 3 parágrafos,
   sem títulos, sem listas, sem emojis."""


def _dados_prompt(ocup: dict, perfil: dict, evidencia: dict | None) -> dict:
    return {
        "ocupacao": ocup.get("titulo"), "cbo": ocup.get("cbo"),
        "exposicao": ocup.get("exposicao"), "gradiente_oit": ocup.get("gradiente_rotulo"),
        "percentil_entre_ocupacoes": ocup.get("percentil"), "natureza": ocup.get("natureza", {}).get("rotulo"),
        "tarefas_mais_expostas": ocup.get("tarefas_mais_expostas"),
        "tarefas_menos_expostas": ocup.get("tarefas_menos_expostas"),
        "tarefas_de_ocupacoes_vizinhas": ocup.get("tarefas_de_ocupacoes_vizinhas", False),
        "perfil": perfil, "evidencia_br": evidencia,
    }


def texto_reserva(ocup: dict, perfil: dict, evidencia: dict | None) -> str:
    nivel = {"alta": "alta", "media": "média", "baixa": "baixa"}[ocup["exposicao"]]
    t = (f"A ocupação {ocup.get('titulo') or ocup['cbo']} tem exposição {nivel} à IA generativa "
         f"({ocup['gradiente_rotulo']}, segundo a OIT). Isso mede quanto das tarefas a IA consegue fazer "
         f"tecnicamente, e não se o emprego vai acabar. {ocup['natureza']['texto']}")
    if evidencia and evidencia.get("classe") in ("queda", "alta", "estável"):
        t += (f" No Brasil, as admissões para o seu perfil nesta família de ocupações ficaram "
              f"{'em queda' if evidencia['classe'] == 'queda' else 'em alta' if evidencia['classe'] == 'alta' else 'estáveis'}"
              f" em relação ao mercado como um todo desde 2022 (dado descritivo do CAGED, não prova de efeito da IA).")
    if perfil.get("tempo_funcao") in ("menos_1", "1_3"):
        t += (" Em início de carreira, as tarefas de entrada costumam ser as mais expostas: vale investir nas "
              "atividades que exigem julgamento e contato com pessoas, e em usar a IA como apoio.")
    return t


@lru_cache(maxsize=4096)
def _gerar(chave: str, payload: str) -> str:
    from openai import OpenAI
    cliente = OpenAI(timeout=float(os.environ.get("OPENAI_TIMEOUT", "25")))
    r = cliente.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        instructions=INSTRUCOES,
        input="Dados:\n" + payload,
        max_output_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "1200")),
        store=False,
    )
    return r.output_text.strip()


def explicar(ocup: dict, perfil: dict, evidencia: dict | None) -> tuple[str, str]:
    """Devolve (texto, origem) com origem ∈ {"gpt", "reserva"}."""
    if not os.environ.get("OPENAI_API_KEY"):
        return texto_reserva(ocup, perfil, evidencia), "reserva"
    dados = _dados_prompt(ocup, perfil, evidencia)
    payload = json.dumps(dados, ensure_ascii=False, sort_keys=True)
    try:
        txt = _gerar(f"{ocup['cbo']}|{sorted(perfil.items())}", payload)
        return (txt, "gpt") if txt else (texto_reserva(ocup, perfil, evidencia), "reserva")
    except Exception:  # sem rede, cota, modelo inválido: o painel não pode cair por causa do texto
        return texto_reserva(ocup, perfil, evidencia), "reserva"
