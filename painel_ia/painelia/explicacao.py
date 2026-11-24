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

INSTRUCOES = """Você escreve, em português do Brasil, o resultado de um painel sobre a EXPOSIÇÃO de uma ocupação à IA
generativa, para a pessoa que trabalha nela. Escreva como uma conversa clara e respeitosa, em segunda pessoa.

Use APENAS os dados do JSON. Não invente números, estudos, salários, previsões nem tarefas que não estejam nas
listas do JSON. Não cite exemplos de fora do JSON. Se um dado faltar, diga que não há dado, em vez de supor.
Seja breve: frases curtas, sem repetir o que já disse e sem introdução nem conclusão.
Escreva números como no Brasil, com vírgula decimal (2,4%, não 2.4%), e não use termos técnicos do JSON.

Estrutura obrigatória: exatamente 5 parágrafos, separados por uma linha em branco, cada um começando com o trecho
em negrito indicado (use **...**). Cada parágrafo tem no máximo 2 frases:
1. "**Sua ocupação está [posicao_exposicao].**" Diga em 1 frase que exposição não é perda de emprego e como o
   trabalho tende a mudar, de acordo com "natureza".
2. "**No dia a dia,**" se "posicao_exposicao" disser "entre as menos expostas", diga que as
   tarefas principais continuam sendo da pessoa, citando 2 de "tarefas_menos_expostas", e que a IA pode no
   máximo ajudar em tarefas de apoio, como organização e comunicação; nunca diga que a IA já faz bem tarefas
   físicas ou práticas. Nos outros casos, cite 2 tarefas que a IA já consegue fazer em boa parte (de
   "tarefas_mais_expostas"; diga "já consegue", nunca "já faz bem", porque o índice mede o que é tecnicamente
   possível) e 1 que continua dependendo da pessoa (de "tarefas_menos_expostas"). Reescreva cada tarefa em
   linguagem simples e curta, com até 8 palavras, como a própria pessoa a descreveria, e junte-as com vírgulas.
   Não comente a falta de dados das listas (se faltar tarefa, omita). Use SOMENTE tarefas dessas listas, escolhendo
   as que façam sentido para a ocupação e descartando as que claramente não se aplicam a ela. Se
   "tarefas_de_ocupacoes_vizinhas" for true, tenha ainda mais cuidado nessa escolha.
3. "**Nas contratações,**" diga o que "evidencia_br" mostra para o perfil da pessoa, em comparação com o resto do
   mercado desde 2022, e que é um retrato, não prova de que a causa seja a IA. Sem dados ou com dados
   insuficientes, diga isso em 1 frase, sem números. Diga só se as contratações caíram, subiram ou ficaram
   estáveis, sem repetir percentuais técnicos nem palavras como "classe" ou "janelas". Descreva os períodos em
   palavras, como "desde 2022", nunca com códigos como 2021T4.
4. "**A pesquisa por trás deste painel**" resuma em 1 frase o campo "achado_artigo", sem mudar os números. Se
   o campo trouxer um número de vagas com intervalo, cite sempre o intervalo junto com o valor central.
5. "**Para você,**" dê 1 ou 2 caminhos concretos, ligados às tarefas menos expostas ou ao uso da IA como apoio,
   de acordo com a faixa etária, o tempo na função e "uso_ia" (nunca, as_vezes, semanal, diario). Use
   esses dados só para orientar o conselho, sem repeti-los de volta à pessoa. O conselho deve ser uma ação
   (aprender, praticar, assumir), e não só repetir uma tarefa da lista. Quem está no começo da carreira (até 29
   anos ou até 3 anos na função) cresce nas tarefas de entrada da ocupação; quem tem experiência tem nela um
   ativo.

Tom sereno, sem alarmismo nem otimismo exagerado. Sem listas, sem títulos, sem emojis. Cada parágrafo com no
máximo 30 palavras; entre 110 e 150 palavras no total."""

POSICAO = {"alta": "entre as mais expostas à IA generativa", "media": "entre as de exposição intermediária à IA generativa",
           "baixa": "entre as menos expostas à IA generativa"}
MUDANCA = {"substituicao": "o seu trabalho é dos que mais tendem a mudar, com menos tempo em tarefas de rotina e mais "
                           "tempo revisando, decidindo e lidando com pessoas",
           "transformacao": "parte importante do seu trabalho tende a se transformar: a IA pode assumir algumas tarefas, "
                            "e as outras ganham peso",
           "complementaridade": "a IA tende a funcionar como apoio em algumas tarefas, enquanto o centro do seu trabalho "
                                "continua dependendo de você",
           "pouco_afetada": "o seu trabalho depende de esforço físico, de habilidade prática ou de estar presente, "
                            "coisas que a IA generativa, que lida com textos, imagens e informação, faz muito pouco"}
GRUPO = {"18-24": "18 a 24 anos", "25-29": "25 a 29 anos", "30+": "30 anos ou mais"}


def _tarefas_pt(lista: list | None, n: int) -> list[str]:
    out = []
    for t in (lista or [])[:n]:
        pt = t.get("tarefa_pt")
        if pt:
            out.append(pt[0].lower() + pt[1:].rstrip(".;"))
    return out


def _juntar(itens: list[str]) -> str:
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def _dados_prompt(ocup: dict, perfil: dict, evidencia: dict | None, achado: str | None = None) -> dict:
    return {
        "ocupacao": ocup.get("titulo"), "cbo": ocup.get("cbo"),
        "posicao_exposicao": POSICAO[ocup["exposicao"]], "gradiente_oit": ocup.get("gradiente_rotulo"),
        "natureza": ocup.get("natureza", {}).get("rotulo"),
        "tarefas_mais_expostas": [t.get("tarefa_pt") or t.get("tarefa") for t in ocup.get("tarefas_mais_expostas", [])],
        "tarefas_menos_expostas": [t.get("tarefa_pt") or t.get("tarefa") for t in ocup.get("tarefas_menos_expostas", [])],
        "tarefas_de_ocupacoes_vizinhas": ocup.get("tarefas_de_ocupacoes_vizinhas", False),
        "perfil": perfil, "evidencia_br": evidencia, "achado_artigo": achado,
    }


def texto_reserva(ocup: dict, perfil: dict, evidencia: dict | None, achado: str | None = None) -> str:
    """Os mesmos 5 parágrafos do GPT, montados sem IA (sem chave, sem rede ou se a OpenAI falhar)."""
    nivel, nat = ocup["exposicao"], ocup.get("natureza", {}).get("tipo", "pouco_afetada")
    p = [f"**Sua ocupação está {POSICAO[nivel]}.** Segundo o índice da Organização Internacional do Trabalho, isso "
         f"mede quanto das tarefas típicas da sua ocupação a IA consegue fazer tecnicamente. Não quer dizer que o "
         f"seu emprego vai acabar. Indica que {MUDANCA[nat]}."]
    mais, menos = _tarefas_pt(ocup.get("tarefas_mais_expostas"), 3), _tarefas_pt(ocup.get("tarefas_menos_expostas"), 2)
    if nivel == "baixa":
        t = "**No dia a dia,** as tarefas principais do seu trabalho continuam sendo suas"
        t += f", como {_juntar(mais + menos)}" if mais or menos else ""
        p.append(t + ". A IA tende a ajudar mais nas tarefas de apoio, como organização e comunicação.")
    elif mais:
        t = f"**No dia a dia,** a IA já faz bem tarefas como {_juntar(mais)}."
        t += f" Continuam dependendo mais de você tarefas como {_juntar(menos)}." if menos else ""
        p.append(t)
    else:
        p.append("**No dia a dia,** a IA tende a ajudar sobretudo nas tarefas de texto, cálculo e organização de "
                 "informação, enquanto o contato com pessoas e as decisões continuam dependendo de você.")
    if evidencia and evidencia.get("classe") in ("queda", "alta", "estável"):
        grupo = GRUPO.get(evidencia.get("grupo_idade"), evidencia.get("grupo_idade"))
        dipl = " com diploma" if evidencia.get("escolaridade") == "superior" else ""
        verbo = {"queda": "caíram", "alta": "cresceram", "estável": "ficaram estáveis"}[evidencia["classe"]]
        p.append(f"**Nas contratações,** os dados do Ministério do Trabalho mostram que, desde 2022, as contratações "
                 f"de pessoas de {grupo}{dipl} nessa família de ocupações {verbo} em comparação com o resto do "
                 f"mercado. É um retrato do que aconteceu, e não prova que a causa seja a IA.")
    else:
        p.append("**Nas contratações,** não há dados suficientes no Novo CAGED para mostrar como andaram as "
                 "contratações do seu perfil nesta família de ocupações.")
    if achado:
        p.append("**A pesquisa por trás deste painel** encontrou o seguinte: " + achado[0].lower() + achado[1:])
    inicio = perfil.get("tempo_funcao") in ("menos_1", "1_3") or perfil.get("faixa_etaria") in ("14-17", "18-24", "25-29")
    uso = perfil.get("uso_ia")
    if nivel == "baixa":
        t = "**Para você,** o seu ofício segue valorizado."
        t += (" Se quiser, vale experimentar a IA como apoio nas tarefas administrativas, para sobrar mais tempo ao "
              "que só você faz." if uso in (None, "nunca", "as_vezes") else
              " Você já usa IA no trabalho: ela pode poupar tempo na parte administrativa, para você se dedicar ao "
              "que só você faz.")
    elif inicio:
        t = ("**Para você,** que está no começo da carreira, dois caminhos: assumir a revisão e a conferência do que a "
             "IA produz, que é onde o olhar humano faz falta, e buscar tarefas de contato e coordenação, que a IA faz "
             "pouco e que contam muito para crescer.")
        t += (" Se ainda não usa IA no trabalho, vale começar pelas tarefas mais repetitivas." if uso in (None, "nunca")
              else " Como você já usa IA, está um passo à frente.")
    else:
        t = ("**Para você,** a experiência é um ativo: as tarefas menos expostas costumam exigir conhecimento "
             "acumulado, julgamento e relacionamento. Dominar a IA nas tarefas mais expostas tende a aumentar a sua "
             "produtividade, e não a substituí-la.")
        t += (" Se ainda não usa IA no trabalho, experimentar nas tarefas de rotina é um bom começo." if uso in
              (None, "nunca") else "")
    p.append(t)
    return "\n\n".join(p)


@lru_cache(maxsize=4096)
def _gerar(chave: str, payload: str) -> str:
    from openai import OpenAI
    cliente = OpenAI(timeout=float(os.environ.get("OPENAI_TIMEOUT", "25")))
    # O limite conta também os tokens de raciocínio do GPT-5. Se estourar, a resposta vem cortada
    # (status "incomplete"): tentamos de novo com o dobro e, se falhar, levantamos erro. O erro não
    # entra no cache e o painel usa o texto de reserva, em vez de mostrar um texto pela metade.
    limite = int(os.environ.get("OPENAI_MAX_TOKENS", "3000"))
    for _ in range(2):
        r = cliente.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
            instructions=INSTRUCOES,
            input="Dados:\n" + payload,
            max_output_tokens=limite,
            store=False,
            # GPT-5: pouco raciocínio e resposta enxuta. Se trocar OPENAI_MODEL por um modelo sem esses
            # parâmetros, a chamada falha e o painel usa o texto de reserva.
            reasoning={"effort": os.environ.get("OPENAI_REASONING", "low")},
            text={"verbosity": os.environ.get("OPENAI_VERBOSITY", "low")},
        )
        if getattr(r, "status", "completed") != "incomplete":
            return r.output_text.strip()
        limite *= 2
    raise RuntimeError("resposta da OpenAI incompleta")


def explicar(ocup: dict, perfil: dict, evidencia: dict | None, achado: str | None = None) -> tuple[str, str]:
    """Devolve (texto, origem) com origem ∈ {"gpt", "reserva"}."""
    if not os.environ.get("OPENAI_API_KEY"):
        return texto_reserva(ocup, perfil, evidencia, achado), "reserva"
    dados = _dados_prompt(ocup, perfil, evidencia, achado)
    payload = json.dumps(dados, ensure_ascii=False, sort_keys=True)
    try:
        txt = _gerar(f"{ocup['cbo']}|{sorted(perfil.items())}", payload)
        return (txt, "gpt") if txt else (texto_reserva(ocup, perfil, evidencia, achado), "reserva")
    except Exception:  # sem rede, cota, modelo inválido: o painel não pode cair por causa do texto
        return texto_reserva(ocup, perfil, evidencia, achado), "reserva"
