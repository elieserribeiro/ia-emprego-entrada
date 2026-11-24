"""Traduz UMA VEZ as tarefas da OIT (inglês) para o português do Brasil, e guarda em dados/tarefas_pt.csv.

Assim o painel mostra as tarefas em português mesmo sem chamar a OpenAI na hora (texto de reserva).

Uso (precisa de OPENAI_API_KEY; são cerca de 3.300 tarefas, em lotes, e custa poucos centavos de dólar):
    python -m painelia.traduzir_tarefas
Pode ser interrompido e retomado: tarefas já traduzidas não são traduzidas de novo.
Depois rode de novo:  python -m painelia.classificacao
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ORIGEM = RAIZ / "dados" / "ilo2025_tarefas_isco08.csv"
SAIDA = RAIZ / "dados" / "tarefas_pt.csv"
LOTE = 40

INSTRUCOES = """Traduza para o português do Brasil cada tarefa ocupacional (da classificação ISCO-08 da OIT).
Regras: linguagem simples e direta, como alguém descreveria o próprio trabalho; comece com verbo no infinitivo
(ex.: "Redigir relatórios de rotina"); sem ponto final; até 18 palavras; mantenha o sentido, sem acrescentar nada.
Responda SÓ com um objeto JSON {"<id>": "<tradução>", ...}, com exatamente as mesmas chaves recebidas."""


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Defina OPENAI_API_KEY antes de rodar.")
    from openai import OpenAI
    cliente = OpenAI()
    modelo = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    t = pd.read_csv(ORIGEM, dtype={"isco08": str, "tarefa_id": str})
    t["chave"] = t.isco08 + "|" + t.tarefa_id
    feitas = pd.read_csv(SAIDA, dtype=str) if SAIDA.exists() else pd.DataFrame(columns=["isco08", "tarefa_id", "tarefa_pt"])
    ja = set(feitas.isco08 + "|" + feitas.tarefa_id)
    falta = t[~t.chave.isin(ja)]
    print(f"{len(t)} tarefas; {len(ja)} já traduzidas; faltam {len(falta)}")
    for i in range(0, len(falta), LOTE):
        lote = falta.iloc[i:i + LOTE]
        entrada = {r.chave: r.tarefa for r in lote.itertuples()}
        r = cliente.responses.create(model=modelo, instructions=INSTRUCOES, store=False,
                                     input=json.dumps(entrada, ensure_ascii=False))
        texto = r.output_text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            trad = json.loads(texto)
        except json.JSONDecodeError:
            print(f"lote {i // LOTE + 1}: resposta inválida, pulado (rode de novo para completar)")
            continue
        novas = [{"isco08": k.split("|")[0], "tarefa_id": k.split("|")[1], "tarefa_pt": v.strip()}
                 for k, v in trad.items() if k in entrada and isinstance(v, str) and v.strip()]
        feitas = pd.concat([feitas, pd.DataFrame(novas)], ignore_index=True)
        feitas.to_csv(SAIDA, index=False)
        print(f"lote {i // LOTE + 1}/{-(-len(falta) // LOTE)}: {len(novas)} tarefas")
    print(f"pronto: {len(feitas)} tarefas em {SAIDA}")


if __name__ == "__main__":
    main()
