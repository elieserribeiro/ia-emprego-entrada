"""Linha de comando: python -m iaemprego <etapa> [--config arquivo.yaml]

Etapas, na ordem:
  baixar-caged   baixar-pnad   baixar-ipca     (Etapa 1)
  ingerir        validar                       (Etapa 2)
  exposicao                                    (Etapa 3)
  painel                                       (Etapa 4)
  descritivas                                  (Etapa 5)
  modelos        heterogeneidade  robustez  pretendencia  aceleracao  diferencas  evento-raca   (Etapas 6, 7 e 9)
  pnad                                         (Etapa 8)
  mercado                                      (Etapa 10: mercado como um todo)
  figuras                                      (figuras e tabelas finais do artigo, PT e EN)
  tudo           roda da ingestão aos modelos (sem downloads)
"""
import argparse
import sys
import time

from . import caged, descritivas, download, exposicao, figuras, mercado, modelos, painel, pnad
from .util import carregar_config, configurar_log, log

ETAPAS = {
    "baixar-caged": download.baixar_caged,
    "baixar-pnad": download.baixar_pnad,
    "baixar-ipca": download.baixar_ipca,
    "ingerir": caged.ingerir,
    "validar": caged.validar,
    "exposicao": exposicao.construir,
    "painel": painel.construir,
    "descritivas": descritivas.rodar,
    "modelos": modelos.rodar_principal,
    "heterogeneidade": modelos.rodar_heterogeneidade,
    "robustez": modelos.rodar_robustez,
    "pretendencia": modelos.rodar_pretendencia,
    "aceleracao": modelos.rodar_aceleracao,
    "diferencas": modelos.rodar_diferencas,
    "evento-raca": modelos.rodar_evento_raca,
    "pnad": pnad.rodar,
    "mercado": mercado.rodar,
    "figuras": figuras.rodar,
}
TUDO = ["ingerir", "validar", "exposicao", "painel", "descritivas", "modelos", "heterogeneidade", "robustez", "pretendencia", "aceleracao", "diferencas", "evento-raca", "pnad", "mercado", "figuras"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="iaemprego", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("etapa", choices=list(ETAPAS) + ["tudo"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    configurar_log()
    cfg = carregar_config(a.config)
    for etapa in (TUDO if a.etapa == "tudo" else [a.etapa]):
        t0 = time.time()
        log.info("=== %s ===", etapa)
        ETAPAS[etapa](cfg)
        log.info("=== %s concluída em %.0fs ===", etapa, time.time() - t0)


if __name__ == "__main__":
    sys.exit(main())
