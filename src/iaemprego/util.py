"""Funções comuns: configuração, caminhos, log e normalização de nomes de colunas."""
from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
log = logging.getLogger("iaemprego")


def configurar_log(nivel=logging.INFO):
    logging.basicConfig(level=nivel, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def carregar_config(arquivo: str | os.PathLike | None = None) -> dict:
    arquivo = Path(arquivo or os.environ.get("IAEMPREGO_CONFIG", RAIZ / "config.yaml"))
    cfg = yaml.safe_load(arquivo.read_text(encoding="utf-8"))
    base = arquivo.resolve().parent
    cfg["_caminhos"] = {k: (base / v).resolve() for k, v in cfg["caminhos"].items()}
    for p in cfg["_caminhos"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg


def caminho(cfg: dict, chave: str, *partes: str) -> Path:
    p = cfg["_caminhos"][chave].joinpath(*partes)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def normalizar_nome(nome: str) -> str:
    """'competênciamov' -> 'competenciamov'; remove acentos, espaços e caixa."""
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch == "_")


def meses_publicacao(inicio: str, fim: str) -> list[str]:
    import pandas as pd
    return [p.strftime("%Y%m") for p in pd.period_range(inicio, fim, freq="M")]


def intervalo_comp(cfg, con=None) -> tuple[int, int]:
    """(primeira, última) competência AAAAMM do estudo. `fim: null` = última presente nos dados MOV."""
    ini = int(cfg["periodo"]["inicio"].replace("-", ""))
    fim = cfg["periodo"]["fim"]
    if fim:
        return ini, int(str(fim).replace("-", ""))
    if con is None:
        raise ValueError("periodo.fim é null: passe a conexão DuckDB para descobrir o último mês")
    return ini, int(con.sql("SELECT max(comp) FROM mov2 WHERE tipo_arquivo = 'MOV'").fetchall()[0][0])


def trimestre(comp: int) -> int:
    """202305 -> 20232"""
    return (comp // 100) * 10 + ((comp % 100) - 1) // 3 + 1
