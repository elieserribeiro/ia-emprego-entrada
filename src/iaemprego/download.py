"""Etapa 1 — Download dos microdados.

Novo CAGED (FTP do MTE): para cada mês de publicação desde jan/2020, CAGEDMOV, CAGEDFOR e CAGEDEXC
(FOR/EXC trazem ajustes de competências anteriores).
PNAD Contínua (HTTP do IBGE): todos os trimestres de 2020 em diante + arquivo de input (layout).
Downloads são retomáveis: um arquivo com o tamanho do servidor é pulado.
"""
from __future__ import annotations

import ftplib
import re
import time
from pathlib import Path

import requests

from .util import caminho, log, meses_publicacao


# ----------------------------------------------------------------------------- CAGED
def _ftp(cfg):
    ftp = ftplib.FTP(cfg["caged"]["ftp_host"], timeout=180)
    ftp.login()
    ftp.encoding = "latin-1"
    ftp.set_pasv(True)
    return ftp


def _ultimo_mes_disponivel(ftp, base: str) -> str:
    anos = sorted(n.split("/")[-1] for n in ftp.nlst(base) if n.split("/")[-1].isdigit())
    meses = sorted(n.split("/")[-1] for n in ftp.nlst(f"{base}/{anos[-1]}") if n.split("/")[-1].isdigit())
    return meses[-1]


def _baixar_ftp(ftp, remoto: str, destino: Path, tentativas=3):
    try:
        tamanho = ftp.size(remoto)
    except ftplib.error_perm:
        log.warning("ausente no servidor: %s", remoto)
        return False, ftp
    if destino.exists() and destino.stat().st_size == tamanho:
        return True, ftp
    for t in range(tentativas):
        try:
            inicio = destino.stat().st_size if destino.exists() else 0
            with open(destino, "ab" if inicio else "wb") as fh:
                ftp.retrbinary(f"RETR {remoto}", fh.write, rest=inicio or None)
            if destino.stat().st_size == tamanho:
                log.info("ok %s (%.1f MB)", destino.name, tamanho / 1e6)
                return True, ftp
        except (ftplib.all_errors) as e:  # conexão caiu: reconecta e retoma
            log.warning("falha %s (%s), tentativa %d", destino.name, e, t + 1)
            time.sleep(5)
            ftp = _ftp_reconectar(ftp)
    raise RuntimeError(f"não foi possível baixar {remoto}")


def _ftp_reconectar(ftp):
    try:
        ftp.quit()
    except Exception:
        pass
    host = ftp.host
    novo = ftplib.FTP(host, timeout=180)
    novo.login()
    novo.encoding = "latin-1"
    return novo


def baixar_caged(cfg):
    c = cfg["caged"]
    ftp = _ftp(cfg)
    fim = c["publicacao_fim"] or _ultimo_mes_disponivel(ftp, c["ftp_base"])
    meses = meses_publicacao(c["publicacao_inicio"], f"{fim[:4]}-{fim[4:]}")
    log.info("CAGED: meses de publicação %s a %s", meses[0], meses[-1])
    for am in meses:
        for tipo in ("MOV", "FOR", "EXC"):
            nome = f"CAGED{tipo}{am}.7z"
            _, ftp = _baixar_ftp(ftp, f"{c['ftp_base']}/{am[:4]}/{am}/{nome}", caminho(cfg, "raw", "caged", nome))
    # layout oficial (dicionário de códigos) — guardado junto para conferência
    try:
        for n in ftp.nlst(c["ftp_base"]):
            if "layout" in n.lower() or "dicion" in n.lower():
                _, ftp = _baixar_ftp(ftp, n, caminho(cfg, "raw", "caged", Path(n).name))
    except ftplib.all_errors:
        pass
    ftp.quit()


# ----------------------------------------------------------------------------- PNAD
def _listar_http(url: str) -> list[str]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return re.findall(r'href="([^"?/][^"]*)"', r.text)


def _baixar_http(url: str, destino: Path):
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if destino.exists() and total and destino.stat().st_size == total:
            return
        with open(destino, "wb") as fh:
            for bloco in r.iter_content(1 << 20):
                fh.write(bloco)
    log.info("ok %s (%.1f MB)", destino.name, destino.stat().st_size / 1e6)


def baixar_pnad(cfg):
    base = cfg["pnad"]["base_url"].rstrip("/")
    for ano in cfg["pnad"]["anos"]:
        try:
            arquivos = _listar_http(f"{base}/{ano}/")
        except requests.HTTPError:
            log.warning("PNAD %s: pasta ainda não existe", ano)
            continue
        for t in cfg["pnad"]["trimestres"]:
            alvo = [a for a in arquivos if re.match(rf"PNADC_0{t}{ano}.*\.zip$", a)]
            if not alvo:
                log.warning("PNAD %s T%s ainda não publicada", ano, t)
                continue
            _baixar_http(f"{base}/{ano}/{alvo[-1]}", caminho(cfg, "raw", "pnad", f"PNADC_0{t}{ano}.zip"))
    doc = [a for a in _listar_http(f"{base}/Documentacao/") if a.lower().startswith("dicionario_e_input")]
    if doc:
        _baixar_http(f"{base}/Documentacao/{doc[-1]}", caminho(cfg, "raw", "pnad", "Dicionario_e_input.zip"))


# ----------------------------------------------------------------------------- IPCA
def baixar_ipca(cfg):
    """IPCA número-índice mensal (SIDRA 1737/v2266) -> data/ref/ipca.csv (ano_mes, indice)."""
    import pandas as pd
    destino = caminho(cfg, "ref", "ipca.csv")
    r = requests.get(cfg["ipca"]["url"], timeout=120)
    r.raise_for_status()
    dados = r.json()
    cab = dados[0]  # primeira linha é o cabeçalho: descobre a chave do código do mês
    chave = next(k for k, v in cab.items() if str(v).startswith("Mês (Código)"))
    js = [x for x in dados[1:] if x["V"] not in ("...", "-", "")]
    df = pd.DataFrame({"ano_mes": [int(x[chave]) for x in js], "indice": [float(x["V"]) for x in js]})
    df.to_csv(destino, index=False)
    log.info("IPCA: %d meses, último %s", len(df), df.ano_mes.max())
