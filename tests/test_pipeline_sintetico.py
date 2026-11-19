"""Teste ponta a ponta com microdados SIMULADOS (sem rede).

Gera CAGED mensal jan/2020–jun/2026 (MOV/FOR/EXC em .7z, layout do MTE) e PNAD trimestral falsos.
Processo gerador (log das admissões de JOVENS 18–24 na família c, relativo a adultos):
  pandemia: −0.6·s_c em abr–jun/2020; −0.3·s_c de jul/2020 a dez/2021
  boom:     +0.4·b_c em 2022T1, +0.3 em 2022T2, +0.2 em 2022T3, +0.1 em 2022T4, 0 depois
  IA:       BETA·expo_z a partir de 2023T1
s e b são correlacionados com a exposição (0.5·z + ruído), então ignorar a pandemia VIESA o estimador
(o fim do boom parece queda nas ocupações expostas). Com os controles, o pipeline deve recuperar BETA.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
BETA = -0.15
CAB = ("competênciamov;região;uf;município;seção;subclasse;saldomovimentação;cbo2002ocupação;categoria;"
       "graudeinstrução;idade;horascontratuais;raçacor;sexo;tipoempregador;tipoestabelecimento;"
       "tipomovimentação;tipodedeficiência;indtrabintermitente;indtrabparcial;salário;tamestabjan;"
       "indicadoraprendiz;origemdainformação;competênciadec;indicadordeforadoprazo;unidadesaláriocódigo;"
       "valorsaláriofixo")


def _cbos(rng, n=80):
    p = pd.read_csv(RAIZ / "data/ref/ponte_cbo2002_isco08_ilo2025.csv", dtype={"cbo2002": str})
    p = p.dropna(subset=["ilo_mean_2025"])
    # uma CBO por família, para a exposição da família ser a da CBO
    p = p.groupby(p.cbo2002.str[:4]).head(1).sample(n, random_state=1)
    return p.cbo2002.to_numpy(), p.ilo_mean_2025.to_numpy()


def _log_jovem(comp, expo_z, choq, boom):
    """Log do fator multiplicativo das admissões de jovens por família, no mês `comp`."""
    t = (comp // 100) * 10 + ((comp % 100) - 1) // 3 + 1
    L = np.zeros_like(expo_z)
    if 202004 <= comp <= 202006:
        L = L - 0.6 * choq
    elif 202007 <= comp <= 202112:
        L = L - 0.3 * choq
    L = L + {20221: 0.4, 20222: 0.3, 20223: 0.2, 20224: 0.1}.get(t, 0.0) * boom
    if t >= 20231:
        L = L + BETA * expo_z
    return L - L.max()  # P(manter) ≤ 1; a constante do mês é absorvida pelos efeitos fixos faixa × UF × t


def _linhas(rng, comp, n, cbos, expo_z, choq, boom):
    i = rng.integers(0, len(cbos), n)
    idade = rng.choice(np.r_[16, 18, 20, 22, 24, 27, 33, 38, 45, 52], n)
    adm = rng.random(n) < 0.52
    jovem = (idade >= 18) & (idade <= 24)
    manter = rng.random(n) < np.exp(_log_jovem(comp, expo_z, choq, boom)[i])
    keep = ~(adm & jovem) | manter
    i, idade, adm = i[keep], idade[keep], adm[keep]
    n = len(i)
    uf = rng.choice([24, 35, 33, 29, 43], n)
    reg = np.select([uf == 24, uf == 29, uf == 35, uf == 33], [2, 2, 3, 3], 4)
    sal = np.round(rng.lognormal(7.6, 0.4, n), 2)
    df = pd.DataFrame({
        "c": comp, "r": reg, "uf": uf, "mun": 240810, "sec": rng.choice(list("CGJNO"), n), "sub": "6201501",
        "s": np.where(adm, 1, -1), "cbo": cbos[i], "cat": 101, "instr": rng.choice([5, 7, 7, 9], n),
        "idade": idade, "h": 44, "raca": rng.choice([1, 2, 3, 9], n), "sexo": rng.choice([1, 3], n),
        "te": 0, "tes": 1, "tm": np.where(adm, rng.choice([10, 20, 20], n), 31), "def": 0,
        "inter": (rng.random(n) < 0.02).astype(int), "parc": 0,
        "sal": [f"{v:.2f}".replace(".", ",") for v in sal], "tam": 5,
        "apr": (rng.random(n) < 0.03).astype(int), "orig": 1, "cdec": comp, "fp": 0, "us": 5, "vs": 0})
    return df


def _gravar_7z(df, destino: Path):
    txt = CAB + "\n" + "\n".join(";".join(map(str, r)) for r in df.itertuples(index=False))
    with py7zr.SevenZipFile(destino, "w") as z:
        z.writestr(txt.encode("utf-8"), destino.stem + ".txt")


MESES = [int(p.strftime("%Y%m")) for p in pd.period_range("2020-01", "2026-06", freq="M")]


def _gerar_caged(raw: Path, ref: Path):
    rng = np.random.default_rng(0)
    cbos, expo = _cbos(rng, 40)
    z = (expo - expo.mean()) / expo.std()
    choq = 0.5 * z + rng.normal(0, 0.85, len(z))
    boom = 0.5 * z + rng.normal(0, 0.85, len(z))
    (raw / "caged").mkdir(parents=True)
    tot5, tot51 = [], {}
    for comp in MESES:
        mov = _linhas(rng, comp, 14000, cbos, z, choq, boom)
        _gravar_7z(mov, raw / "caged" / f"CAGEDMOV{comp}.7z")
        a, d = int((mov.s == 1).sum()), int((mov.s == -1).sum())
        tot5.append(dict(ano=comp // 100, mes=comp % 100, estoque=0, admissoes=a, desligamentos=d, saldo=a - d))
        tot51[comp] = [a, d]
    # FOR/EXC: publicados no mês seguinte, referentes à competência anterior
    for comp, pub in zip(MESES[:-1], MESES[1:]):
        fo = _linhas(rng, comp, 250, cbos, z, choq, boom)
        ex = _linhas(rng, comp, 50, cbos, z, choq, boom)
        _gravar_7z(fo, raw / "caged" / f"CAGEDFOR{pub}.7z")
        _gravar_7z(ex, raw / "caged" / f"CAGEDEXC{pub}.7z")
        tot51[comp][0] += int((fo.s == 1).sum()) - int((ex.s == 1).sum())
        tot51[comp][1] += int((fo.s == -1).sum()) - int((ex.s == -1).sum())
    pd.DataFrame(tot5).to_csv(ref / "mte_tabela5_sem_ajustes.csv", index=False)
    pd.DataFrame([dict(ano=c // 100, mes=c % 100, estoque=0, admissoes=v[0], desligamentos=v[1], saldo=v[0] - v[1])
                  for c, v in tot51.items()]).to_csv(ref / "mte_tabela5_1_com_ajustes.csv", index=False)
    meses = pd.period_range("2020-01", "2026-12", freq="M")
    pd.DataFrame({"ano_mes": [int(p.strftime("%Y%m")) for p in meses],
                  "indice": 5000 * 1.004 ** np.arange(len(meses))}).to_csv(ref / "ipca.csv", index=False)


PNAD_LAYOUT = [("Ano", 4), ("Trimestre", 1), ("UF", 2), ("V1028", 15), ("V2007", 1), ("V2009", 3), ("V2010", 1),
               ("VD3004", 1), ("VD4001", 1), ("VD4002", 1), ("V4010", 4), ("VD4009", 2), ("V4040", 1)]


def _gerar_pnad(raw: Path):
    rng = np.random.default_rng(1)
    (raw / "pnad").mkdir(parents=True)
    ilo = pd.read_csv(RAIZ / "data/ref/ilo2025_isco08_ocupacao.csv", dtype={"ISCO_08": str})
    cods = ilo.ISCO_08.sample(60, random_state=2).to_numpy()
    pos, linhas = 1, []
    for v, w in PNAD_LAYOUT:
        linhas.append(f"@{pos:04d} {v} $" + f"{w}." if v != "V1028" else f"@{pos:04d} {v} {w}.")
        pos += w
    with zipfile.ZipFile(raw / "pnad" / "Dicionario_e_input.zip", "w") as zf:
        zf.writestr("input_PNADC_trimestral.txt", "input\n" + "\n".join(linhas) + "\n;")
    for ano in range(2020, 2027):
        for t in ((1, 2) if ano == 2026 else (1, 2, 3, 4)):
            n = 2500
            idade = rng.integers(14, 65, n)
            ocup = rng.random(n) < 0.6
            reg = []
            for k in range(n):
                vals = [str(ano), str(t), "24", f"{rng.uniform(500, 1500):015.4f}", "1", f"{idade[k]:03d}", "1",
                        str(rng.choice([5, 7])), "1", "1" if ocup[k] else "2",
                        rng.choice(cods) if ocup[k] else "    ", "01", str(rng.choice([1, 2, 3, 4])) if ocup[k] else " "]
                reg.append("".join(v.rjust(w) if v.strip() else " " * w for v, (_, w) in zip(vals, PNAD_LAYOUT)))
            with zipfile.ZipFile(raw / "pnad" / f"PNADC_0{t}{ano}.zip", "w") as zf:
                zf.writestr(f"PNADC_0{t}{ano}.txt", "\n".join(reg))


@pytest.fixture(scope="module")
def cfg_sintetico(tmp_path_factory):
    base = tmp_path_factory.mktemp("proj")
    ref = base / "data/ref"
    ref.mkdir(parents=True)
    for f in ("ponte_cbo2002_isco08_ilo2025.csv", "ilo2025_isco08_ocupacao.csv"):
        shutil.copy(RAIZ / "data/ref" / f, ref / f)
    _gerar_caged(base / "data/raw", ref)
    _gerar_pnad(base / "data/raw")
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    (base / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    from iaemprego.util import carregar_config
    return carregar_config(base / "config.yaml")


def test_pipeline_completo(cfg_sintetico):
    from iaemprego import caged, descritivas, exposicao, modelos, painel, pnad
    cfg = cfg_sintetico
    caged.ingerir(cfg)
    val = caged.validar(cfg)
    assert val.dif_admissoes.abs().max() == 0, "totais dos microdados devem bater com a referência"
    assert val.dif_desligamentos.abs().max() == 0
    exposicao.construir(cfg)
    p = painel.construir(cfg)
    assert {"adm", "adm_1emp", "adm_sup", "adm_base", "sal_n"} <= set(p.columns)
    assert p.adm_1emp.sum() > 0 and p.adm_base.sum() > 0
    pand = pd.read_parquet(Path(cfg["_caminhos"]["processed"]) / "pandemia_cbo4.parquet")
    assert pand.choque_z.notna().all() and len(pand) >= 35
    corr = pd.read_csv(Path(cfg["_caminhos"]["outputs"]) / "validacao/correlacao_expo_pandemia.csv")
    print("correlação exposição × pandemia:", corr.round(3).to_dict("records"))
    assert corr.choque_z.iloc[0] < -0.2 and corr.boom_z.iloc[0] > 0.2, "DGP: choque e boom correlacionados c/ expo"
    descritivas.rodar(cfg)
    res = modelos.rodar_principal(cfg)
    adm = res[res.desfecho == "adm"]
    com = adm[adm.modelo == "tripla_pooled_com_controles"].coef.iloc[0]
    sem = adm[adm.modelo == "tripla_pooled_sem_controles"].coef.iloc[0]
    print(f"beta pooled: com controles {com:.3f} | sem controles {sem:.3f} | verdadeiro {BETA}")
    assert -0.21 < com < -0.09, f"com controles deveria recuperar {BETA}; estimou {com}"
    assert com - sem > 0.02, "sem controles deveria exagerar a queda (o fim do boom confunde)"
    ev = adm[adm.modelo == "tripla_event_com_controles"].set_index("periodo").coef
    print("event study:", ev.round(3).to_dict())
    assert abs(ev.loc[20221]) < 0.08 and abs(ev.loc[20222]) < 0.08, "pré-tendência deveria ser ~0"
    dup = res[(res.modelo == "dupla_por_faixa_pooled") & (res.termo == "expo_post")].set_index("faixa").coef
    print("dupla por faixa (pooled, vs. mesmo trimestre de 2022):", dup.round(3).to_dict())
    assert -0.22 < dup.loc["18_24"] < -0.08, "dupla 18–24 deveria captar o efeito nos jovens"
    assert abs(dup.loc["30_39"]) < 0.05, "dupla 30–39 deveria ser ~0 (adultos sem efeito)"
    assert (res.modelo == "aprendizes_14_24_pooled").any(), "modelo de aprendizes entre jovens"
    t2529 = res[(res.modelo == "tripla_25_29_pooled_com_controles") & (res.desfecho == "adm")].coef.iloc[0]
    ep2529 = res[(res.modelo == "tripla_25_29_pooled_com_controles") & (res.desfecho == "adm")].ep.iloc[0]
    print(f"tripla 25–29 (sem efeito no DGP): {t2529:.3f} (ep {ep2529:.3f})")
    assert abs(t2529) < 2.5 * ep2529 + 0.02, "25–29 não tem efeito no DGP"
    assert not (res.desfecho == "adm_aprendiz").loc[res.modelo.str.startswith("tripla")].any()
    modelos.rodar_heterogeneidade(cfg)
    rob = modelos.rodar_robustez(cfg)
    print(rob[rob.termo == "treat_post"][["grupo", "variante", "coef", "ep"]].round(3).to_string(index=False))
    assert rob.variante.nunique() >= 14 and set(rob.grupo) == {"18_24", "25_29"}
    r2529 = rob[(rob.grupo == "25_29") & (rob.variante == "principal") & (rob.termo == "treat_post")].coef.iloc[0]
    assert abs(r2529) < 0.12, "25–29 sem efeito no DGP"
    pnad.rodar(cfg)
    out = Path(cfg["_caminhos"]["outputs"])
    for f in ("tabelas/modelos_principais.csv", "tabelas/robustez.csv", "tabelas/heterogeneidade.csv",
              "tabelas/pnad_desocupacao_faixa_superior.csv", "figuras/event_adm.png",
              "figuras/event_adm_serie_completa.png", "figuras/dupla_por_faixa.png",
              "figuras/event_aprendizes_14_24.png", "figuras/event_25_29_adm.png", "figuras/indice_admissoes_quartil.png",
              "tabelas/desc_indice_trimestral_movel4.csv", "validacao/cobertura_exposicao.csv"):
        assert (out / f).exists(), f
