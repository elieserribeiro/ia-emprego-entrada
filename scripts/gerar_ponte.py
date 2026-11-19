"""Regenera data/ref/ponte_cbo2002_isco08_ilo2025.csv e ilo2025_isco08_ocupacao.csv (reprodutibilidade).

Fontes (clonar antes, em qualquer pasta):
  git clone https://github.com/pgmyrek/2025_GenAI_scores_ISCO08      # OIT 2025, CC BY 4.0
  git clone https://github.com/moraespeixoto/ocupacoesBR             # tábua MTE CBO2002->CIUO88 e Ganzeboom 88->08, MIT
Uso:  pip install pyreadr openpyxl
      python scripts/gerar_ponte.py --ilo <pasta_ilo> --ocup <pasta_ocupacoesBR>
"""
import argparse
from pathlib import Path

import pandas as pd
import pyreadr

ap = argparse.ArgumentParser()
ap.add_argument("--ilo", required=True)
ap.add_argument("--ocup", required=True)
ap.add_argument("--saida", default=str(Path(__file__).resolve().parents[1] / "data" / "ref"))
a = ap.parse_args()
ler = lambda n: list(pyreadr.read_r(Path(a.ocup, "data", n + ".rda")).values())[0].astype(str).replace("nan", None)

# 1) OIT 2025: uma linha por ocupação ISCO-08 (média e DP dos escores das tarefas, gradiente)
ilo = pd.read_excel(Path(a.ilo, "Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx"))
occ = ilo.groupby("ISCO_08").agg(titulo=("Title", "first"), mean_score_2025=("mean_score_2025", "first"),
                                 sd_2025=("SD_2025", "first"), potential25=("potential25", "first"),
                                 mean_score_2023=("mean_score_2023", "first"), n_tarefas=("taskID", "count")).reset_index()
occ["ISCO_08"] = occ.ISCO_08.astype(int).astype(str).str.zfill(4)
occ.to_csv(Path(a.saida, "ilo2025_isco08_ocupacao.csv"), index=False)

# 2) Domínio oficial da CBO 2002 (2.777 ocupações) e pontes
dom = pd.read_csv(Path(a.ocup, "inst/extdata/fontes/cbo2002_dominio.txt"), header=None, dtype=str)[0].str.zfill(6)
d, fam, esc, x = ler("cbo2002_isco88"), ler("cbo2002_familia_isco88"), ler("cbo2002_escada"), ler("isco88_isco08")
out = pd.DataFrame({"cbo2002": dom}).merge(d[["cbo2002", "isco88"]].dropna(), how="left", on="cbo2002")
out["nivel_ponte"] = out.isco88.notna().map({True: "ocupacao", False: None})
out["familia"] = out.cbo2002.str[:4]
f = out.isco88.isna()
out.loc[f, "isco88"] = out.loc[f, "familia"].map(fam.dropna(subset=["isco88"]).set_index("familia").isco88)
out.loc[f & out.isco88.notna(), "nivel_ponte"] = "familia"
for n in ("3", "2"):
    e = esc[esc.nivel == n].set_index("prefixo").isco88
    f = out.isco88.isna()
    out.loc[f, "isco88"] = out.loc[f, "cbo2002"].str[: int(n)].map(e)
    out.loc[f & out.isco88.notna(), "nivel_ponte"] = f"prefixo{n}"
xx = x.set_index("isco88")
out["isco08"] = out.isco88.map(xx.isco08)
out["isco88_isco08_alternativas"] = pd.to_numeric(out.isco88.map(xx.n_alternativas))
o = occ.set_index("ISCO_08")
out["ilo_mean_2025"] = out.isco08.map(o.mean_score_2025)
out["ilo_sd_2025"] = out.isco08.map(o.sd_2025)
out["ilo_gradiente"] = out.isco08.map(o.potential25)
out["ilo_nivel"] = out.ilo_mean_2025.notna().map({True: "isco4", False: None})
# Ganzeboom às vezes devolve um grupo genérico (ex.: 2500): recua para a média de 3 e 2 dígitos
for k in (3, 2):
    g = occ.assign(p=occ.ISCO_08.str[:k]).groupby("p").mean_score_2025.mean()
    f = out.ilo_mean_2025.isna() & out.isco08.notna()
    out.loc[f, "ilo_mean_2025"] = out.loc[f, "isco08"].str[:k].map(g)
    out.loc[f & out.ilo_mean_2025.notna(), "ilo_nivel"] = f"isco{k}"
out.to_csv(Path(a.saida, "ponte_cbo2002_isco08_ilo2025.csv"), index=False)
print(out.nivel_ponte.value_counts(dropna=False), out.ilo_nivel.value_counts(dropna=False), sep="\n")
