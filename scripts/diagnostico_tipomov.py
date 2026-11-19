"""Diagnóstico: como os tipos de admissão (primeiro emprego, reemprego, ignorado...) se distribuem por ano.

    python scripts/diagnostico_tipomov.py

Grava outputs/validacao/tipomov_por_ano.csv (admissões e % de cada tipo, por ano).
"""
from iaemprego.caged import conexao
from iaemprego.util import caminho, carregar_config

cfg = carregar_config()
con = conexao(cfg)
d = con.sql("""
    SELECT comp // 100 AS ano, tipomov, sum(CASE WHEN saldo_bruto = 1 THEN sinal ELSE 0 END) AS adm
    FROM mov_ajustado WHERE saldo_bruto = 1 GROUP BY ALL ORDER BY ALL""").df()
t = d.pivot(index="ano", columns="tipomov", values="adm").fillna(0)
pct = (100 * t.div(t.sum(axis=1), axis=0)).round(1).add_prefix("pct_")
out = t.astype("int64").add_prefix("adm_").join(pct)
out.to_csv(caminho(cfg, "outputs", "validacao", "tipomov_por_ano.csv"))
print(pct.to_string())
