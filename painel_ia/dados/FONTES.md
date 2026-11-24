# Fontes e licenças dos arquivos de dados do painel

Complementa `data/ref/FONTES.md`, que descreve a ponte CBO→ISCO-08, as tabelas do MTE e o IPCA.

| Arquivo | Origem | Licença / citação |
|---|---|---|
| `ilo2025_tarefas_isco08.csv` | Gmyrek, P. et al. (2025). *Generative AI and Jobs: A Refined Global Index of Occupational Exposure*. ILO Working Paper 140. https://doi.org/10.54394/HETP0387. Dados: https://github.com/pgmyrek/2025_GenAI_scores_ISCO08 | CC BY 4.0: citar a OIT/NASK |
| `tarefas_pt.csv` | Tradução para o português das tarefas do arquivo acima, feita por Eliéser Ribeiro com apoio de tradução automática por IA (script `painelia/traduzir_tarefas.py`). Obra derivada: a alteração é a tradução. | CC BY 4.0 (herdada): citar a OIT/NASK. Tradução não oficial, não revisada nem endossada pela OIT |
| `cbo_titulos_tabua_mte.csv` | Títulos das ocupações da tábua de conversão CBO2002–CBO94–CIUO88 do MTE (cbo.mte.gov.br) | Dados públicos do MTE |
| `cbo_mte/` | Arquivos da estrutura da CBO 2002 (ocupação, sinônimo, família, perfil ocupacional), baixados da área de downloads da CBO, no site do MTE | Dados públicos do MTE |
| `ajustes_ponte.csv` | Correções manuais da ponte CBO→ISCO-08, elaboradas pelos autores | Autoria própria |
| `consultas_teste.csv` | Cargos de teste para avaliar o buscador de CBO, elaborados pelos autores | Autoria própria |
