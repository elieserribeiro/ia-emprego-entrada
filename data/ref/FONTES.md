# Fontes e licenças dos arquivos de referência

| Arquivo | Origem | Licença / citação |
|---|---|---|
| `ilo2025_isco08_ocupacao.csv` | Gmyrek, P. et al. (2025). *Generative AI and Jobs: A Refined Global Index of Occupational Exposure*. ILO Working Paper 140. https://doi.org/10.54394/HETP0387. Dados: https://github.com/pgmyrek/2025_GenAI_scores_ISCO08 | CC BY 4.0 — citar a OIT/NASK |
| `ponte_cbo2002_isco08_ilo2025.csv` | CBO 2002→CIUO-88: Tábua de conversão CBO2002–CBO94–CIUO88 do MTE (cbo.mte.gov.br), raspada e versionada no pacote **ocupacoesBR** (Peixoto, 2026; https://github.com/moraespeixoto/ocupacoesBR, MIT). ISCO-88→ISCO-08: conversão de Ganzeboom & Treiman (https://www.harryganzeboom.nl/isco08/). | MIT (ocupacoesBR); citar Ganzeboom & Treiman e Peixoto (2026) |
| `mte_tabela5_sem_ajustes.csv`, `mte_tabela5_1_com_ajustes.csv` | MTE/PDET, Novo CAGED — Tabelas de junho/2026 (Tabelas 5 e 5.1), usadas só para validar a ingestão | dados públicos |
| `ipca.csv` (gerado na Etapa 1) | IBGE/SIDRA, tabela 1737, variável 2266 | dados públicos |

Limitações a reportar no artigo:

- A tábua do MTE cobre diretamente metade das ocupações da CBO. As demais entram pela família ou pela hierarquia.
- 186 dos 530 códigos ISCO-88 têm mais de um destino na ISCO-08.
- A cobertura **ponderada pelas admissões** sai em `outputs/validacao/cobertura_exposicao.csv` e deve ser relatada.
