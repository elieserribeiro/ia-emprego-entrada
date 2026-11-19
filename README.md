# IA generativa e emprego de entrada no Brasil

**Pergunta:** a IA generativa reduziu as oportunidades de entrada (jovens, primeiro emprego, recém-formados, base da escada salarial) nas ocupações mais expostas?
**Desenho:** diferença tripla em *event study* trimestral, com jovens × adultos e mais × menos expostos à IA. Usa todos os meses do Novo CAGED desde jan/2020, agregados em trimestres. O modelo principal começa em 2022T1, a referência é 2022T3 (último trimestre inteiro antes do ChatGPT), e 2022T4 é transição. A pandemia é isolada com controles medidos no próprio CAGED (ver abaixo). As fontes são os microdados do Novo CAGED e da PNAD Contínua, e a exposição vem do índice OIT 2025.

## 0. Instalar (Windows + VS Code)

```powershell
git init                                  # ou clone o seu repositório vazio e copie estes arquivos
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q -s                              # teste com dados simulados: deve passar em ~2 min
```
Recomendado: [7-Zip](https://www.7-zip.org/) instalado. A ingestão o usa automaticamente e fica muito mais rápida que o `py7zr`. Reserve espaço em disco para os microdados brutos e os Parquet, algo na casa de dezenas de GB.

O teste gera microdados mensais falsos (jan/2020–jun/2026) com três coisas: um choque da pandemia e um boom de 2022, ambos correlacionados com a exposição, e um efeito da IA de −15% nas admissões de jovens por desvio-padrão de exposição a partir de 2023T1. Ele confere se o pipeline: (a) bate os totais de validação; (b) recupera o efeito com os controles da pandemia (≈ −0,15, contra ≈ −0,28 sem eles); (c) não acusa pré-tendência em 2022T1–T2.

## Etapas (rode uma por vez e me envie o que está em "devolver")

| # | Comando | O que faz | Devolver |
|---|---|---|---|
| 1 | `python -m iaemprego baixar-caged` | FTP do MTE: CAGEDMOV, CAGEDFOR e CAGEDEXC de todos os meses desde jan/2020, retomável (~78 meses × 3 arquivos) | o log final |
| 1 | `python -m iaemprego baixar-pnad` | IBGE: PNAD Contínua, todos os trimestres desde 2020 + input (layout) | o log |
| 1 | `python -m iaemprego baixar-ipca` | SIDRA 1737: IPCA número-índice → `data/ref/ipca.csv` | — |
| 2 | `python -m iaemprego ingerir` | .7z → Parquet (DuckDB), nomes de colunas normalizados | o log |
| 2 | `python -m iaemprego validar` | totais mensais × Tabelas 5 e 5.1 do MTE, em todos os meses; frequências dos códigos. A Tabela 5 (só MOV) deve bater quase exatamente. Na 5.1, os meses recentes podem dar admissões a mais, porque você baixou ajustes posteriores à publicação das tabelas | `outputs/validacao/*.csv` |
| 3 | `python -m iaemprego exposicao` | escore OIT por CBO, imputação pela hierarquia, cobertura ponderada por admissões de jan–set/2022 | `cobertura_exposicao.csv`, `exposicao_top40_cbo.csv` |
| 4 | `python -m iaemprego painel` | células ocupação × idade × UF × trimestre, com as medidas de entrada; choque e boom da pandemia por família | o log, `validacao/correlacao_expo_pandemia.csv` |
| 5 | `python -m iaemprego descritivas` | 1S2022 × 1S2023 × 1S2026 por quartil de exposição e idade (Brasil, NE, RN); índice trimestral em soma móvel de 4 trimestres | `outputs/tabelas/desc_*.csv`, figura |
| 6 | `python -m iaemprego modelos` | modelo principal (PPML) com e sem controles da pandemia, para cada medida de entrada; série completa 2020T1+ (descritiva); dupla diferença por faixa; salários | `modelos_principais.csv`, `figuras/event_*.png` |
| 7 | `python -m iaemprego heterogeneidade` | sexo, raça, escolaridade, região, Nordeste, RN | `heterogeneidade.csv` |
| 8 | `python -m iaemprego pnad` | desocupação de jovens com superior por trimestre; ocupados e recém-entrados por ocupação (mesmo desenho trimestral) | `pnad_*.csv`, figuras |
| 9 | `python -m iaemprego robustez` | sem controles da pandemia; série completa e desde 2021 com efeito fixo por fase; base 2023; só 1º semestres; jovem 18–29; exposição alternativa; sem imputadas; sem adm. pública/TI; sem aprendiz/intermitente; só MOV | `robustez.csv` |

`python -m iaemprego tudo` roda as etapas 2 a 9, sem os downloads.

**Antes da Etapa 4:** confira `outputs/validacao/frequencias_codigos.csv` e o layout oficial baixado na Etapa 1. Os códigos da seção `codigos` do `config.yaml` precisam bater com eles: primeiro emprego, superior completo, sexo, raça, aprendiz. Se "primeiro emprego" (código 10) quase não aparecer, essa medida sai da análise, e o modelo pula sozinho desfechos sem dados.

## O modelo principal

$$E[Y_{caut}] = \exp\Big(\sum_{t\neq 2022T3}\beta_t\,\text{Expo}_c\,\text{Jovem}_a\,\mathbb 1[t] + \sum_t \phi_t\,\text{Choque}_c\,\text{Jovem}_a\,\mathbb 1[t] + \sum_t \psi_t\,\text{Boom}_c\,\text{Jovem}_a\,\mathbb 1[t] + \alpha_{ca} + \gamma_{ct} + \delta_{aut}\Big)$$

- **Os índices:** *c* = família CBO (4 dígitos), *a* = faixa etária, *u* = UF, *t* = trimestre (2022T1 em diante).
- **Os efeitos fixos:**
  - $\gamma_{ct}$ (ocupação × trimestre) absorve qualquer choque da ocupação comum a todas as idades, inclusive a IA, a pandemia e a sazonalidade da ocupação. Por isso β_t mede só o efeito *diferencial* sobre a entrada.
  - $\delta_{aut}$ absorve o ciclo e a sazonalidade por idade e UF: juros, salário mínimo, políticas para jovens.
  - No pooled entra também a sazonalidade própria de ocupação × idade × trimestre do ano. No event study ela é colinear com as dummies de trimestre e fica de fora.
- **Os coeficientes:** β_2022T1 e β_2022T2 são o teste de pré-tendência. O pooled usa `treat_post` (2023T1+ contra 2022T1–T3, sem 2022T4).
- **Estimação:** PPML (`pyfixest.fepois`) com erros clusterizados por ocupação.

### Como a pandemia é isolada

A pandemia não atingiu todas as ocupações igualmente, e pode ter atingido mais os jovens justamente nas ocupações expostas à IA (escritório, comércio). Se essas ocupações tiveram um boom de contratação de jovens em 2021–22 e depois voltaram ao normal, o retorno ao normal pareceria um efeito da IA. Três camadas tratam isso:

1. **Amostra principal desde 2022T1**, longe do pior da pandemia. Os anos 2020–21 servem para medir os choques e entram nas robustezes.
2. **Controles medidos no próprio CAGED**, por família CBO, sempre como diferença jovem − adulto (o que cancela o que é comum à ocupação):
   - *Choque* = variação do log das admissões de jovens entre jan–fev/2020 e abr–jun/2020, menos a mesma variação para adultos.
   - *Boom* = o mesmo entre jan–fev/2020 e jan–fev/2022 (mesmos meses, sem sazonalidade).
   Ambos entram padronizados e interagidos com Jovem × cada trimestre, com coeficientes livres. Assim, qualquer trajetória pós-pandemia proporcional ao choque ou ao boom é absorvida. `correlacao_expo_pandemia.csv` mostra o quanto exposição e pandemia se confundem: quanto menor a correlação, mais separáveis são.
3. **Robustezes:** sem os controles; série completa 2020T1+ e desde 2021 com efeito fixo ocupação × idade × fase (pré, pandemia 2020T2–2021T4, pós); base 2023 (pergunta se houve *aceleração* depois de 2023, quando a adoção da IA cresceu; não capta um efeito que já existia em 2023); só 1º semestres.

No teste com dados simulados, ignorar a pandemia quase dobra o efeito estimado (−0,28 contra −0,15 verdadeiro), e os controles o recuperam. Uma limitação: em famílias pequenas, choque e boom são medidos com ruído, e isso deixa um viés residual. Por isso famílias com menos de 20 admissões em jan–fev/2020 ficam sem a medida, e recebem a média.

## Medidas de entrada (desfechos)

| Variável | Definição |
|---|---|
| `adm` | admissões (com ajustes FOR/EXC até 12 meses, como o MTE) |
| ~~`adm_1emp`~~ | primeiro emprego: **fora dos modelos**. A partir de 2022, 99,9% das admissões vêm com tipo 97 ("ignorado", eSocial); o campo deixa de ser informativo (`scripts/diagnostico_tipomov.py`) |
| `adm_sup` | admissões com superior completo ou mais |
| `adm_base` | admissões com salário real ≤ p25 da família CBO em jan–set/2022 |
| `adm_aprendiz` | admissões de aprendizes |
| `lsal` | log do salário real de admissão (R$ de jun/2026, IPCA) |

## Exposição à IA

A exposição é o escore médio das tarefas da ocupação no índice OIT 2025 (Gmyrek et al., WP 140), em ISCO-08 de 4 dígitos. A ponte até a CBO 2002 tem três passos:

1. **CBO 2002 → CIUO/ISCO-88:** pela tábua oficial do MTE, por ocupação. Quando a ocupação não está na tábua, usa a família majoritária e depois o prefixo da hierarquia.
2. **ISCO-88 → ISCO-08:** pela conversão de Ganzeboom & Treiman.
3. **ISCO-08 → escore:** média da OIT a 4 dígitos, com recuo para 3 e 2 dígitos quando o código é genérico.

Regenere a ponte com `scripts/gerar_ponte.py`. As fontes e licenças estão em `data/ref/FONTES.md`. Na PNAD, o código de ocupação (COD) é a própria ISCO-08 em quase todos os grupos.

## Painel interativo

`painel_ia/` é o painel do site: a pessoa descreve o cargo, confirma a CBO e vê a **exposição** da ocupação à IA,
em três camadas (OIT, natureza, contratações no CAGED). Tem modelo de texto ajustado no VS Code, API em FastAPI
e componente para o Lovable. Veja `painel_ia/README.md`.

## Estrutura

```
config.yaml          períodos (trimestres, janelas da pandemia), faixas etárias, códigos
src/iaemprego/       download, caged (ingestão/validação), exposicao, painel, descritivas, modelos, pnad
data/ref/            ponte CBO→OIT, totais oficiais do MTE (validação), ipca.csv  [versionado]
data/raw|interim|processed, outputs/                                             [fora do Git]
tests/               teste ponta a ponta com dados simulados
painel_ia/           painel interativo (modelo CBO, API, Supabase, Lovable)
```
