# Painel: exposição da sua ocupação à IA generativa

A pessoa descreve o cargo e as tarefas. O modelo sugere 3 ocupações (CBO), ela confirma uma e informa o perfil.
O painel então mostra três camadas, sempre de **exposição, não de impacto**:

| Camada | O que mostra | De onde vem |
|---|---|---|
| 1. Exposição: alta, média ou baixa | gradientes oficiais da OIT: G3–4 = alta; G1–2 = média; mínima ou não exposta = baixa. Mostra também o percentil entre as ocupações brasileiras | OIT 2025 (WP 140), regra da Tabela 5 aplicada à média e ao desvio-padrão das tarefas |
| 2. Natureza provável | substituição de tarefas (G4), transformação (G3, ou G2 com ≥ 40% das tarefas expostas), complementaridade (G1–2, mínima com tarefas expostas) ou pouco afetada | escores das tarefas da OIT |
| 3. Contratações no Brasil | queda, estável ou alta das admissões da família CBO para a idade (e a escolaridade) da pessoa, frente ao mercado, dos 4 trimestres até 2022T3 aos 4 mais recentes, com nível de confiança. Traz também o efeito estimado no artigo | microdados do Novo CAGED, pelo pipeline do artigo |

O texto explicativo é escrito pelo GPT (OpenAI) a partir **só** desses dados, e personalizado pelo perfil (início
de carreira × experiência, capacidade de adaptação). Sem chave, ou se a OpenAI falhar, sai um texto de reserva
determinístico.

```
painel_ia/
  painelia/            textos (base CBO), classificacao (camadas 1–2), evidencia_br (camada 3),
                       busca + treinar (CBO a partir de texto), explicacao (GPT), armazenamento (LGPD), api (FastAPI)
  dados/               tarefas da OIT, títulos da tábua MTE, consultas_teste.csv, cbo_mte/ (você coloca)
  artefatos/           gerados: ocupacoes.json, titulos_cbo.json, indice_st/, indice_tfidf/, avaliações
  deploy/              Dockerfile e requirements da API, montar_space.py (Hugging Face Spaces)
  supabase/schema.sql  tabela das respostas consentidas (RLS ligada, só campos categóricos)
  lovable/             componente React, prompt para o Lovable, aviso de privacidade
  tests/               testes da API (privacidade, k ≥ 10, validação)
```

## Passo a passo (no VS Code, depois do pipeline do artigo)

**0. Instalar:**
```powershell
pip install -e ".\painel_ia[api,treino,dev]"
pytest -q painel_ia\tests                # ~30 s, sem rede
```

**1. Baixar a CBO oficial (títulos, sinônimos e perfil ocupacional).** No site da CBO do MTE, entre em
*Downloads*, baixe o pacote com a **estrutura da CBO** (CSVs `CBO2002 - Ocupacao`, `Sinonimo`, `Familia`,
`PerfilOcupacional`) e descompacte em `painel_ia\dados\cbo_mte\`. O script acha os arquivos pelo nome. Sem eles,
o treino usa só os ~1.400 títulos da tábua MTE, e o buscador fica bem mais fraco.

**2. Evidência brasileira (camada 3):**
```powershell
cd painel_ia
python -m painelia.evidencia_br          # lê data/processed/painel_*.parquet e outputs/tabelas do artigo
```

**3. Classificação das ocupações (camadas 1 e 2):**
```powershell
python -m painelia.classificacao
```
→ **me devolva `artefatos/distribuicao_exposicao.csv`.** Com os dados do CAGED, o `pct_ponderado` mostra a
distribuição por admissões. Só com os pesos das CBOs (sem CAGED), hoje fica assim: alta 6,9%, média 8,4%, baixa
84,7%. Antes de publicar, decidimos juntos se os cortes ficam como estão.

**4. Treinar o buscador:**
```powershell
python -m painelia.treinar --so-tfidf                      # linha de base, segundos
python -m painelia.treinar                                 # + ajuste fino (multilingual-e5-small, 2 épocas)
```
O ajuste fino roda em CPU (dezenas de minutos) ou GPU. → **me devolva `artefatos/avaliacao_buscador.csv`.** Ela
compara TF-IDF, o modelo sem ajuste (`st-0`) e o ajustado (`st`), em top-1 e top-3, em dois conjuntos:
sinônimos oficiais retidos do treino e as descrições escritas como um usuário escreveria
(`dados/consultas_teste.csv`). Aumente esse arquivo com exemplos reais, sem dados pessoais, e ele vira o
melhor termômetro. Só com os títulos, a linha de base TF-IDF dá top-3 = 46% nessas consultas.

**5. Testar a API localmente:**
```powershell
$env:OPENAI_API_KEY="sk-..."; $env:PAINEL_SQLITE="teste.db"; $env:ADMIN_TOKEN="troque"
uvicorn painelia.api:app --reload --no-access-log
```
Abra http://127.0.0.1:8000/docs e teste `/sugerir-cbo` e `/classificar`.

## Publicar

**Supabase:** no SQL Editor, rode `supabase/schema.sql`. Copie a *Project URL* e a *service_role key*
(Settings → API). A service key fica **só** no servidor da API, nunca no site.

**Hugging Face Spaces (Docker):**
1. Crie um Space com o SDK *Docker*, por exemplo `exposicao-ia`.
2. `python deploy/montar_space.py ..\space_exposicao_ia`, depois `git lfs install`, clone o Space, copie a pasta,
   `git add . && git commit -m "api" && git push`.
3. Em **Settings → Variables and secrets**, crie como *secrets*: `OPENAI_API_KEY`, `SUPABASE_URL`,
   `SUPABASE_SERVICE_KEY`, `ADMIN_TOKEN`. Como *variables*: `OPENAI_MODEL` (confira o nome de um modelo atual e
   barato da OpenAI) e `ORIGENS_PERMITIDAS` (domínios do seu site, separados por vírgula).
4. Endereço da API: `https://SEU-USUARIO-exposicao-ia.hf.space`. Teste `/saude`.

No plano gratuito, o Space "dorme" após dias sem uso, e a primeira consulta demora. O Render (Docker) é a
alternativa, com o mesmo Dockerfile, mudando a porta para `$PORT`.

**Lovable:** siga `lovable/prompt_lovable.md`.

**Relatório das respostas consentidas:**
`GET /relatorio?por=exposicao,faixa_etaria,percepcao`, com o cabeçalho `X-Admin-Token`. Células com menos de
10 respostas são suprimidas. Quando só uma célula seria suprimida, a menor visível também é, para não
reconstruir por diferença.

## LGPD: o que o código garante

Projeto de pesquisa **independente** (sem vínculo institucional). Responsável pelos dados: Eliéser Ribeiro,
horizontedigital.ai@gmail.com. Aviso completo: `lovable/aviso_privacidade.md` (vira a página `/privacidade`).

- **Perguntas:** cargo (texto livre, nunca guardado), ocupação confirmada, faixa etária, escolaridade, UF, tempo na
  função, uso de IA no trabalho, percepção (antes do resultado), sexo e raça/cor (estas duas opcionais).
- **Consentimento em duas camadas:** a caixa geral vem desmarcada, e sem ela nada é gravado. Raça/cor é dado
  sensível (LGPD, art. 11) e só é gravada com uma segunda caixa específica (`consentimento_sensivel`).
- **Minimização:** grava só campos categóricos (`armazenamento.CAMPOS`) e o mês, sem dia.
- **Nunca gravado:** texto livre, IP, e-mail, identificadores de navegador. Sem log de acesso (`--no-access-log`).
- **OpenAI:** recebe só a ocupação, dados públicos e categorias do perfil (sem percepção, sexo ou raça/cor), com
  `store=false`. A chave fica no servidor.
- **Supabase:** RLS ligada e sem acesso público. **Relatórios:** k ≥ 10. **Retenção:** 5 anos (`schema.sql`).

## Limites que o painel deixa explícitos
- **Exposição × efeito:** o índice da OIT mede capacidade técnica da IA nas tarefas, não demissões.
- **Camada 3:** é descritiva. As admissões caem ou sobem por muitos motivos, e o efeito causal estimado é o do
  artigo.
- **Ponte CBO → ISCO-08:** em parte das CBOs ela só chega a 3 ou 2 dígitos. Nesses casos as tarefas vêm de
  ocupações vizinhas e o resultado indica isso. As tarefas estão no original em inglês, e o texto do GPT as
  traduz.
