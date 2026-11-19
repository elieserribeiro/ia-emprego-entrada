-- Tabela das classificações CONSENTIDAS do painel (LGPD: minimização de dados).
-- Rode no SQL Editor do Supabase. Só campos categóricos; NENHUM texto livre, IP, e-mail ou identificador.
create table if not exists public.respostas (
  id            bigint generated always as identity primary key,
  mes           text not null check (mes ~ '^\d{4}-\d{2}$'),            -- AAAA-MM, sem dia nem hora
  cbo           text not null check (cbo ~ '^\d{6}$'),
  cbo_familia   text not null check (cbo_familia ~ '^\d{4}$'),
  exposicao     text not null check (exposicao in ('alta','media','baixa')),
  natureza      text not null check (natureza in ('substituicao','transformacao','complementaridade','pouco_afetada')),
  sinal_br      text check (sinal_br in ('queda','estável','alta','dados insuficientes')),
  faixa_etaria  text not null check (faixa_etaria in ('14-17','18-24','25-29','30-39','40-49','50+')),
  escolaridade  text not null check (escolaridade in ('fundamental','medio','superior','pos')),
  uf            text not null check (uf ~ '^[A-Z]{2}$'),
  tempo_funcao  text not null check (tempo_funcao in ('menos_1','1_3','3_10','mais_10')),
  percepcao     text check (percepcao in ('nada','pouco','moderado','muito','nao_sei'))
);

-- RLS ligada e SEM políticas: o público (chave anon do site) não lê nem grava.
-- Só a API, com a service key guardada no servidor, insere e lê.
alter table public.respostas enable row level security;
revoke all on public.respostas from anon, authenticated;

-- Relatório já com supressão k >= 10 (para consultas manuais no painel do Supabase).
create or replace view public.relatorio_exposicao_idade_k10 with (security_invoker = true) as
select exposicao, faixa_etaria, count(*) as n
from public.respostas
group by exposicao, faixa_etaria
having count(*) >= 10;
revoke all on public.relatorio_exposicao_idade_k10 from anon, authenticated;

-- Retenção (opcional): apagar respostas com mais de 5 anos.
-- delete from public.respostas where mes < to_char(now() - interval '5 years', 'YYYY-MM');
