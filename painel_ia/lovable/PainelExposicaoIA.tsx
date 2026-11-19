// Painel "Exposição da sua ocupação à IA generativa" — componente para o site no Lovable (React + Vite + Tailwind).
// Configure a variável VITE_PAINEL_API_URL (ex.: https://SEU-USUARIO-exposicao-ia.hf.space).
// O site NUNCA fala com a OpenAI nem guarda nada: só chama a API, que tem as chaves.
import { useState } from "react";

const API = (import.meta as any).env?.VITE_PAINEL_API_URL ?? "";

type Sugestao = { cbo: string; titulo: string; similaridade: number };
type Tarefa = { tarefa: string; escore: number };
type Resultado = {
  cbo: string; titulo: string;
  exposicao: { nivel: "alta" | "media" | "baixa"; gradiente_rotulo: string; percentil: number; percentil_base: string };
  natureza: { tipo: string; rotulo: string; texto: string };
  tarefas_mais_expostas: Tarefa[]; tarefas_menos_expostas: Tarefa[];
  evidencia_br: null | { classe: string; variacao_relativa_pct?: number; confianca?: string; grupo_idade: string;
    janelas?: { antes: string; recente: string } };
  mensagem_perfil: string; explicacao: string; aviso: string; fontes: string[]; gravado: boolean;
};

const FAIXAS = ["14-17", "18-24", "25-29", "30-39", "40-49", "50+"];
const ESCOLAR = [["fundamental", "Até o fundamental"], ["medio", "Médio completo ou superior incompleto"],
  ["superior", "Superior completo"], ["pos", "Pós-graduação"]];
const TEMPO = [["menos_1", "Menos de 1 ano"], ["1_3", "1 a 3 anos"], ["3_10", "3 a 10 anos"], ["mais_10", "Mais de 10 anos"]];
const UFS = "AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO".split(" ");
const PERCEPCAO = [["nada", "Nada"], ["pouco", "Pouco"], ["moderado", "Moderadamente"], ["muito", "Muito"], ["nao_sei", "Não sei"]];
const COR = { alta: "bg-orange-100 text-orange-900 border-orange-300", media: "bg-amber-50 text-amber-900 border-amber-300",
  baixa: "bg-emerald-50 text-emerald-900 border-emerald-300" } as const;
const NOME = { alta: "Alta", media: "Média", baixa: "Baixa" } as const;

async function post<T>(caminho: string, corpo: unknown): Promise<T> {
  const r = await fetch(`${API}${caminho}`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(corpo) });
  if (!r.ok) throw new Error(`Erro ${r.status}`);
  return r.json();
}

export default function PainelExposicaoIA() {
  const [etapa, setEtapa] = useState<1 | 2 | 3 | 4>(1);
  const [texto, setTexto] = useState("");
  const [sugestoes, setSugestoes] = useState<Sugestao[]>([]);
  const [cbo, setCbo] = useState<Sugestao | null>(null);
  const [perfil, setPerfil] = useState({ faixa_etaria: "", escolaridade: "", uf: "", tempo_funcao: "" });
  const [percepcao, setPercepcao] = useState<string>("");
  const [consentimento, setConsentimento] = useState(false); // desmarcado por padrão (LGPD)
  const [res, setRes] = useState<Resultado | null>(null);
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState("");

  const sugerir = async () => {
    setErro(""); setCarregando(true);
    try {
      const r = await post<{ sugestoes: Sugestao[] }>("/sugerir-cbo", { texto });
      setSugestoes(r.sugestoes); setEtapa(2);
    } catch { setErro("Não consegui buscar agora. Tente de novo em instantes."); }
    setCarregando(false);
  };

  const classificar = async () => {
    if (!cbo) return;
    setErro(""); setCarregando(true);
    try {
      const r = await post<Resultado>("/classificar", { cbo: cbo.cbo, ...perfil,
        percepcao: percepcao || null, consentimento });
      setRes(r); setEtapa(4);
      setTexto(""); // o texto livre não fica nem na memória da página
    } catch { setErro("Não consegui calcular agora. Tente de novo em instantes."); }
    setCarregando(false);
  };

  const perfilOk = Object.values(perfil).every(Boolean);
  const caixa = "w-full rounded-lg border border-slate-300 p-2 bg-white";

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4 text-slate-800">
      <header>
        <h1 className="text-2xl font-semibold">Quanto a sua ocupação está exposta à IA generativa?</h1>
        <p className="mt-1 text-sm text-slate-600">
          Exposição não é previsão de perda de emprego: indica quanto das tarefas a IA consegue fazer hoje.
        </p>
      </header>

      {erro && <p className="rounded bg-red-50 p-2 text-sm text-red-800">{erro}</p>}

      {etapa === 1 && (
        <section className="space-y-3">
          <label className="block font-medium" htmlFor="texto">Qual é o seu cargo e o que você faz no dia a dia?</label>
          <textarea id="texto" className={caixa} rows={4} maxLength={1000} value={texto}
            onChange={(e) => setTexto(e.target.value)}
            placeholder="Ex.: assistente administrativo; cuido de planilhas, notas fiscais e atendimento a fornecedores" />
          <p className="text-xs text-slate-500">
            Não escreva seu nome, empresa ou dados pessoais. Este texto serve só para sugerir a ocupação e não é guardado.
          </p>
          <button className="rounded-lg bg-slate-900 px-4 py-2 text-white disabled:opacity-40"
            disabled={texto.trim().length < 3 || carregando} onClick={sugerir}>
            {carregando ? "Buscando…" : "Encontrar minha ocupação"}
          </button>
        </section>
      )}

      {etapa === 2 && (
        <section className="space-y-3">
          <p className="font-medium">Qual destas é a sua ocupação (Classificação Brasileira de Ocupações)?</p>
          {sugestoes.map((s) => (
            <button key={s.cbo} onClick={() => { setCbo(s); setEtapa(3); }}
              className="block w-full rounded-lg border border-slate-300 p-3 text-left hover:bg-slate-50">
              <span className="font-medium">{s.titulo}</span>
              <span className="ml-2 text-xs text-slate-500">CBO {s.cbo.slice(0, 4)}-{s.cbo.slice(4)}</span>
            </button>
          ))}
          <button className="text-sm underline" onClick={() => setEtapa(1)}>Nenhuma destas: descrever de outro jeito</button>
        </section>
      )}

      {etapa === 3 && cbo && (
        <section className="space-y-4">
          <p className="text-sm">Ocupação: <strong>{cbo.titulo}</strong></p>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="text-sm">Faixa etária
              <select className={caixa} value={perfil.faixa_etaria}
                onChange={(e) => setPerfil({ ...perfil, faixa_etaria: e.target.value })}>
                <option value="">Escolha</option>{FAIXAS.map((f) => <option key={f} value={f}>{f} anos</option>)}
              </select></label>
            <label className="text-sm">Escolaridade
              <select className={caixa} value={perfil.escolaridade}
                onChange={(e) => setPerfil({ ...perfil, escolaridade: e.target.value })}>
                <option value="">Escolha</option>{ESCOLAR.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
              </select></label>
            <label className="text-sm">Estado (UF)
              <select className={caixa} value={perfil.uf} onChange={(e) => setPerfil({ ...perfil, uf: e.target.value })}>
                <option value="">Escolha</option>{UFS.map((u) => <option key={u} value={u}>{u}</option>)}
              </select></label>
            <label className="text-sm">Tempo nesta função
              <select className={caixa} value={perfil.tempo_funcao}
                onChange={(e) => setPerfil({ ...perfil, tempo_funcao: e.target.value })}>
                <option value="">Escolha</option>{TEMPO.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
              </select></label>
          </div>

          <fieldset className="space-y-1">
            <legend className="text-sm font-medium">
              Antes de ver o resultado: quanto você acha que a IA vai afetar o seu trabalho nos próximos dois anos?
              <span className="font-normal text-slate-500"> (opcional)</span>
            </legend>
            <div className="flex flex-wrap gap-2">
              {PERCEPCAO.map(([v, t]) => (
                <button key={v} type="button" onClick={() => setPercepcao(percepcao === v ? "" : v)}
                  className={`rounded-full border px-3 py-1 text-sm ${percepcao === v ? "bg-slate-900 text-white" : "border-slate-300"}`}>
                  {t}</button>))}
            </div>
          </fieldset>

          <label className="flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-sm">
            <input type="checkbox" className="mt-1" checked={consentimento} onChange={(e) => setConsentimento(e.target.checked)} />
            <span>
              Autorizo o uso <strong>anônimo</strong> da ocupação escolhida, do resultado e das respostas acima (faixa etária,
              escolaridade, UF, tempo na função e percepção) na pesquisa sobre IA e trabalho. O texto que digitei não é
              guardado. <a className="underline" href="/privacidade" target="_blank" rel="noreferrer">Aviso de privacidade</a>.
              Opcional: você vê o resultado mesmo sem marcar.
            </span>
          </label>

          <button className="rounded-lg bg-slate-900 px-4 py-2 text-white disabled:opacity-40"
            disabled={!perfilOk || carregando} onClick={classificar}>
            {carregando ? "Calculando…" : "Ver resultado"}
          </button>
        </section>
      )}

      {etapa === 4 && res && (
        <section className="space-y-4">
          <h2 className="text-xl font-semibold">{res.titulo}</h2>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className={`rounded-xl border p-3 ${COR[res.exposicao.nivel]}`}>
              <p className="text-xs uppercase tracking-wide">Exposição à IA</p>
              <p className="text-2xl font-semibold">{NOME[res.exposicao.nivel]}</p>
              <p className="text-xs">{res.exposicao.gradiente_rotulo} (OIT, 2025)</p>
              <p className="mt-1 text-xs">Mais exposta que {res.exposicao.percentil}% das ocupações</p>
            </div>
            <div className="rounded-xl border border-slate-300 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-500">Natureza provável</p>
              <p className="font-semibold">{res.natureza.rotulo}</p>
            </div>
            <div className="rounded-xl border border-slate-300 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-500">Contratações no Brasil</p>
              {res.evidencia_br && res.evidencia_br.classe !== "dados insuficientes" ? (<>
                <p className="font-semibold capitalize">{res.evidencia_br.classe}</p>
                <p className="text-xs">{res.evidencia_br.grupo_idade} anos, frente ao mercado
                  {res.evidencia_br.janelas ? ` (${res.evidencia_br.janelas.antes} → ${res.evidencia_br.janelas.recente})` : ""};
                  confiança {res.evidencia_br.confianca}</p>
              </>) : <p className="text-sm text-slate-500">Dados insuficientes para o seu perfil</p>}
            </div>
          </div>

          <div className="space-y-3 whitespace-pre-line leading-relaxed">{res.explicacao}</div>
          <p className="rounded-lg bg-slate-50 p-3 text-sm">{res.mensagem_perfil}</p>

          <details className="text-sm">
            <summary className="cursor-pointer font-medium">Tarefas mais e menos expostas (OIT, em inglês)</summary>
            <ul className="mt-2 list-disc pl-5">
              {res.tarefas_mais_expostas.map((t) => <li key={t.tarefa}>{t.tarefa} <em>({t.escore})</em></li>)}
            </ul>
            <ul className="mt-2 list-disc pl-5 text-slate-600">
              {res.tarefas_menos_expostas.map((t) => <li key={t.tarefa}>{t.tarefa} <em>({t.escore})</em></li>)}
            </ul>
          </details>

          <p className="text-xs text-slate-500">{res.aviso} Fontes: {res.fontes.join("; ")}.</p>
          <button className="text-sm underline" onClick={() => { setRes(null); setCbo(null); setPercepcao("");
            setConsentimento(false); setEtapa(1); }}>Fazer outra consulta</button>
        </section>
      )}
    </div>
  );
}
