# Como colocar o painel no seu site do Lovable

1. No Lovable, crie o arquivo `src/components/PainelExposicaoIA.tsx` e cole o conteúdo de `PainelExposicaoIA.tsx`.
2. Em **Project Settings → Environment variables** (ou no `.env`), crie
   `VITE_PAINEL_API_URL=https://SEU-USUARIO-exposicao-ia.hf.space`, que é o endereço do seu Space, sem barra no
   final.
3. Cole no chat do Lovable o prompt abaixo.

---

**Prompt para o Lovable:**

> Crie uma rota `/painel-ia` que renderize o componente `src/components/PainelExposicaoIA.tsx`, sem alterar a
> lógica dele, só adaptando o estilo às cores e fontes do site. Adicione o item "Painel: IA e sua ocupação" ao
> menu principal. Crie também uma rota `/privacidade` com o texto do aviso de privacidade que vou colar a seguir,
> formatado como página de texto simples. Importante:
> - não adicione analytics, pixels, cookies de rastreamento nem formulários de e-mail nessas páginas;
> - não salve nada no Supabase a partir dessas páginas, porque a gravação (com consentimento) é feita pela API
>   externa;
> - não chame a OpenAI no site, porque a API externa faz isso;
> - mantenha a caixa de consentimento desmarcada por padrão.
>
> [cole aqui o texto de aviso_privacidade.md, já preenchido]

---

Teste a integração: abra `/painel-ia`, descreva um cargo, escolha a CBO, preencha o perfil e veja o resultado.
Se aparecer "Não consegui buscar agora", verifique se `ORIGENS_PERMITIDAS`, no Space, contém exatamente o
domínio do seu site (por exemplo, `https://seusite.lovable.app,https://www.seudominio.com.br`).
