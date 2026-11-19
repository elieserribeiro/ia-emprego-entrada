# Imagem da API para Hugging Face Spaces (SDK "Docker") ou Render.
# Monte a pasta do Space assim (o script deploy/montar_space.py faz isso):
#   Dockerfile, requirements.txt, painelia/, artefatos/ (ocupacoes.json, titulos_cbo.json, indice_st/ ou indice_tfidf/)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf
WORKDIR /app
COPY requirements.txt .
# torch só para CPU (imagem bem menor)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY painelia ./painelia
COPY artefatos ./artefatos
RUN useradd -m usuario && chown -R usuario /app
USER usuario
EXPOSE 7860
# --no-access-log: o servidor não registra IPs nem caminhos das requisições
CMD ["uvicorn", "painelia.api:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log", "--proxy-headers"]
