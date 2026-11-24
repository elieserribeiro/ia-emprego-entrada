"""Monta a pasta pronta para subir no Hugging Face Space (ou Render).

    python deploy/montar_space.py ../meu-space      # depois: cd ../meu-space && git add . && git commit && git push
O modelo ajustado (artefatos/indice_st/modelo) tem centenas de MB: o Space usa Git LFS (o .gitattributes
gerado aqui já marca os arquivos grandes).
"""
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
dest = Path(sys.argv[1] if len(sys.argv) > 1 else RAIZ.parent / "space_painel_ia")
dest.mkdir(parents=True, exist_ok=True)
for f in ("Dockerfile", "requirements.txt"):
    shutil.copy(RAIZ / "deploy" / f, dest / f)
shutil.copy(RAIZ / "deploy" / "README_space.md", dest / "README.md")
shutil.copytree(RAIZ / "painelia", dest / "painelia", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
art = dest / "artefatos"
art.mkdir(exist_ok=True)
for f in ("ocupacoes.json", "titulos_cbo.json"):
    shutil.copy(RAIZ / "artefatos" / f, art / f)
for idx in ("indice_st", "indice_tfidf"):
    if (RAIZ / "artefatos" / idx).exists():
        shutil.copytree(RAIZ / "artefatos" / idx, art / idx, dirs_exist_ok=True)
(dest / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                                     "*.bin filter=lfs diff=lfs merge=lfs -text\n"
                                     "*.npy filter=lfs diff=lfs merge=lfs -text\n"
                                     "*.pkl filter=lfs diff=lfs merge=lfs -text\n"
                                     "tokenizer.json filter=lfs diff=lfs merge=lfs -text\n")
print(f"pasta do Space pronta em {dest}")
