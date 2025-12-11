# Image de base Python 3.13
FROM python:3.13-slim

# Métadonnées
LABEL maintainer="RAG API"
LABEL description="API RAG avec FAISS et FastAPI"

# Installation curl pour healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie requirements et installe les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie tout le code
COPY . .

# Crée les répertoires nécessaires
RUN mkdir -p data/raw data/processed logs

# Script de démarrage avec auto-ingestion
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo " Démarrage RAG API..."\n\
\n\
# Vérifie si index existe\n\
if [ ! -f "data/processed/index.faiss" ]; then\n\
  echo " Index non trouvé - Lancement ingestion..."\n\
  python scripts/ingest.py || echo " Ingestion échouée (documents manquants?)"\n\
else\n\
  echo " Index existant trouvé"\n\
fi\n\
\n\
echo " Lancement API sur port 8007..."\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8007\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose le port
EXPOSE 8007

# Commande de démarrage
CMD ["/app/start.sh"]