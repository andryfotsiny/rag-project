# RAG API - Test Technique

API RAG (Retrieval-Augmented Generation) complète avec FAISS et sentence-transformers.

**Démo en ligne** : http://62.72.42.173:8007/docs


---

## Installation

### Méthode 1 : Docker (Recommandé)

```bash
# Clone le projet
git clone https://github.com/andryfotsiny/rag-project.git
cd rag-project

# Lance avec Docker
docker-compose up --build

# API disponible sur http://localhost:8007/docs
```

L'ingestion se lance automatiquement au démarrage si aucun index n'existe.

---

### Méthode 2 : Installation locale

#### Prérequis
- Python 3.13+
- pip

#### Étapes

1. Clone et setup

```bash
git clone <repo>
cd rag-project
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

2. Configuration

Le fichier `.env` est déjà configuré. Modifie-le si nécessaire.

3. Ajoute des documents

Place tes documents dans `data/raw/` :

```bash
cp mes_documents/*.{txt,md,pdf} data/raw/
```

Des exemples sont déjà fournis (intro.txt, architecture.md, faq.txt).

4. Lance l'ingestion

```bash
python scripts/ingest.py
```

Output attendu :
```
Début de l'ingestion
X documents chargés
X chunks créés
X embeddings générés
X chunks indexés
Index sauvegardé
Ingestion terminée avec succès
```

5. Lance l'API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API disponible sur :
- Swagger UI : http://localhost:8000/docs
- Health : http://localhost:8000/api/v1/health

---



---

## Fonctionnalités

- Ingestion de documents (TXT, MD, PDF)
- Chunking intelligent avec overlap
- Embeddings avec sentence-transformers
- Vector store FAISS (IndexFlatIP)
- API REST avec FastAPI
- Documentation Swagger automatique
- Logs structurés avec Loguru
- Typage complet avec Pydantic V2
- Évaluation de performance (Recall@K, Precision@K)

---

## Architecture

```
rag-project/
├── app/
│   ├── api/endpoints/     # Routes API
│   ├── core/              # Configuration & logs
│   ├── services/          # Services métier
│   ├── models/            # Schémas Pydantic
│   └── main.py            # Application FastAPI
├── scripts/
│   └── ingest.py          # Script d'ingestion
├── data/
│   ├── raw/               # Documents sources
│   └── processed/         # Index FAISS
└── tests/                 # Tests unitaires
```


## Endpoints API

### Health Check

```bash
GET /api/v1/health
```

Response :
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "vector_store_loaded": true,
  "total_chunks": 150
}
```

---

### Embed Text

```bash
POST /api/v1/embed
Content-Type: application/json

{
  "text": "Comment fonctionne le RAG ?"
}
```

Response :
```json
{
  "text": "Comment fonctionne le RAG ?",
  "embedding": [0.123, -0.456, ...],
  "dimension": 384,
  "model": "all-MiniLM-L6-v2"
}
```

---

### Search

```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "Comment fonctionne le système ?",
  "k": 5,
  "min_score": 0.4
}
```

Response :
```json
{
  "query": "Comment fonctionne le système ?",
  "results": [
    {
      "chunk": {
        "id": "doc1_chunk_5",
        "text": "Le système fonctionne...",
        "metadata": {
          "source": "guide.pdf",
          "chunk_index": 5
        }
      },
      "score": 0.89,
      "relevance": "high"
    }
  ],
  "total_found": 8,
  "filtered_count": 5,
  "min_score_used": 0.4
}
```

---

### RAG Pipeline

```bash
POST /api/v1/rag
Content-Type: application/json

{
  "query": "Quels sont les avantages du RAG ?",
  "k": 5,
  "min_score": 0.4
}
```

Response :
```json
{
  "query": "Quels sont les avantages du RAG ?",
  "context": "Le RAG combine...\n\n---\n\nLes avantages incluent...",
  "sources": ["rag_guide.pdf", "architecture.md"],
  "scores": [0.89, 0.85, 0.82],
  "chunk_count": 3,
  "metadata": {
    "total_chars": 1500,
    "avg_score": 0.85
  }
}
```

---

### Evaluate (Bonus)

```bash
POST /api/v1/evaluate
Content-Type: application/json

{
  "queries": [
    {
      "query": "Quels sont les avantages du RAG ?",
      "expected_sources": ["intro.txt", "faq.txt"]
    }
  ],
  "k": 5
}
```

Response :
```json
{
  "recall_at_k": 0.85,
  "precision_at_k": 0.60,
  "avg_similarity": 0.72,
  "total_queries": 1,
  "details": [...]
}
```

---

## Exemple de Requête RAG

```bash
curl -X POST http://localhost:8000/api/v1/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Comment améliorer les performances du RAG ?",
    "k": 5,
    "min_score": 0.4
  }'
```

---

## Configuration

Variables dans `.env` :

```bash
# Embeddings
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION=384
EMBEDDING_DEVICE="cpu"  # ou "cuda" si GPU

# Chunking
CHUNK_SIZE=400
CHUNK_OVERLAP=80

# RAG
DEFAULT_TOP_K=5
MIN_SIMILARITY_SCORE=0.40
MAX_CONTEXT_LENGTH=3000

# API
API_PORT=8000
```

---

## Technologies Utilisées

- **FastAPI** : Framework API moderne et performant
- **Pydantic V2** : Validation et typage stricte
- **FAISS** : Vector store haute performance
- **sentence-transformers** : Génération d'embeddings
- **Loguru** : Logging structuré avec rotation
- **pypdf** : Lecture de PDFs
- **Docker** : Containerisation

---

## Améliorations pour un Produit Réel

### Court Terme (0-3 mois)
1. **Reranking** : Implémenter un cross-encoder après la recherche initiale pour améliorer la pertinence
2. **Filtrage métadonnées** : Permettre de filtrer par type de document, date, auteur
3. **Cache intelligent** : Mettre en cache les embeddings des queries fréquentes
4. **Tests automatisés** : Suite de tests avec métriques (Recall@K, MRR)

### Moyen Terme (3-6 mois)
1. **Migration Qdrant** : Pour scalabilité et fonctionnalités avancées (API REST, réplication)
2. **Query expansion** : Enrichir les requêtes avec synonymes et reformulations
3. **Chunking sémantique** : Découper par thèmes via embeddings plutôt que par taille fixe
4. **Monitoring** : Prometheus/Grafana pour latences, scores, satisfaction

### Long Terme (6-12 mois)
1. **Hypothetical Document Embeddings (HyDE)** : Générer un doc hypothétique, l'embedder, puis chercher
2. **Multi-modal** : Support images, tableaux, graphiques
3. **Feedback loop** : Apprendre des retours utilisateurs pour améliorer le ranking
4. **A/B Testing** : Tester différentes stratégies de chunking/scoring en production

### Métriques de Succès
- **Latence P95** < 500ms
- **Recall@5** > 85%
- **Satisfaction utilisateur** > 4/5
- **Coût** < 0.01€ par query

---

## Points Forts de l'Implémentation

1. **Architecture modulaire** : Services séparés et testables
2. **Type-safety** : Pydantic + type hints partout
3. **Gestion d'erreurs** : Exceptions custom + logs détaillés
4. **Performance** : Async/await + batch processing
5. **Production-ready** : Configuration, logs, healthcheck
6. **Documentation** : Swagger auto-généré + README détaillé
7. **Portabilité** : FAISS local, pas de dépendances externes

---

## Métriques de Qualité

### Chunking
- Overlap intelligent (20%)
- Respect des paragraphes quand possible
- Taille configurable (400 tokens par défaut)

### Embeddings
- Modèle léger et performant (all-MiniLM-L6-v2)
- Normalisation pour cosine similarity
- Batch processing pour efficacité

### Scoring
- Seuils configurables (high/medium/low)
- Filtrage par score minimum (0.40 par défaut)
- Métriques d'évaluation (Recall@K, Precision@K)

### Context Aggregation
- Agrégation intelligente avec limite tokens
- Séparateurs entre chunks
- Traçabilité des sources

---

## Limitations Connues

1. **Score de similarité**
   - Avec peu de documents (10 chunks), scores typiques 0.3-0.5
   - Normal : peu de contexte = moins de similarité sémantique
   - Solution : Plus de documents ou ajuster MIN_SIMILARITY_SCORE à 0.30-0.40

2. **Pas de reranking**
   - Ordre basé uniquement sur cosine similarity
   - Amélioration possible : cross-encoder en 2e étape
   - Impact : Précision pourrait augmenter de 10-15%

3. **Chunking approximatif**
   - Basé sur caractères avec approximation tokens (1 token ≈ 4 chars)
   - Différence < 5% vs tokenization exacte
   - Acceptable pour ce use case, évite fuite mémoire

4. **FAISS limitations**
   - Pas de filtres avancés sur métadonnées
   - Pas d'API REST native
   - Scalabilité limitée à ~1M vecteurs
   - Pour production : migration Qdrant recommandée

---

## Développeur

**Rôle dans le test** : Support technique

**Compétences démontrées** :
- Python avancé (3.13, async/await, type hints)
- Architecture logicielle (SOLID, modularité)
- Machine Learning (embeddings, similarity search)
- API Design (REST, OpenAPI, validation)
- DevOps (Docker, logs, monitoring)
- Documentation technique

---

## License

MIT