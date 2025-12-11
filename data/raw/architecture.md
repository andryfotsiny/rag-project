# Architecture d'un Système RAG

## Introduction

L'architecture d'un système RAG repose sur plusieurs composants clés qui travaillent ensemble pour fournir des réponses précises et contextuelles. Cette architecture modulaire permet une maintenance facile et des améliorations progressives.

## Composants Principaux

### 1. Document Loader
Le Document Loader est responsable du chargement des documents depuis différentes sources. Il supporte de multiples formats comme TXT, PDF, Markdown, DOCX, et peut être étendu pour gérer d'autres types. Ce composant gère également l'encodage des caractères et la validation des fichiers.

### 2. Chunker
Le Chunker découpe les documents en morceaux cohérents appelés chunks. Il utilise des stratégies intelligentes comme :
- Découpage par nombre de tokens (300-500 tokens recommandés)
- Overlap entre chunks (10-20%) pour maintenir le contexte
- Respect des limites de paragraphes quand possible
- Pré-processing du texte (normalisation, nettoyage)

La taille optimale d'un chunk dépend de l'usage : des chunks plus petits donnent plus de précision, tandis que des chunks plus grands offrent plus de contexte.

### 3. Embedding Service
L'Embedding Service transforme le texte en vecteurs numériques (embeddings) pour la recherche sémantique. Il utilise des modèles pré-entraînés comme :
- all-MiniLM-L6-v2 : rapide et léger (384 dimensions)
- all-mpnet-base-v2 : meilleure qualité (768 dimensions)
- multilingual-MiniLM : support multilingue

Les embeddings capturent le sens sémantique du texte et permettent de calculer des similarités entre documents.

### 4. Vector Store
Le Vector Store stocke les embeddings et permet des recherches rapides par similarité. Options populaires :
- FAISS : bibliothèque locale ultra-rapide, idéale pour prototypes
- Qdrant : serveur complet avec API REST, filtres avancés
- Pinecone : service cloud managé, très scalable
- Weaviate : open-source avec GraphQL

Le choix dépend des besoins : FAISS pour le développement, Qdrant/Weaviate pour la production.

### 5. RAG Pipeline
Le RAG Pipeline orchestre le processus complet :
1. Réception de la query utilisateur
2. Génération de l'embedding de la query
3. Recherche des chunks les plus similaires (top-k)
4. Filtrage par score de similarité minimum
5. Agrégation du contexte
6. Génération de la réponse (optionnel avec LLM)

## Bonnes Pratiques

### Chunking
- Utiliser des chunks de 200-400 tokens pour un bon équilibre
- Appliquer un overlap de 10-20% entre chunks consécutifs
- Normaliser le texte avant découpage
- Conserver des métadonnées riches (source, position, type)

### Embeddings
- Normaliser les embeddings pour utiliser la similarité cosine
- Utiliser le même modèle pour indexation et recherche
- Considérer des modèles spécialisés pour des domaines techniques
- Implémenter un cache pour les embeddings fréquents

### Recherche
- Filtrer les résultats par score minimum (0.5-0.7 typique)
- Récupérer top-k * 2 puis filtrer pour avoir de la marge
- Implémenter un reranking avec cross-encoder pour améliorer la précision
- Logger toutes les recherches pour analyse

### Production
- Monitorer les latences et la qualité des résultats
- Implémenter des métriques d'évaluation (Recall@K, MRR, NDCG)
- Mettre en place un feedback loop utilisateur
- Versionner les index et les configurations
- Tester avec des queries représentatives

## Optimisations Avancées

### Reranking
Utiliser un modèle cross-encoder après la recherche initiale pour réordonner les résultats par pertinence réelle.

### Hypothetical Document Embeddings (HyDE)
Générer un document hypothétique à partir de la query, l'embedder, puis l'utiliser pour la recherche.

### Query Expansion
Enrichir la query avec des termes synonymes ou des reformulations pour améliorer le recall.

### Filtrage par Métadonnées
Permettre le filtrage des résultats par type de document, date, auteur, etc.
