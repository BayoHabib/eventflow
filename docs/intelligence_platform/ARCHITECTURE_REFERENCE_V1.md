# Intelligence Platform — Architecture de référence v0.1

**Date:** 2026-10-13  
**Statut:** proposition documentée, non déployée.  
**Périmètre:** exploration Web, sources multimodales, observation, mémoire, raisonnement, vérification, prédiction expérimentale et exploration humaine.

## 1. Mission et limites

Créer une infrastructure modulaire capable de (a) découvrir des sources autorisées, (b) enregistrer des observations accompagnées de leur provenance, (c) extraire des assertions, (d) organiser et retrouver les connaissances, (e) formuler et réfuter des hypothèses, (f) entraîner et évaluer des modèles prédictifs, et (g) restituer les résultats sans effacer l'incertitude.

**Non-objectifs initiaux :** profilage individuel, reconnaissance faciale opérationnelle, décisions coercitives automatisées, surveillance non autorisée, prédiction présentée comme une certitude, collecte indiscriminée de données privées.

## 2. Invariants épistémiques

1. Les documents et mesures sont des **preuves candidates**, non des vérités garanties.
2. Observation, inférence, hypothèse, prédiction et vérification restent des **types distincts**.
3. Une assertion dérivée référence ses entrées, son modèle/version et sa date.
4. Les contradictions ne s'écrasent pas : elles coexistent jusqu'à arbitrage explicite.
5. Les événements disposent au minimum de `occurred_at`, `collected_at`, `source_id` et d'un `assertion_id` stable.
6. Les prévisions mentionnent horizon, population/zone, calibrage, date d'entraînement et jeu d'évaluation.
7. Une absence de source ou une indisponibilité technique n'est jamais convertie en « aucun événement ».

## 3. Architecture par ports et adaptateurs

```text
     Internet public      Capteurs autorisés       Données ouvertes
           |                     |                       |
   Agent Reach / Crawl4AI     Surveillance Go        API / Downloader
           |                     |                       |
           +---------------------+-----------------------+
                                 |
                      Source adapter boundary
                     (permissions, quotas, hashes)
                                 |
                       Document/media pipeline
                        Docling / extracteurs
                                 |
                         Assertions / Events
                 EventFlow + versioned contract
                     /            |           \
          Immutable source store   |     Analytical event store
                                 |
                    Temporal graph / search indexes
                   Graphiti (candidate) / hybrid search
                                 |
               Hypothesis & model evaluation services
                 evidence -> conjecture -> test -> revise
                                 |
                      Authenticated query gateway
                                 |
                           OSIRIS frontend
```

**Le diagramme indique des responsabilités, pas un engagement envers une seule base de données.** Le stockage source, l'index temporel, le graphe et les artefacts de modèles peuvent être distincts.

## 4. Frontières et responsabilités

| Port | Entrées | Sorties | Propriétaire |
|---|---|---|---|
| `SourceConnector` | Source autorisée, requête, politique | Référence de contenu, horodatage, provenance | Agent Reach/Crawl4AI/adaptateurs |
| `DocumentExtractor` | Référence et média | Segments et offsets vérifiables | Docling/autres |
| `AssertionExtractor` | Segments, modèle/version | Assertions et citations exactes | Service expérimental |
| `EventNormalizer` | Assertions/événements | Enveloppe versionnée | EventFlow |
| `KnowledgeStore` | Assertions et liens | Historique, versions, contradictions | Graphiti à évaluer |
| `HypothesisEngine` | Graphes et sources | Hypothèses testables, recherche de contre-exemples | R&D spécifique |
| `ForecastEngine` | Événements agrégés et splits | Distributions prévisionnelles évaluées | Chicago Crime/ML |
| `QueryGateway` | Requête authentifiée | Vue filtrée, citations, traces d'inférence | Service propriétaire |
| `Presentation` | Vue filtrée | Cartes, graphes, chronologies | OSIRIS |

## 5. Protocole de connaissance

Enveloppe minimale : `schema_version`, `assertion_id`, `assertion_kind`, `event_type`, `occurred_at`, `provenance.source_id`, `provenance.collected_at`, `upstream_assertion_ids`, `classification`. `location`, `model_id`, `confidence`, `target_assertion_id` sont conditionnels.

La branche de contrat expérimental est [EventFlow PR #14](https://github.com/BayoHabib/eventflow/pull/14). **Ne pas considérer ce schéma v0.1 comme une ontologie achevée.** Conserver les extraits de source avec offsets et dates de récupération; la seule chaîne de texte générée par un LLM ne suffit pas.

Contrat transport : version explicite, idempotence (`issuer + assertion_id`), limite de taille, authenticité de l'émetteur, confidentialité, permission par type de donnée, journaux d'accès, rétention et possibilité de supprimer les données privées conformément aux obligations applicables.

## 6. Première tranche verticale, reproductible et non sensible

**Sujet pilote :** reconstruction de l'évolution d'une proposition scientifique publique à partir de sources sous licence ou accessibles légalement.

1. Utiliser Agent Reach comme catalogue de routes de recherche, avec un connecteur public Web minimal comme témoin.
2. Collecter 10–20 documents publics avec URL et horodatage; exporter des empreintes, sans cookies ou secrets.
3. Extraire un jeu d'assertions **avec références aux passages**.
4. Transformer en `UnifiedEvent`; séparer assertion originale et hypothèse du modèle.
5. Insérer dans une mémoire temporelle; enregistrer un contre-exemple sourcé sans effacer l'assertion initiale.
6. Exposer une API de lecture authentifiée; afficher la chronologie et le statut dans OSIRIS **après durcissement de ses API**.

### Critères de réussite

- 100 % des assertions de démo affichées ont une provenance vers une source et un passage consultable.
- 0 hypothèse présentée comme observation confirmée.
- Une contradiction a une histoire reconstructible après redémarrage.
- Ré-exécuter le jeu fixe de documents donne les mêmes identifiants et une sortie déterministe de normalisation.
- Aucune donnée restreinte transmise aux flux SSE/SDK publics d'OSIRIS.
- Un test adversarial « injection d'instructions dans une page collectée » ne peut pas déclencher de commandes ni modifier les règles du système.

## 7. Décisions et portes de validation

- **ADR-001:** architecture modulaire par adaptateurs, EventFlow comme contrat/transformation et non comme entrepôt universel.
- **ADR-002:** Agent Reach propose la route d'accès; un orchestrateur de collecte explicite applique droits, limites et contrôle des identifiants.
- **ADR-003:** OSIRIS est un client optionnel du portail d'information, pas le dépositaire exclusif des connaissances.
- **ADR-004:** Graphiti est un candidat soumis à benchmark; aucun verrouillage avant tests de provenance, révision, coût et reconstruction.
- **ADR-005:** adopter « détection puis vérification » pour toutes les affirmations générées par LLM.

**Gate G0:** licences et sécurité des dépendances vérifiées.  
**Gate G1:** contrat d'assertions et stockage/relecture.  
**Gate G2:** recherche reproductible et provenance.  
**Gate G3:** graphe temporel et contradiction.  
**Gate G4:** OSIRIS lecture seule sécurisée.  
**Gate G5:** raisonnement agent et comparaison à des baselines sans agent.

## 8. Traçabilité

Catalogue et réserves : [GITHUB_COMPONENT_CATALOG_V1.md](./GITHUB_COMPONENT_CATALOG_V1.md).  
Expérience et tests : [VERTICAL_SLICE_ACCEPTANCE_V1.md](./VERTICAL_SLICE_ACCEPTANCE_V1.md).

Aucune intégration/installation/benchmark n'est revendiquée dans cette version. Une sélection « candidate » ne signifie pas « adoptée ».
