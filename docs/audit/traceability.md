# Traceability matrix — EventFlow objectives → code → tests → proof

**Source audit:** `~/workspace/your_files/eventflow-eda-chicago/Audit-EventFlow.md`
(2026-11-15, commit `9903a6c`, verdict 7.5/10 — 28 objectives: 15 attained,
6 partial, 6 not attained, 1 unverifiable).
**Phase-0 baseline (re-measured 2026-11-15 in this tree):** 535 tests
collected, 534 passed, 1 failed (`test_to_pandas`, pandas missing),
coverage 81 % (5,145 statements, 1,000 missed); ruff, black, mypy clean.
**After phase 0 (measured by `make audit`, 2026-11-15):** 564 tests collected, 564 passed, 0 failed, 0 skipped; coverage 82 % (5,176 statements, 928 missed).

**How to read this file:** one row per objective. "Statut audit" is the
2026-11-15 verdict; "Statut phase 0" is the status after the phase-0
changes, updated only where phase 0 produced a verifiable change. Statuses
are never inflated: a fix to the *honesty* of an objective does not count
as implementing the objective itself.

| # | Objectif | Statut audit | Statut phase 0 | Preuve phase 0 |
|---|---|---|---|---|
| 1 | Bibliothèque générique de transformation d'événements spatio-temporels | Atteint | Atteint | Inchangé. Cœur sans logique Chicago ; la portée générique est actée dans `MEMORY.md` (tout problème spatio-temporel). |
| 2 | Évaluation paresseuse (lazy) avec Polars | Atteint | Atteint | Inchangé. |
| 3 | 5 adapters, formes (N,F), (B,T,F), (T,C,H,W), nœuds+arêtes, timestamps continus | Partiel | Partiel | **Quick Start réparé** : les 5 exemples du README sont corrigés et exécutés par `tests/test_readme_examples.py` (formes vérifiées : table (12,3), séquence (4,30,1), raster (3,2,10,10), graphe nœuds (4,3) / arêtes (2,4), stream 12 timestamps). Reste partiel : `to_tensorflow()` limité au Raster (transpose NumPy) et `to_torch()` absent de Table — inchangés, hors phase 0. |
| 4 | Steps temporels, spatiaux, point process, streaming, validation | Atteint | Atteint | Inchangé. |
| 5 | Pipelines composables, registre de steps | Atteint | Atteint | Inchangé. |
| 6 | « 290 tests, 68 % de couverture » | Atteint (chiffres périmés, réalité meilleure) | Atteint | **Chiffres mis à jour** dans le README (badge + « Validated » + arbre) avec les valeurs mesurées par `make audit` ; `pandas` déclaré en dev et `test_to_pandas` conditionnel — suite verte après `pip install -e ".[dev]"`. Voir `audit/phase-0/manifest.json`. |
| 7 | Version 0.1.0, statut Alpha | Atteint | Atteint | Inchangé. |
| 8 | Exécution asynchrone | Atteint | Atteint | Inchangé. |
| 9 | Panneaux d'observation | Atteint | Atteint | Inchangé. |
| 10 | Qualité d'observation et contrôles de cadence | Atteint | Atteint | Inchangé. |
| 11 | Domaines spatiaux génériques | Atteint | Atteint | Inchangé. |
| 12 | Audit données réelles, sémantique, identité de grille, K/L/H | Atteint | Atteint | Inchangé. |
| 13 | Recette Chicago : 300 m, bins 6 h, EPSG:4326 → EPSG:26971 | Atteint | Atteint | **Rejouée par `make audit`** sur les données T1 2024 : 56 491 lignes, somme `count` = 60 235 (voir `audit/phase-0/replay_chicago.json`). |
| 14 | CLI : version, list-datasets, list-recipes, validate, run | Atteint | Atteint | Inchangé, et `validate` vérifie désormais aussi que le pipeline se construit (`src/eventflow/cli/main.py`). |
| 15 | Enrichissement contextuel (météo, événements, démographie) | Partiel | Partiel | **Marqué expérimental** : les 3 sources Chicago portent une mention EXPERIMENTAL explicite ; toujours non câblées, non testées, contexte désactivé dans la recette. Aucune prétention ajoutée. |
| 16 | Suivi d'expériences MLflow | Partiel | Partiel | **Marqué expérimental** (`mlflow_tracker.py`) ; ajout de `tests/tracking/test_tracking.py` (import sans MLflow OK, `ImportError` claire à l'instanciation sans MLflow). Toujours aucun test contre un vrai backend MLflow. |
| 17 | Utilitaires `core/features.py` | Non atteint | Partiel | **Tranché** : `aggregate_counts`, `moving_window_aggregation`, `lag_features` supprimés (remplacés par des Steps testés) ; `compute_ratios` corrigé (dénominateur nul → null) et testé (`tests/core/test_features_utils.py`) ; `aggregate_by_category`, `encode_categorical`, `normalize_features` marqués EXPERIMENTAL avec leurs limites documentées ; docs mises à jour (`docs/api_reference.md`, `docs/architecture.md`). |
| 18 | Documentation API et docstrings | Partiel | Partiel | **README réparé** (Quick Start exécuté, chiffres, section « ce qu'EventFlow ne fait pas », notebook qualifié de non vérifié) ; **URLs corrigées** dans `pyproject.toml` (`BayoHabib/eventflow`, URL ReadTheDocs supprimée car inexistante). Reste partiel : pas de Sphinx/MkDocs, pas de ReadTheDocs. |
| 19 | Logging structuré, exceptions personnalisées | Partiel | Partiel | Inchangé (hors phase 0). Note : les erreurs de configuration sont désormais des `ValueError`/`ValidationError` explicites avec la liste des clés valides. |
| 20 | CI : tests + lint + types sur 3 OS × 3 Python, build | Atteint | Atteint | **Échec `test_to_pandas` corrigé** (pandas déclaré + test conditionnel). Réserve inchangée : l'état réel de la CI distante n'a pas été consulté. |
| 21 | TODO haute priorité : alias adapters | Non atteint | Non atteint | Hors phase 0 — non commencé, comme décidé. |
| 22 | TODO haute priorité : service FastAPI | Non atteint | Non atteint | Hors phase 0 — non commencé. |
| 23 | TODO haute priorité : publication PyPI | Non atteint | Non atteint | Hors phase 0 — non commencé. |
| 24 | TODO haute priorité : Docker | Non atteint | Non atteint | Hors phase 0 — non commencé. |
| 25 | TODO : couverture 68 % → 80 %+ | Atteint | Atteint | **Item coché dans `TODO.md`** (81 % mesurés, maintenus par `make audit` ; seuil de sortie phase 0 : ≥ 81 %). |
| 26 | Tests d'intégration, propriétés, performance | Partiel | Partiel | Inchangé sur le fond ; les nouveaux tests de phase 0 sont des tests d'intégration de configuration et d'exemples exécutés. Hypothesis toujours inutilisé, aucun test de performance. |
| 27 | Recette Chicago : features catégorielles, fenêtres 1d/7d/30d, partitionnement, zéros | Non atteint | Non atteint | **La malhonnêteté est corrigée, pas l'objectif** : les clés ont été retirées du YAML et déplacées dans `docs/design-notes/chicago-declared-features.md` ; `FeatureConfig` est typé et strict ; toute réintroduction lève une erreur bloquante (tests `tests/core/test_recipe_features_config.py`) ; `validate` construit désormais le pipeline. Les features elles-mêmes restent à implémenter en phase 1. |
| 28 | Notebook de démo Chicago | Non vérifiable | Non vérifiable | Inchangé : notebook non exécuté, chiffres non confirmés. Le README et le TODO le disent désormais explicitement. |

## Compteur après phase 0

Sur la base du tableau ci-dessus : **15 atteints, 7 partiels, 5 non atteints,
1 non vérifiable** (contre 15 / 6 / 6 / 1 à l'audit). Le seul mouvement de
statut est l'objectif 17 (non atteint → partiel). Les objectifs 21–24
restent volontairement non atteints : ils sont hors phase 0.
