# Traceability matrix — Shikong objectives → code → tests → proof

**Source audit:** `~/workspace/your_files/eventflow-eda-chicago/Audit-Shikong.md`
(2026-11-15, commit `9903a6c`, verdict 7.5/10 — 28 objectives: 15 attained,
6 partial, 6 not attained, 1 unverifiable).
**Phase-0 baseline (re-measured 2026-11-15 in this tree):** 535 tests
collected, 534 passed, 1 failed (`test_to_pandas`, pandas missing),
coverage 81 % (5,145 statements, 1,000 missed); ruff, black, mypy clean.
**After phase 0 (measured by `make audit`, 2026-11-15):** 564 tests collected, 564 passed, 0 failed, 0 skipped; coverage 82 % (5,176 statements, 928 missed).

**How to read this file:** one row per objective. "Statut audit" is the
2026-11-15 verdict; "Statut phase 0" is the status after the phase-0
changes, updated only where phase 0 produced a verifiable change. The
"Après phase 1" and "Après phase 2" sections at the bottom record the
phase-1 and phase-2 movements the same way. Statuses are never inflated: a fix to the *honesty* of an
objective does not count as implementing the objective itself.

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
| 14 | CLI : version, list-datasets, list-recipes, validate, run | Atteint | Atteint | Inchangé, et `validate` vérifie désormais aussi que le pipeline se construit (`src/shikong/cli/main.py`). |
| 15 | Enrichissement contextuel (météo, événements, démographie) | Partiel | Partiel | **Marqué expérimental** : les 3 sources Chicago portent une mention EXPERIMENTAL explicite ; toujours non câblées, non testées, contexte désactivé dans la recette. Aucune prétention ajoutée. |
| 16 | Suivi d'expériences MLflow | Partiel | Partiel | **Marqué expérimental** (`mlflow_tracker.py`) ; ajout de `tests/tracking/test_tracking.py` (import sans MLflow OK, `ImportError` claire à l'instanciation sans MLflow). Toujours aucun test contre un vrai backend MLflow. |
| 17 | Utilitaires `core/features.py` | Non atteint | Partiel | **Tranché** : `aggregate_counts`, `moving_window_aggregation`, `lag_features` supprimés (remplacés par des Steps testés) ; `compute_ratios` corrigé (dénominateur nul → null) et testé (`tests/core/test_features_utils.py`) ; `aggregate_by_category`, `encode_categorical`, `normalize_features` marqués EXPERIMENTAL avec leurs limites documentées ; docs mises à jour (`docs/api_reference.md`, `docs/architecture.md`). |
| 18 | Documentation API et docstrings | Partiel | Partiel | **README réparé** (Quick Start exécuté, chiffres, section « ce qu'Shikong ne fait pas », notebook qualifié de non vérifié) ; **URLs corrigées** dans `pyproject.toml` (`BayoHabib/shikong`, URL ReadTheDocs supprimée car inexistante). Reste partiel : pas de Sphinx/MkDocs, pas de ReadTheDocs. |
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

## Après phase 1

**Mesuré par `make audit`, 2026-11-15** (voir `audit/phase-1/manifest.json`) :
verdict PASS (5/5 : ruff, black, mypy, pytest, replay), 705 tests passés,
couverture 84 % (5 718 statements, 912 missed) ; replay Chicago T1 2024 :
1 908 088 lignes (5 242 cellules × 364 bins), somme `count` = 60 235,
1 851 597 lignes à zéro explicite, toutes `observation_status ==
"observed"`, aucun `count` null. Lag, moyenne glissante, lag spatial,
EWMA, tendance et entropie vérifiés à la main contre les données brutes.

- **Objectif 27 — Non atteint → Partiel.** Les features catégorielles et
  les fenêtres 1d/7d/30d sont implémentées génériquement
  (`CategoryCountsStep`, `TemporalLagStep`, `MovingAverageStep`), avec la
  complétion du panel en zéros explicites (`CompletePanelStep`) et les
  groupes dépendants (EWMA, tendance, lag spatial queen du lag le plus
  court, temps depuis le dernier événement, burstiness, features zéro,
  encodages cycliques) — voir `docs/design-notes/phase1-causality.md` et
  `docs/design-notes/chicago-declared-features.md`. Il reste partiel car
  le partitionnement de sortie (`output.partition_by`) n'est toujours pas
  implémenté (toujours une erreur bloquante dans la CLI).
- **Objectif 13 — Atteint (chiffres mis à jour).** Recette rejouée par
  `make audit` : 1 908 088 lignes, somme `count` = 60 235 (voir
  `audit/phase-1/replay_chicago.json`). Le panel creux de 56 491 lignes
  est remplacé par le panel complété ; la somme est inchangée.
- **Objectif 17 — Partiel (précisé).** `aggregate_by_category`
  (EXPERIMENTAL, sans garde-fou) est supplanté par le
  `CategoryCountsStep` testé (top-N + bucket `"other"`, noms assainis,
  parts nulles quand le total est 0, entropie de Shannon) —
  `tests/core/test_categorical_steps.py` (27 tests, références calculées
  à la main + non-fuite). `compute_ratios` reste corrigé et testé.
- **Objectif 25 — Atteint (maintenu).** Couverture ≥ 82 % maintenue après
  phase 1 (voir `audit/phase-1/manifest.json`).
- **Objectif 26 — Partiel (précisé).** Chaque groupe de features a des
  tests de référence calculés à la main et des tests de non-fuite
  (causal cutoff) ; le replay Chicago est un test d'intégration sur
  données réelles (58,7 s, lag spatial ~19 s sur 364 snapshots).
  Hypothesis toujours inutilisé, aucun test de performance dédié.

**Compteur après phase 1 : 15 atteints, 8 partiels, 4 non atteints,
1 non vérifiable.** Le seul mouvement de statut est l'objectif 27
(non atteint → partiel). Les objectifs 21–24 restent volontairement non
atteints : ils sont hors phase 1.

## Après phase 2

**Mesuré par `make audit`, 2026-11-15** (voir `audit/phase-2/manifest.json`) :
verdict PASS (5/5 : ruff, black, mypy, pytest, replay), 838 tests passés,
couverture 85 % (6 261 statements, 944 missed) ; replay Chicago T1 2024 :
1 908 088 lignes × 90 colonnes (5 242 cellules × 364 bins), somme `count`
= 60 235, 1 851 597 lignes à zéro explicite, aucun `count` null. Les 4
nouvelles familles (lag spatial multi-échelle anneaux 1/2/3, KDE
gaussienne, STKDE, near-repeat, Hawkes simplifié) sont vérifiées à la main
contre les données brutes (14 contrôles, cellule la plus chargée
1930/1195, bin 2024-02-17 12:00 — voir
`docs/design-notes/phase2-spacetime-v2.md`).

- **Objectif 13 — Atteint (chiffres mis à jour).** Recette rejouée par
  `make audit` : 1 908 088 lignes × 90 colonnes, somme `count` = 60 235
  (voir `audit/phase-2/replay_chicago.json`). Le lag spatial queen
  mono-échelle est remplacé par le lag multi-échelle (anneaux 1/2/3) ;
  la valeur de référence `count_lag_1_spatial_lag` = 0.375 est conservée
  pour l'anneau 1.
- **Objectif 25 — Atteint (maintenu).** Couverture 85 % après phase 2
  (voir `audit/phase-2/manifest.json`).
- **Objectif 26 — Partiel (précisé).** Chaque nouvelle famille a des
  tests de référence calculés à la main / par brute-force indépendante
  et des tests de non-fuite (causal cutoff) ; les timings par étape du
  replay Chicago sont mesurés et publiés dans la note de conception
  phase 2 (lag multi-échelle ~21 s, KDE ~25 s, STKDE ~29 s,
  near-repeat ~17 s, total ~98 s). Hypothesis toujours inutilisé,
  aucun test de performance dédié.
- **Objectif 2 — Atteint (précisé).** La bibliothèque reste paresseuse
  par défaut, avec une exception documentée : les étapes à grosses
  jointures (KDE/STKDE, near-repeat, lag spatial vectorisé) collectent
  leur résultat avant de retourner, une jointure géante paresseuse
  ayant tué le replay par OOM à deux reprises (voir
  `docs/audit/known-limitations.md`).
- **Objectif 6 — Atteint (chiffres mis à jour).** 838 tests, 85 % de
  couverture (mesurés par `make audit`).
- **Objectif 27 — Partiel (inchangé).** Le partitionnement de sortie
  (`output.partition_by`) n'est toujours pas implémenté (toujours une
  erreur bloquante dans la CLI). Les features espace-temps de la phase 2
  ne relèvent pas du libellé de cet objectif.

**Compteur après phase 2 : 15 atteints, 8 partiels, 4 non atteints,
1 non vérifiable** — inchangé. Aucun mouvement de statut en phase 2 :
le travail approfondit des objectifs déjà atteints ou partiels
(lag spatial vectorisé, features espace-temps, garde-fous de validation)
sans en clore de nouveaux. Les objectifs 21–24 restent volontairement
non atteints : ils sont hors phase 2.

## Après phase 3

**Mesuré par `make audit`, 2026-11-16** (voir `audit/phase-3/manifest.json`) :
verdict PASS (5/5 : ruff, black, mypy, pytest, replay), 916 tests passés,
couverture 85 % (6 598 statements, 959 missed) ; replay Chicago T1 2024 :
1 908 088 lignes × 90 colonnes, somme `count` = 60 235 — inchangé par
rapport à la phase 2 (non-régression). Les deux nouvelles preuves sont
hors audit Chicago : Beijing 420 768 lignes × 64 colonnes (12 stations ×
35 064 heures), NA identiques avant/après complétion et après le pipeline
complet (PM2.5 8 739 → 8 739) ; GeoLife sous-ensemble 1 471 lignes ×
26 colonnes (2 utilisateurs, 1 trajectoire chacun), panel de comptage
112 lignes, somme `count` = 1 471 ; connecteur générique démontré
(0/1 471 appariements GeoLife × Beijing — les plages ne se recouvrent
pas — et 8/9 sur le contrôle positif, vérifiés à la main).

- **Objectif 1 — Atteint (renforcé, sans changement de statut).** La
  généricité est désormais prouvée par l'exécution, pas seulement
  déclarée : deuxième domaine (capteurs fixes, mesures avec NA) et
  troisième représentation (trajectoires mobiles) rejoués avec les
  mêmes Steps, sans hypothèse Chicago ; les deux recettes sont
  enregistrées et validées par la CLI (`shikong validate` : 6 et
  10 étapes). Voir `docs/design-notes/phase3-beijing.md` et
  `docs/design-notes/phase3-geolife-boundary.md`.
- **Objectif 13 — Atteint (chiffres mis à jour).** Recette rejouée par
  `make audit` : 1 908 088 lignes × 90 colonnes, somme `count` =
  60 235 (voir `audit/phase-3/replay_chicago.json`).
- **Objectif 15 — Partiel (précisé, statut inchangé).** Le chargement
  de contexte est désormais générique et testé (`TabularFileSource` :
  fichier local déclaré, validation bloquante, jointures temporelles /
  spatiales existantes inchangées) — mais les sources Chicago restent
  non câblées et non testées, et aucun fournisseur live (POI, météo,
  taux) n'existe. Voir `docs/design-notes/phase3-connectors.md`.
- **Objectif 25 — Atteint (maintenu).** Couverture 85 % après phase 3
  (voir `audit/phase-3/manifest.json`).
- **Objectif 26 — Partiel (précisé).** Tests d'intégration sur données
  réelles pour deux nouveaux domaines (Beijing, GeoLife) avec
  vérifications manuelles contre les CSV bruts ; Hypothesis toujours
  inutilisé, aucun test de performance dédié.
- **Objectif 27 — Partiel (inchangé).** Le partitionnement de sortie
  (`output.partition_by`) n'est toujours pas implémenté (toujours une
  erreur bloquante dans la CLI).
- **Objectif 6 — Atteint (chiffres mis à jour).** 916 tests, 85 % de
  couverture (mesurés par `make audit`).

**Compteur après phase 3 : 15 atteints, 8 partiels, 4 non atteints,
1 non vérifiable** — inchangé. Aucun mouvement de statut en phase 3 :
le travail prouve la généricité (objectif 1, déjà atteint) et dote le
contexte d'un chargeur générique testé (objectif 15, toujours partiel)
sans clore de nouvel objectif. Les objectifs 21–24 restent volontairement
non atteints : ils sont hors phase 3.

## Après phase 4

**Mesuré par `make audit`, 2026-11-16** (voir `audit/phase-4/manifest.json`) :
verdict PASS (5/5 : ruff, black, mypy, pytest, replay), 991 tests passés,
couverture 86 % (6 953 statements, 983 missed) ; replay Chicago T1 2024
inchangé : 1 908 088 lignes × 90 colonnes, somme `count` = 60 235.
Replays Beijing et GeoLife rejoués et identiques aux valeurs phase 3
(Beijing 420 768 lignes × 64 colonnes, NA PM2.5 8 739 → 8 739 ;
GeoLife sous-ensemble 1 471 lignes, panel count 112 lignes, somme 1 471 ;
voir `audit/phase-4/phase4_evaluation.json`).

- **Objectif 6 — Atteint (chiffres mis à jour).** 991 tests, 86 % de
  couverture (mesurés par `make audit`).
- **Objectif 13 — Atteint (inchangé).** Recette Chicago rejouée par
  `make audit` : 1 908 088 lignes × 90 colonnes, somme `count` =
  60 235 (voir `audit/phase-4/replay_chicago.json`).
- **Objectif 17 — Partiel (précisé, statut inchangé).** L'inventaire
  des statistiques ajustées est désormais explicite : le top-N de
  `CategoryCountsStep` est la seule statistique ajustée parmi les Steps
  (test d'inventaire) ; `encode_categorical`, `aggregate_by_category`
  et `normalize_features` restent EXPERIMENTAL, ajustés dans l'appel
  mais sans état inter-appels, utilisés par aucune recette — avertissement
  explicite ajouté (`docs/design-notes/phase4-fitted-statistics.md`).
  Le target encoding n'existe pas dans la bibliothèque (retiré en phase 0)
  : l'item est documenté non applicable, aucun encodeur inventé.
- **Objectif 25 — Atteint (maintenu).** Couverture 86 % après phase 4
  (voir `audit/phase-4/manifest.json`).
- **Objectif 26 — Partiel (précisé, statut inchangé).** Évaluation
  anti-fuite désormais outillée et testée : splits bloqués
  (temporel / spatial / entité) avec rejet des splits qui fuient,
  walk-forward à fenêtre croissante avec refit par fold, vérité terrain
  synthétique avec paramètres plantés récupérés (lag-1 0,529 pour 0,6
  planté, hotspot exact, ratio week-end 1,535 pour 1,6 planté) et fuite
  plantée exposée (PR-AUC 1,00 en split aléatoire contre 0,41 en split
  bloqué causal) ; exemples aval PAI / PR-AUC sur Chicago
  (PAI cellule 7,96 à 1 %, PR-AUC 0,0684 pour `count_ma_7`). Hypothesis
  toujours inutilisé, aucun test de performance dédié.
- **Objectif 27 — Partiel (inchangé).** Le partitionnement de sortie
  (`output.partition_by`) n'est toujours pas implémenté (toujours une
  erreur bloquante dans la CLI).

**Compteur après phase 4 : 15 atteints, 8 partiels, 4 non atteints,
1 non vérifiable** — inchangé. Aucun mouvement de statut en phase 4 :
le travail outille l'évaluation anti-fuite (objectifs 17 et 26,
toujours partiels, précisés) et fige le cadrage des statistiques
ajustées sans clore de nouvel objectif. Les objectifs 21–24 restent
volontairement non atteints : ils sont hors phase 4.

## Après phase 5

**Mesuré par `make audit`, 2026-11-16** (voir `audit/phase-5/manifest.json`) :
verdict PASS (5/5 : ruff, black, mypy, pytest, replay), 1 046 tests
passés, 2 ignorés, couverture 86 % (6 982 statements, 979 missed) ;
replay Chicago T1 2024 inchangé : 1 908 088 lignes × 90 colonnes,
somme `count` = 60 235, 1 851 597 lignes à zéro explicite.
Replays Beijing et GeoLife rejoués et identiques aux valeurs phase 3
(Beijing 420 768 lignes × 64 colonnes, NA PM2.5 8 739 → 8 739 ;
GeoLife sous-ensemble 1 471 lignes × 26 colonnes, panel count
112 lignes ; voir `audit/phase-5/beijing_replay.json` et
`audit/phase-5/geolife_replay.json`). Paquet construit depuis une
copie propre : `shikong-0.1.0.tar.gz` + `shikong-0.1.0-py3-none-any.whl`,
`twine check` PASSED sur les deux ; installation propre vérifiée dans
un venv neuf (wheel seule, `shikong --help`, `validate` 21 étapes,
quick start 120 événements → 516 lignes, somme `count` = 120).

- **Objectif 21 — Non atteint → Atteint.** Les alias sont ajoutés et
  testés (`SequenceOutput.attention_mask`, `StreamOutput.inter_event_times`,
  `StreamOutput.state_features` — propriétés en lecture seule retournant
  l'objet tableau identique, anciens noms inchangés, sérialisation
  avec les deux clés, désérialisation acceptant les deux) ; l'accesseur
  uniforme `.data` existe sur les 5 sorties ; la convention
  `feature_names` est actée (`docs/design-notes/phase5-adapter-naming.md`,
  `tests/core/test_adapter_aliases.py`, 21 tests).
- **Objectif 23 — Non atteint → Partiel.** Le paquet se construit et
  passe `twine check` ; `CHANGELOG.md` (Keep a Changelog, 0.1.0 Alpha,
  explicitement non publié) et la procédure de release existent
  (`docs/release.md`). Il reste partiel car rien n'a été publié sur
  PyPI : la publication elle-même est un geste du mainteneur, et
  l'adresse de contact auteur réelle manque encore (noté dans
  `docs/release.md`).
- **Objectif 24 — Non atteint → Partiel.** L'environnement d'exemple
  reproductible existe : `examples/Dockerfile` (exemples uniquement,
  pas de service), `examples/chicago_quickstart.py` exécuté
  (test `tests/integration/test_examples.py`), procédure d'installation
  propre documentée et vérifiée dans un venv neuf
  (`examples/README.md`). Il reste partiel car le Dockerfile lui-même
  n'a pas été construit : aucun Docker dans cette VM.
- **Objectif 22 — Non atteint (inchangé, hors périmètre).** Aucun
  service FastAPI : sorti du périmètre de release par décision du
  mainteneur (retouche TODO, PR #36) ; les services de prédiction
  restent des applications aval optionnelles.
- **Objectif 18 — Partiel (précisé, statut inchangé).** La
  documentation API couvre désormais les phases 1–4
  (`docs/api_reference.md` réécrit : panel, dynamiques, catégoriel +
  fit/freeze, spatial v2, évaluation, connecteurs, recettes Beijing /
  GeoLife ; chaque exemple est exécuté par `tests/test_docs_examples.py`,
  32 tests) ; README, GETTING_STARTED, datasets et architecture
  corrigés (recettes fantômes supprimées, exemples aux vrais Steps).
  Il reste partiel : pas de Sphinx/MkDocs, pas de site ReadTheDocs.
- **Objectif 6 — Atteint (chiffres mis à jour).** 1 046 tests, 86 % de
  couverture (mesurés par `make audit`).
- **Objectif 13 — Atteint (inchangé).** Recette Chicago rejouée par
  `make audit` : 1 908 088 lignes × 90 colonnes, somme `count` =
  60 235 (voir `audit/phase-5/replay_chicago.json`).
- **Objectif 25 — Atteint (maintenu).** Couverture 86 % après phase 5
  (voir `audit/phase-5/manifest.json`).
- **Objectif 27 — Partiel (inchangé).** Le partitionnement de sortie
  (`output.partition_by`) n'est toujours pas implémenté (toujours une
  erreur bloquante dans la CLI).
- **Objectif 28 — Non vérifiable (inchangé, précisé).** Le notebook de
  démo a été exécuté en phase 5 et échoue (`ValueError: cannot concat
  empty list` — il lit des CSV mensuels absents ici) ; il porte
  désormais une mention UNVERIFIED explicite en première cellule et
  `notebooks/README.md` consigne l'échec (`tests/test_docs_examples.py`
  ne l'exécute pas). Ses chiffres (ρ = 0,515, 104 anomalies) ne sont
  présentés nulle part comme des faits.

**Compteur après phase 5 : 16 atteints, 10 partiels, 1 non atteint,
1 non vérifiable.** Mouvements : objectif 21 (non atteint → atteint),
objectifs 23 et 24 (non atteints → partiels). Le seul objectif non
atteint restant est le service FastAPI (objectif 22), hors périmètre
par décision du mainteneur.
