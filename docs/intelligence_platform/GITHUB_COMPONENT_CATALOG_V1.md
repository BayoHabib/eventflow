# GitHub Component Catalog v1 — audit de présélection

Date de consultation : 2026-10-13. **Méthode :** métadonnées des dépôts GitHub + fichiers LICENSE lus directement lorsque mentionnés. **Non effectué :** audit transitive des dépendances, examen de tous les fichiers source, benchmark, essais d'intégration. Les recommandations sont hypothèses à tester.

| Composant | Dépôt | Licence observée dans LICENSE | Fonction | Décision | Risque ou étude nécessaire |
|---|---|---|---|---|---|
| Agent Reach | https://github.com/Panniantong/Agent-Reach | MIT | accès plateformes/sources internet | **pilote** | authentification, droits d'accès, plateformes volatiles |
| Crawl4AI | https://github.com/unclecode/crawl4ai | Apache-2.0 | exploration et extraction Web | **pilote** | robots/conditions, attaques par prompt injection, quotas |
| Docling | https://github.com/docling-project/docling | MIT | conversion documents | **pilote** | formats hostiles, provenance des passages, coût RAM/CPU |
| EventFlow | https://github.com/BayoHabib/eventflow | MIT | traitement spatio-temporel et contrat | **noyau conservé** | versionnage du schéma, ingestion incrémentale |
| Graphiti | https://github.com/getzep/graphiti | Apache-2.0 | graphe de connaissance temporel | **benchmark** | dépendances LLM/graph DB, contradictions, reproductibilité |
| OSIRIS | https://github.com/simplifaisoul/osiris | MIT | cartes, OSINT, interface | **client expérimental** | vulnérabilités d'API relevées dans revue statique précédente; interdiction des données restreintes avant correction |
| Surveillance System | https://github.com/BayoHabib/surveillance-system | MIT (selon README; non vérifiée ici) | orchestration Go et événements de vision | **adaptateur ultérieur** | README indique détecteurs simulés; sécurité et consentement |
| Frigate | https://github.com/blakeblackshear/frigate | MIT (code du dépôt) | pipeline vidéo | **comparaison** | matériel, licence assets/branding, privacy |
| go2rtc | https://github.com/AlexxIT/go2rtc | MIT | proxy/convertisseur de flux | **comparaison** | exposition RTSP, sécurisation réseau |
| LangGraph | https://github.com/langchain-ai/langgraph | MIT | workflows d'agents | **benchmark différé** | complexité et persistance de l'état |
| OpenCTI | https://github.com/opencti-platform/opencti | Apache-2.0 *Community Edition* selon son LICENSE | modèle de renseignement | **référence** | distinctions éditions et conditions, schéma plus cyber que généraliste |
| Qdrant | https://github.com/qdrant/qdrant | Apache-2.0 | index vectoriel | **option différée** | charge opérationnelle, pertinence vs recherche native |

## Politique « buy/build/reuse »

- **Réutiliser:** connecteurs web, parsing de documents, streaming vidéo, indexation, interface.
- **Adapter:** transport d'assertions, intégrations, contrôle de permissions, gestion temporelle.
- **Construire:** moteur de conjectures falsifiables, évaluation de preuves, protocole de révision, métriques scientifiques d'apprentissage.
- **Écarter provisoirement:** infrastructure distribuée complexe sans besoin mesuré; connexion de contenus privés à OSIRIS; activation de services de reconnaissance ou d'analyse individuelle.

## Matrice de décision (pour benchmark ultérieur)

Chaque candidat sera noté sur 0 à 5 avec **preuves reproductibles** pour : adéquation fonctionnelle (25%), interopérabilité (20%), sécurité (20%), licence et usage permis (15%), maintenance (10%) et coût ressources (10%). Il ne s'agit PAS de scores actuellement attribués aux projets. Un échec sur la confidentialité, une licence incompatible ou l'impossibilité de vérifier la provenance est éliminatoire.

## Évaluations obligatoires

1. GitHub: licence au chemin exact + dépendances intégrées + publication des releases + état de maintenance.
2. Sécurité: analyse statique, dépendances, configuration, droits des connecteurs, isolation de l'exécution Web/document.
3. API: version, pagination, retries, identifiants, idempotence, propagation des erreurs et limites.
4. Performance: ingestion 1000 sources/heure en mode test ou métrique réduite déclarée; profil mémoire et CPU.
5. Qualité: récupération d'une assertion sourcée et traitement d'une correction ou contradiction.
6. Coûts: tokens, indexation, stockage, modèles locaux ou distants, licences de services.

## Incertitudes

La licence indiquée est uniquement celle du dépôt consulté; elle ne garantit pas la réutilisabilité de chaque dataset, image, modèle pré-entraîné, dépendance ou marque. Le cas OpenCTI dépend de l'édition. Ne pas considérer l'absence d'alerte de sécurité comme la preuve de l'absence de vulnérabilité.
