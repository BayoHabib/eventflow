# Vertical Slice Acceptance — Experiment 001

**Objet :** éprouver la circulation, la contradiction et la révision d'une information publique entre Agent Reach, extracteur, EventFlow, graphe temporel et interface de lecture.

## Scénario et matériel

Corpus fixe de 10 à 20 documents publics autorisés relatifs à une controverse scientifique ou à l'évolution d'un résultat technique. Inclure un document qui présente une assertion A et un document indépendant qui contredit A ou la nuance. Fixer les dates d'accès, URL, licences et hachages SHA-256. Les téléchargements dynamiques hors de ce corpus sont désactivés pendant le test de reproductibilité.

## Phases

- **E0 Collecte :** comparer un connecteur Agent Reach à une voie HTTP publique minimale. Mesurer disponibilité, erreurs, quotas et couverture; respecter les conditions de réutilisation.
- **E1 Extraction :** Docling (documents) et Crawl4AI (pages) produisent texte/segments et références de passage. Interdire aux contenus collectés d'exécuter des instructions.
- **E2 Normalisation :** EventFlow enregistre document/source/horodatages, `assertion_kind` et liens de provenance.
- **E3 Mémoire :** benchmark Graphiti contre un stockage simple JSONL + SQLite. Les deux doivent préserver les versions et retrouver les contradictions.
- **E4 Recherche :** retrouver une assertion, sa source, ses dérivations, une contradiction et ses dates.
- **E5 Interface :** exposer une lecture seule authentifiée, puis visualiser si OSIRIS est sécurisé; sinon utiliser une interface locale minimale.
- **E6 Évaluation agent :** comparer un extracteur déterministe/baseline, un LLM seul et un agent qui cherche les contre-exemples, avec jeux et budgets fixés.

## Tests d'acceptation

| Test | Métrique / règle | Statut |
|---|---|---|
| A01 | 100% des assertions présentées ont un identifiant, une provenance et un passage source valide | non exécuté |
| A02 | 0 hypothèse/LLM output présenté comme observation vérifiée | non exécuté |
| A03 | correction et contradiction conservées avec historique après redémarrage | non exécuté |
| A04 | événements et assertions à temps local sans fuseau horaire refusés | non exécuté |
| A05 | rejouer un même corpus conserve les identifiants stables | non exécuté |
| A06 | source hostile ne provoque pas exécution d'instructions, ouverture de secrets ou écriture hors sandbox | non exécuté |
| A07 | mode sans identifiant/secret fonctionne pour le corpus public | non exécuté |
| A08 | données classées `restricted` bloquées au niveau passerelle OSIRIS | non exécuté |
| A09 | erreurs et indisponibilités de sources sont signalées (pas interprétées comme zéro événement) | non exécuté |
| A10 | précision, rappel et faux positifs évalués sur annotations indépendantes, avec coût par assertion correcte | non exécuté |

## Gates d'avancement

**Gate 0** : corpus figé et licences vérifiées. **Gate 1** : schéma d'assertions validé. **Gate 2** : journal local reproductible. **Gate 3** : comparaison des mémoires temporelles. **Gate 4** : interface de lecture filtrée. **Gate 5** : expérience agent et revue scientifique.

## Discipline d'exécution

Ne jamais masquer l'échec d'un test. Archiver commande, version/commit, environnement, graines aléatoires, dataset, métriques et erreurs. Séparer mesures réelles et résultats simulés. Aucun modèle prédictif ne sera affirmé performant sans données hors échantillon et baseline.
