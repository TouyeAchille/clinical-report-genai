# PFE – Étapes 1 à 3 sur PARHAF avec BioMistral (Colab)

| Fichier | Rôle | GPU |
| --- | --- | --- |
| `config.yaml` | Tous les paramètres (seed, chemins, prompts, budgets) | – |
| `schemas.py` | Schémas Pydantic : `ResumeSejour` (étape 2), `DonneesPatient` (étape 3) | – |
| `common.py` | Chargement PARHAF, comptage de tokens, JSONL avec reprise, client LangChain → vLLM | – |
| `step1_explore.py` | Caractérisation du corpus face à la fenêtre de 2 048 tokens | Non |
| `step2_generate.py` | Baseline : CRH → résumé JSON | Oui |
| `step2_evaluate.py` | Métriques de la baseline | Non |
| `step3_extract.py` | Pré-extraction des données structurées pour correction humaine | Oui |
| `step3_validate.py` | Contrôle des fichiers corrigés, export du jeu de référence | Non |

## Installation sur Colab

Copier le dossier dans `MyDrive/pfe/code`, puis dans un notebook :

```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # Secrets Colab (icône clé)
%cd /content/drive/MyDrive/pfe/code
!pip install -q -r requirements.txt
!pip freeze > requirements.lock.txt
```

Si l'installation de vLLM remplace PyTorch, redémarrer la session puis relancer la cellule (sans le `pip install`).

Lancer le serveur vLLM (étapes 2 et 3 uniquement) :

```bash
!nohup vllm serve BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM \
    --quantization awq --max-model-len 2048 --enforce-eager --seed 42 \
    --port 8000 > /content/vllm.log 2>&1 &
```

Le chargement prend 2 à 3 minutes ; les scripts attendent eux-mêmes que le serveur réponde. En cas de problème : `!tail -50 /content/vllm.log`.

Tous les résultats vont dans `MyDrive/pfe/outputs`. Les scripts 2 et 3 reprennent là où ils se sont arrêtés si Colab coupe : il suffit de les relancer.

---

## Étape 1 – Caractériser le corpus (runtime CPU, ~10 min)

**But.** Savoir quelle part de PARHAF BioMistral peut traiter, avant de lancer le moindre calcul GPU.

```bash
!python step1_explore.py
```

**Sorties** (`outputs/step1/`) : `summary.md`, `tokens_par_type.png`, `tokens_par_patient.png`, `documents.csv`, `patients.csv`.

**À regarder.**
- La part des CRH qui tiennent dans le budget (colonnes `pct_tient_256` et `pct_tient_512`). Si elle est faible, c'est l'argument chiffré pour comparer avec un modèle à long contexte.
- Le ratio tokens par mot : il remplace mon estimation de mémoire (1,3 à 1,6).
- Le remplissage de `structured_abstract` : il dit sur combien de patients l'étape 2 est évaluable, champ par champ.

**Décision à prendre ensuite.** Garder `doc_types: [CRH]` ou changer de type selon les chiffres ; ajuster `max_tokens` de sortie.

## Étape 2 – Baseline CRH → résumé JSON (GPU T4, ~1 h pour 40 patients)

**But.** Obtenir le premier point de référence chiffré de BioMistral.

```bash
!python step2_generate.py      # écrit outputs/step2/<run_name>.jsonl
!python step2_evaluate.py      # écrit <run_name>_par_patient.csv et <run_name>_resume.json
```

**Ce qui est mesuré** (`step2_evaluate.py`) :

| Métrique | Signification |
| --- | --- |
| `pct_json_valide`, `pct_sortie_invalide`, `pct_depassement_contexte` | Robustesse et couverture |
| `diag_code_exact_pct`, `diag_code_3car_pct`, `diag_sim_moyenne` | Diagnostic principal face à `structured_abstract` |
| `acte_*`, `entree_sim_moyenne`, `sortie_sim_moyenne`, `duree_*` | Autres champs de référence |
| `pct_nombres_non_sources` | Nombres de la sortie absents du texte : indicateur d'hallucination |
| `pct_codes_non_sources` | Codes inventés (non écrits dans le texte) |
| `latence_*`, `tokens_*` | Coût et performance système |

**Limites à garder en tête.**
- `structured_abstract` n'est pas curé : relire à la main les lignes `a_relire = True` du CSV avant de conclure.
- La similarité lexicale (`token_set_ratio`) récompense les inclusions : « urgences » face à « entrée par les urgences » vaut 1,0. C'est un indicateur, pas une vérité.
- Les codes CIM-10 ne sont pas toujours écrits dans les CRH : un `diag_code_exact` faible peut signifier que le modèle a (correctement) mis `null`. Regarder aussi `pct_codes_non_sources`.

**Faire varier ensuite** (une variable à la fois, en changeant `run_name`) : prompt, `max_tokens`, type de document, puis modèle.

## Étape 3 – Jeu de référence « données structurées → CR » (GPU ~30 min + correction humaine)

**But.** Créer 20 à 30 paires françaises « données structurées corrigées → CRH humain ». C'est l'entrée exacte du sujet et le jeu d'évaluation relu demandé par la guideline. L'étape 4 générera un CR à partir de `donnees` et le comparera au CRH d'origine.

```bash
!python step3_extract.py       # un fichier par patient dans outputs/step3_gold/
# ... correction manuelle des fichiers ...
!python step3_validate.py      # contrôle + export outputs/step3_gold_set.jsonl
```

La sélection est stratifiée par spécialité et ne retient que les CRH qui tiennent dans 1 300 tokens.

### Guide de correction (à suivre par tous les annotateurs)

Ouvrir chaque fichier `step3_gold/<id>.json` (éditeur de texte, VS Code ou Drive) et corriger **uniquement** le bloc `donnees`, en lisant `source_text`.

1. **Rien qui ne soit pas dans le texte.** Supprimer toute information absente du CRH, même plausible.
2. **Rien d'important d'oublié.** Ajouter les faits manquants dans la bonne rubrique.
3. **Valeurs recopiées telles qu'écrites** : `"1,2"` et pas `"1.2"`, unités et posologies comprises.
4. **Un fait par élément de liste**, formulé brièvement.
5. Âge et sexe : vérifier avec le texte ; `scenario` n'est qu'une aide et peut différer.
6. Ne jamais modifier `extraction.donnees_initiales` : il sert à mesurer la qualité de l'extraction automatique.
7. Remplir `relecture` : `statut` (`relu` ou `rejete` si le CRH est inexploitable), `relecteur` (initiales), `date`, `temps_min`, `notes`.

Conseil : faire corriger les 5 premiers fichiers par deux personnes, comparer et préciser le guide avant de continuer. Si un médecin peut relire, lui confier un échantillon.

`step3_validate.py` signale :
- les erreurs (JSON illisible, statut inconnu, schéma non respecté) à corriger ;
- les avertissements de traçabilité (valeur ou molécule introuvable dans le texte) à vérifier ;
- la **précision et le rappel de l'extraction automatique** : part des éléments extraits conservés tels quels par l'annotateur. C'est un résultat gratuit sur BioMistral en extraction.

**Important pour la suite.** Réserver 3 à 5 paires pour les exemples few-shot de l'étape 4 et ne jamais les utiliser en test.

---

## Ce qu'il faut noter dans le journal

- La version exacte du code, `config.yaml` et `requirements.lock.txt` de chaque run.
- Les résultats de l'étape 1 et la décision sur le périmètre (types de documents, budget).
- Le `_resume.json` de chaque run de l'étape 2, avec ce qui a changé.
- Le temps de correction moyen par fichier à l'étape 3 (coût réel du jeu de référence).
