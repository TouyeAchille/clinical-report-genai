"""Étape 3b – Contrôle des fichiers corrigés à la main et export du jeu de référence.

Pour chaque fichier de step3_gold :
  - JSON lisible et conforme au schéma DonneesPatient ;
  - avertissements de traçabilité : valeurs biologiques et molécules introuvables
    dans le texte source (pas forcément des erreurs, mais à vérifier) ;
  - qualité de l'extraction automatique : précision et rappel des éléments de liste
    conservés tels quels par l'annotateur (résultat gratuit sur BioMistral en extraction).
Exporte les fichiers au statut « relu » dans <output_dir>/step3_gold_set.jsonl.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import mlflow
from pydantic import ValidationError

from common import append_jsonl, extract_numbers, load_config, normalize_text, output_dir
from mlflow_tracking import backup_mlflow, setup_mlflow
from schemas import DonneesPatient

VALID_STATUSES = {"a_relire", "relu", "rejete"}


def traceability_warnings(donnees: DonneesPatient, source: str) -> list[str]:
    src_norm, src_numbers = normalize_text(source), extract_numbers(source)
    warnings = []
    for bio in donnees.biologie:
        missing = extract_numbers(bio.valeur) - src_numbers
        if missing:
            warnings.append(f"biologie '{bio.examen}' : valeur {sorted(missing)} absente du texte")
    for trt in donnees.traitements_sortie:
        if normalize_text(trt.molecule) not in src_norm:
            warnings.append(f"traitement '{trt.molecule}' absent du texte (forme différente ?)")
    return warnings


def kept_items(initial: dict[str, Any], final: dict[str, Any]) -> tuple[int, int, int]:
    """(éléments extraits, éléments finaux, éléments conservés à l'identique), toutes listes confondues."""
    n_init = n_final = n_kept = 0
    for name, value in final.items():
        if not isinstance(value, list):
            continue
        init_set = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in initial.get(name) or []}
        final_set = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in value}
        n_init, n_final, n_kept = n_init + len(init_set), n_final + len(final_set), n_kept + len(init_set & final_set)
    return n_init, n_final, n_kept


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    gold_dir = output_dir(cfg, cfg["step3"]["gold_dir_name"])
    export_path = output_dir(cfg) / "step3_gold_set.jsonl"
    export_path.unlink(missing_ok=True)

    statuses: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    n_errors = 0
    totals = [0, 0, 0]  # extraits, finaux, conservés
    for path in sorted(gold_dir.glob("*.json")):
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            print(f"ERREUR {path.name} : JSON illisible ({err})")
            n_errors += 1
            continue

        status = payload.get("relecture", {}).get("statut")
        if status not in VALID_STATUSES:
            print(f"ERREUR {path.name} : statut inconnu '{status}'")
            n_errors += 1
            continue
        statuses[status] += 1
        if status != "relu":
            continue

        try:
            donnees = DonneesPatient.model_validate(payload["donnees"])
        except ValidationError as err:
            print(f"ERREUR {path.name} : schéma invalide\n{err}")
            n_errors += 1
            continue

        for w in traceability_warnings(donnees, payload["source_text"]):
            print(f"AVERTISSEMENT {path.name} : {w}")
        initial = payload.get("extraction", {}).get("donnees_initiales") or {}
        for k, v in enumerate(kept_items(initial, donnees.model_dump())):
            totals[k] += v
        for name, value in donnees.model_dump().items():
            if isinstance(value, list):
                field_counts[name] += len(value)

        append_jsonl(export_path, {
            "patient_id": payload["patient_id"],
            "specialty": payload["specialty"],
            "donnees": donnees.model_dump(),
            "reference_text": payload["source_text"],   # CRH humain = référence de l'étape 4
            "relecteur": payload["relecture"].get("relecteur"),
        })

    print(f"\nStatuts : {dict(statuses)} | erreurs : {n_errors}")
    print(f"Faits par rubrique (fichiers relus) : {dict(field_counts)}")
    n_init, n_final, n_kept = totals
    if n_init and n_final:
        print(f"Extraction automatique : précision {100 * n_kept / n_init:.1f} % | "
              f"rappel {100 * n_kept / n_final:.1f} % (éléments conservés tels quels)")
    print(f"Jeu de référence exporté : {export_path}")

    # MLflow : un run par validation (l'avancement de la correction devient une courbe).
    setup_mlflow(cfg, "step3")
    with mlflow.start_run(run_name=f"validation_{cfg['step3']['gold_dir_name']}"):
        metrics = {f"n_{k}": float(v) for k, v in statuses.items()}
        metrics.update({"n_erreurs": float(n_errors),
                        **{f"faits_{k}": float(v) for k, v in field_counts.items()}})
        if n_init and n_final:
            metrics.update({"extraction_precision_pct": 100 * n_kept / n_init,
                            "extraction_rappel_pct": 100 * n_kept / n_final})
        mlflow.log_metrics(metrics)
        if export_path.exists():
            mlflow.log_artifact(str(export_path))
    backup_mlflow(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
