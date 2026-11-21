"""Étape 3a – Pré-extraction des données structurées à partir des CRH, pour correction humaine.

Pour chaque patient retenu, écrit <output_dir>/step3_gold/<patient_id>.json :
texte source + extraction automatique + bloc « relecture » à compléter.
Un fichier existant n'est JAMAIS écrasé (protège les corrections humaines).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Any

import mlflow
from pydantic import ValidationError

from common import (TokenCounter, build_chat_llm, invoke_timed, iter_documents, load_config,
                    load_parhaf, output_dir, safe_filename, serialize_documents, set_seed,
                    wait_for_server)
from mlflow_tracking import backup_mlflow, log_config_params, setup_mlflow
from schemas import DonneesPatient, empty_donnees


def select_stratified(ds: Any, counter: TokenCounter, prompt_tpl: str, doc_types: list[str],
                      max_input_tokens: int, n: int, seed: int) -> list[int]:
    """Patients dont le prompt tient dans le budget, répartis entre spécialités (tourniquet)."""
    by_specialty: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(ds):
        if not any(d["type"] in doc_types for d in iter_documents(p)):
            continue
        prompt = prompt_tpl.format(documents=serialize_documents(p, doc_types))
        if counter.count_prompt(prompt) <= max_input_tokens:
            by_specialty[p["specialty"]].append(i)

    rng = random.Random(seed)
    queues = []
    for specialty in sorted(by_specialty):
        idx = by_specialty[specialty][:]
        rng.shuffle(idx)
        queues.append(idx)
    selected: list[int] = []
    while len(selected) < n and any(queues):
        for q in queues:
            if q and len(selected) < n:
                selected.append(q.pop())
    return selected


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    s3 = cfg["step3"]
    set_seed(cfg["seed"])
    gold_dir = output_dir(cfg, s3["gold_dir_name"])

    wait_for_server(cfg["model"]["base_url"])
    counter = TokenCounter(cfg["model"]["tokenizer"])
    llm = build_chat_llm(cfg, s3["temperature"], s3["max_tokens"], DonneesPatient.model_json_schema(),
                         s3.get("repetition_penalty", 1.0))

    ds = load_parhaf(cfg)
    indices = select_stratified(ds, counter, s3["prompt"], s3["doc_types"],
                                s3["max_input_tokens"], s3["n_patients"], cfg["seed"])
    print(f"{len(indices)} patients sélectionnés (répartis par spécialité).")

    setup_mlflow(cfg, "step3")
    mlflow.langchain.autolog()   # une trace par extraction
    counts: Counter[str] = Counter()
    with mlflow.start_run(run_name=f"extraction_{s3['gold_dir_name']}"):
        log_config_params(cfg, "step3")
        for rank, i in enumerate(indices, start=1):
            patient = ds[i]
            path = gold_dir / f"{safe_filename(patient['id'])}.json"
            if path.exists():
                continue
            documents = serialize_documents(patient, s3["doc_types"])
            call = invoke_timed(llm, s3["prompt"].format(documents=documents), name=patient["id"])
            try:
                donnees = DonneesPatient.model_validate_json(call["raw_output"]).model_dump()
                status = "ok"
            except ValidationError:
                donnees, status = empty_donnees(), "invalid_output"

            scenario = patient.get("suggested_scenario") or {}
            payload = {
                "patient_id": patient["id"],
                "specialty": patient["specialty"],
                # Scénario fourni au rédacteur : aide à vérifier âge et sexe (peut différer du texte).
                "scenario": {"age": scenario.get("age"), "sexe": scenario.get("sex")},
                "source_text": documents,
                "extraction": {"modele": cfg["model"]["name"], "statut": status,
                               "latency_s": call["latency_s"], "usage": call["usage"],
                               "raw_output": call["raw_output"] if status != "ok" else None,
                               # Copie figée : permet de mesurer ce que l'annotateur a corrigé.
                               "donnees_initiales": donnees},
                "donnees": json.loads(json.dumps(donnees)),  # copie éditable
                "relecture": {"statut": "a_relire", "relecteur": None, "date": None,
                              "temps_min": None, "notes": ""},
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{rank}/{len(indices)}] {patient['id']} -> {status} ({call['latency_s']} s)")
            counts[status] += 1
            mlflow.log_metrics({"latence_s": call["latency_s"],
                                "tokens_sortie": call["usage"].get("output_tokens", 0),
                                "extraction_valide": float(status == "ok")}, step=rank)
            if rank % cfg["mlflow"]["backup_every"] == 0:
                backup_mlflow(cfg)
        if counts:
            mlflow.log_metric("pct_extraction_valide", 100 * counts["ok"] / sum(counts.values()))
    backup_mlflow(cfg)

    print(f"Brouillons dans {gold_dir}. Étape suivante : correction manuelle (voir README).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
