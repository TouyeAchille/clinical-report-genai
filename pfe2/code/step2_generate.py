"""Étape 2a – Baseline : documents PARHAF -> résumé JSON par BioMistral (vLLM + LangChain).

Prérequis : serveur vLLM lancé (voir README). Reprend là où il s'est arrêté après
une coupure Colab : les patients déjà présents dans le JSONL sont ignorés.
Sortie : <output_dir>/step2/<run_name>.jsonl
MLflow : un run nommé <run_name> (paramètres, prompt, latence et tokens par patient)
et une trace par appel au modèle (autolog LangChain).
"""
from __future__ import annotations

import argparse
import random
from typing import Any

import mlflow
from pydantic import ValidationError

from common import (TokenCounter, append_jsonl, build_chat_llm, done_ids, invoke_timed,
                    iter_documents, load_config, load_parhaf, normalize_abstract, output_dir,
                    serialize_documents, set_seed, has_reference, wait_for_server)
from mlflow_tracking import backup_mlflow, find_run_id, log_config_params, setup_mlflow
from schemas import ResumeSejour


def select_patients(ds: Any, doc_types: list[str], n: int, seed: int) -> list[int]:
    """Échantillon aléatoire parmi les patients ayant une référence et un document du bon type.

    Le filtre de longueur n'est PAS appliqué ici : le taux de dépassement de
    contexte est mesuré sur l'échantillon, c'est un résultat.
    """
    eligible = [
        i for i, p in enumerate(ds)
        if has_reference(p) and any(d["type"] in doc_types for d in iter_documents(p))
    ]
    return sorted(random.Random(seed).sample(eligible, k=min(n, len(eligible))))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    s2 = cfg["step2"]
    set_seed(cfg["seed"])
    out_path = output_dir(cfg, "step2") / f"{s2['run_name']}.jsonl"
    already_done = done_ids(out_path)

    wait_for_server(cfg["model"]["base_url"])
    counter = TokenCounter(cfg["model"]["tokenizer"])
    llm = build_chat_llm(cfg, s2["temperature"], s2["max_tokens"], ResumeSejour.model_json_schema(),
                         s2.get("repetition_penalty", 1.0))
    input_budget = cfg["model"]["max_model_len"] - s2["max_tokens"]

    ds = load_parhaf(cfg)
    indices = select_patients(ds, s2["doc_types"], s2["n_patients"], cfg["seed"])
    print(f"{len(indices)} patients sélectionnés, {len(already_done)} déjà traités.")

    exp_id = setup_mlflow(cfg, "step2")
    mlflow.langchain.autolog()   # une trace par appel : prompt, sortie, latence, tokens
    run_id = find_run_id(exp_id, s2["run_name"]) if already_done else None
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else s2["run_name"]):
        if run_id is None:
            log_config_params(cfg, "step2")
        for rank, i in enumerate(indices, start=1):
            patient = ds[i]
            if patient["id"] in already_done:
                continue
            documents = serialize_documents(patient, s2["doc_types"])
            prompt = s2["prompt"].format(documents=documents)
            record: dict[str, Any] = {
                "patient_id": patient["id"],
                "specialty": patient["specialty"],
                "run_name": s2["run_name"],
                "model": cfg["model"]["name"],
                "prompt_tokens_est": counter.count_prompt(prompt),
                "source_text": documents,
                "reference": normalize_abstract(patient.get("structured_abstract")),
            }
            if record["prompt_tokens_est"] > input_budget:
                record["status"] = "skipped_context_overflow"  # pas de troncature silencieuse
            else:
                record.update(invoke_timed(llm, prompt, name=patient["id"]))
                try:
                    record["prediction"] = ResumeSejour.model_validate_json(record["raw_output"]).model_dump()
                    record["status"] = "ok"
                except ValidationError as err:
                    record["status"] = "invalid_output"
                    record["error"] = str(err)
                mlflow.log_metrics({
                    "latence_s": record["latency_s"],
                    "tokens_entree": record["usage"].get("input_tokens", 0),
                    "tokens_sortie": record["usage"].get("output_tokens", 0),
                }, step=rank)
            append_jsonl(out_path, record)
            print(f"[{rank}/{len(indices)}] {patient['id']} -> {record['status']} "
                  f"({record.get('latency_s', '-')} s)")
            if rank % cfg["mlflow"]["backup_every"] == 0:
                backup_mlflow(cfg)
        mlflow.log_artifact(str(out_path))   # JSONL complet, téléchargeable depuis l'interface
    backup_mlflow(cfg)

    print(f"Terminé : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
