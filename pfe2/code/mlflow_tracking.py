"""Suivi MLflow commun aux étapes 1 à 3.

Stockage :
  - la base MLflow (SQLite + artefacts) vit en local dans `mlflow.local_dir` (/content/mlflow)
    car SQLite sur Google Drive est sujet aux verrous ;
  - elle est sauvegardée sur Drive (`mlflow.drive_backup_dir`) régulièrement et en fin de script,
    puis restaurée automatiquement au début d'une nouvelle session Colab.
Interface : `mlflow server` lancé depuis runner.ipynb (voir README).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import mlflow


def _paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    m = cfg["mlflow"]
    return Path(m["local_dir"]), Path(m["drive_backup_dir"])


def setup_mlflow(cfg: dict[str, Any], experiment_key: str) -> str:
    """Restaure la base si besoin, configure le tracking et sélectionne l'expérience.

    `experiment_key` : clé de `mlflow.experiments` dans config.yaml (step1, step2, step3).
    Retourne l'identifiant de l'expérience.
    """
    local, backup = _paths(cfg)
    if not (local / "mlflow.db").exists() and (backup / "mlflow.db").exists():
        shutil.copytree(backup, local, dirs_exist_ok=True)   # nouvelle session Colab
        print(f"Base MLflow restaurée depuis {backup}")
    (local / "artifacts").mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{local / 'mlflow.db'}")
    name = cfg["mlflow"]["experiments"][experiment_key]
    exp = mlflow.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        name, artifact_location=(local / "artifacts" / name).as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def backup_mlflow(cfg: dict[str, Any]) -> None:
    """Copie la base locale sur Drive (à appeler régulièrement : Colab peut couper)."""
    local, backup = _paths(cfg)
    if local.exists():
        shutil.copytree(local, backup, dirs_exist_ok=True)


def find_run_id(experiment_id: str, run_name: str) -> str | None:
    """Run le plus récent portant ce nom (pour reprendre la génération ou y ajouter l'évaluation)."""
    runs = mlflow.search_runs([experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    return None if runs.empty else str(runs.iloc[0]["run_id"])


def log_config_params(cfg: dict[str, Any], step_key: str) -> None:
    """Paramètres du run : modèle + section de l'étape. Le prompt va en artefact (trop long)."""
    step = cfg[step_key]
    params = {
        "seed": cfg["seed"],
        "model": cfg["model"]["name"],
        "max_model_len": cfg["model"]["max_model_len"],
        **{k: (",".join(map(str, v)) if isinstance(v, list) else v)
           for k, v in step.items() if k != "prompt"},
    }
    mlflow.log_params(params)
    if "prompt" in step:
        mlflow.log_text(step["prompt"], "prompt.txt")
