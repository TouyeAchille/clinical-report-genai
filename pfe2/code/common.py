"""Fonctions partagées par les étapes 1 à 3."""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:  # imports lourds seulement pour le typage
    from datasets import Dataset
    from langchain_openai import ChatOpenAI


# ------------------------------------------------------------------ Config
def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def output_dir(cfg: dict[str, Any], sub: str = "") -> Path:
    path = Path(cfg["paths"]["output_dir"]) / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ PARHAF
def load_parhaf(cfg: dict[str, Any]) -> "Dataset":
    from datasets import load_dataset

    return load_dataset(cfg["data"]["dataset"], split=cfg["data"]["split"])


def to_records(value: Any) -> list[dict[str, Any]]:
    """Séquence HF (dict de listes) ou liste de dicts -> liste de dicts."""
    if value is None:
        return []
    if isinstance(value, dict):
        keys = list(value)
        if not keys:
            return []
        return [dict(zip(keys, row)) for row in zip(*(value[k] or [] for k in keys))]
    return list(value)


def iter_documents(patient: dict[str, Any]) -> list[dict[str, Any]]:
    return to_records(patient["documents"])


def serialize_documents(patient: dict[str, Any], doc_types: list[str]) -> str:
    """Concatène les documents des types demandés, avec leur en-tête."""
    parts = [
        f"### {d['type']} – {d.get('header') or ''}\n{(d['text'] or '').strip()}"
        for d in iter_documents(patient)
        if d["type"] in doc_types
    ]
    return "\n\n".join(parts)


def normalize_abstract(abstract: dict[str, Any] | None) -> dict[str, Any]:
    """`structured_abstract` au format liste de dicts, quel que soit le stockage HF."""
    abstract = abstract or {}
    return {
        "primary_diagnosis": to_records(abstract.get("primary_diagnosis")),
        "primary_procedure": to_records(abstract.get("primary_procedure")),
        "admission_mode": abstract.get("admission_mode"),
        "discharge_mode": abstract.get("discharge_mode"),
        "length_of_stay": abstract.get("length_of_stay"),
    }


def has_reference(patient: dict[str, Any]) -> bool:
    """Vrai si le patient a au moins un diagnostic principal dans `structured_abstract`."""
    return bool(normalize_abstract(patient.get("structured_abstract"))["primary_diagnosis"])


# ------------------------------------------------------------------ Tokens
class TokenCounter:
    """Compte les tokens comme le modèle les verra (gabarit de chat inclus)."""

    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)

    def count_text(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def count_prompt(self, prompt: str) -> int:
        if getattr(self.tok, "chat_template", None):
            rendered = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:  # repli : format Mistral Instruct
            rendered = f"<s>[INST] {prompt} [/INST]"
        return self.count_text(rendered)


# ------------------------------------------------------------------ JSONL (reprise après coupure)
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def done_ids(path: Path) -> set[str]:
    return {r["patient_id"] for r in read_jsonl(path)}


def safe_filename(patient_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(patient_id))


# ------------------------------------------------------------------ Serveur vLLM + LangChain
def wait_for_server(base_url: str, timeout_s: int = 900) -> None:
    import requests

    health_url = base_url.removesuffix("/v1") + "/health"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            if requests.get(health_url, timeout=5).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"Serveur vLLM injoignable : {health_url} (voir vllm.log)")


def build_chat_llm(cfg: dict[str, Any], temperature: float, max_tokens: int,
                   json_schema: dict[str, Any], repetition_penalty: float = 1.0) -> "ChatOpenAI":
    """Client LangChain vers vLLM, avec décodage contraint par un schéma JSON.

    `repetition_penalty` (> 1) décourage les boucles du type « pas de traitement » x 50.
    Paramètre propre à vLLM, transmis via extra_body.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg["model"]["name"],
        base_url=cfg["model"]["base_url"],
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        seed=cfg["seed"],
        timeout=600,
        max_retries=1,
        # Paramètre propre à vLLM (>= 0.12), remplace l'ancien guided_json.
        extra_body={"structured_outputs": {"json": json_schema},
                    "repetition_penalty": repetition_penalty},
    )


def invoke_timed(llm: "ChatOpenAI", prompt: str, name: str | None = None) -> dict[str, Any]:
    """Appel unique, sans message system (gabarit Mistral v0.1).

    `name` (ex. l'identifiant patient) nomme l'appel dans les traces MLflow.
    """
    from langchain_core.messages import HumanMessage

    t0 = time.perf_counter()
    msg = llm.invoke([HumanMessage(content=prompt)], config={"run_name": name} if name else None)
    return {
        "latency_s": round(time.perf_counter() - t0, 3),
        "usage": dict(msg.usage_metadata or {}),
        "raw_output": msg.content,
    }


# ------------------------------------------------------------------ Texte
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> set[str]:
    """Nombres d'un texte, virgule décimale normalisée en point."""
    return {m.replace(",", ".") for m in NUMBER_RE.findall(text or "")}


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()
