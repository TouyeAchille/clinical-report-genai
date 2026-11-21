"""Étape 1 – Caractérisation de PARHAF vis-à-vis de la fenêtre de BioMistral.

Pas besoin de GPU (runtime CPU Colab suffit, quelques minutes).
Sorties dans <output_dir>/step1 : documents.csv, patients.csv, summary.md, figures PNG.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import (TokenCounter, iter_documents, load_config, load_parhaf,  # noqa: E402
                    normalize_abstract, output_dir, set_seed)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    out = output_dir(cfg, "step1")
    counter = TokenCounter(cfg["model"]["tokenizer"])
    ds = load_parhaf(cfg)

    max_len: int = cfg["model"]["max_model_len"]
    reserves: list[int] = cfg["step1"]["output_reserve_tokens"]
    # Coût fixe du prompt de l'étape 2 (consignes + gabarit de chat), sans les documents.
    overhead = counter.count_prompt(cfg["step2"]["prompt"].format(documents=""))

    doc_rows: list[dict] = []
    patient_rows: list[dict] = []
    for patient in ds:
        docs = iter_documents(patient)
        total_tokens = 0
        for d in docs:
            text = d["text"] or ""
            n_tokens = counter.count_text(text)
            total_tokens += n_tokens
            doc_rows.append({
                "patient_id": patient["id"],
                "specialty": patient["specialty"],
                "type": d["type"],
                "words": len(text.split()),
                "tokens": n_tokens,
            })
        abstract = normalize_abstract(patient.get("structured_abstract"))
        los = abstract["length_of_stay"] or {}
        patient_rows.append({
            "patient_id": patient["id"],
            "specialty": patient["specialty"],
            "n_docs": len(docs),
            "doc_types": "+".join(sorted(d["type"] for d in docs)),
            "tokens_all_docs": total_tokens,
            "abs_diagnostic": bool(abstract["primary_diagnosis"]),
            "abs_acte": bool(abstract["primary_procedure"]),
            "abs_mode_entree": abstract["admission_mode"] is not None,
            "abs_mode_sortie": abstract["discharge_mode"] is not None,
            "abs_duree": los.get("value") is not None,
        })

    docs_df = pd.DataFrame(doc_rows)
    pat_df = pd.DataFrame(patient_rows)
    for r in reserves:
        budget = max_len - r - overhead
        docs_df[f"tient_{r}"] = docs_df["tokens"] <= budget
        # Approximation : ignore les quelques tokens d'en-tête ajoutés entre documents.
        pat_df[f"tient_{r}"] = pat_df["tokens_all_docs"] <= budget

    # --- Tableau par type de document
    agg: dict = {
        "n": ("tokens", "size"),
        "mots_median": ("words", "median"),
        "tokens_median": ("tokens", "median"),
        "tokens_p90": ("tokens", lambda s: s.quantile(0.9)),
        "tokens_max": ("tokens", "max"),
    }
    agg.update({f"pct_tient_{r}": (f"tient_{r}", lambda s: 100 * s.mean()) for r in reserves})
    by_type = docs_df.groupby("type").agg(**agg).round(1).sort_values("n", ascending=False)

    tokens_per_word = docs_df["tokens"].sum() / max(docs_df["words"].sum(), 1)
    fill_rates = (100 * pat_df.filter(like="abs_").mean()).round(1)
    pat_fit = {r: round(100 * pat_df[f"tient_{r}"].mean(), 1) for r in reserves}

    # --- Figures
    top_types = by_type.index[:4]
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in top_types:
        ax.hist(docs_df.loc[docs_df["type"] == t, "tokens"], bins=60, alpha=0.5, label=t)
    for r in reserves:
        ax.axvline(max_len - r - overhead, linestyle="--", color="black")
        ax.text(max_len - r - overhead, ax.get_ylim()[1] * 0.9, f" budget (sortie {r})", fontsize=8)
    ax.set(xlabel="Tokens BioMistral par document", ylabel="Nombre de documents",
           title="PARHAF : longueur des documents face à la fenêtre de 2 048 tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tokens_par_type.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pat_df["tokens_all_docs"], bins=60)
    ax.axvline(max_len - max(reserves) - overhead, linestyle="--", color="black")
    ax.set(xlabel="Tokens de tous les documents d'un patient", ylabel="Nombre de patients",
           title="PARHAF : longueur cumulée par patient")
    fig.tight_layout()
    fig.savefig(out / "tokens_par_patient.png", dpi=150)

    # --- Sauvegardes
    docs_df.to_csv(out / "documents.csv", index=False)
    pat_df.to_csv(out / "patients.csv", index=False)
    summary = "\n\n".join([
        "# Étape 1 – Caractérisation de PARHAF",
        f"Patients : {len(pat_df)} | Documents : {len(docs_df)} | "
        f"Coût fixe du prompt : {overhead} tokens | Tokens par mot : {tokens_per_word:.2f}",
        "## Par type de document\n" + by_type.to_markdown(),
        "## Patients dont tous les documents tiennent dans la fenêtre (%)\n"
        + "\n".join(f"- sortie réservée {r} : {v} %" for r, v in pat_fit.items()),
        "## Remplissage de structured_abstract (%)\n" + fill_rates.to_markdown(),
        "## Combinaisons de documents par patient\n"
        + pat_df["doc_types"].value_counts().head(10).to_markdown(),
    ])
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
