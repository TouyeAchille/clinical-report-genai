"""Étape 2b – Évaluation de la baseline (aucun GPU nécessaire).

Métriques :
  1. statuts : JSON valide, sortie invalide, dépassement de contexte ;
  2. exactitude par champ face à `structured_abstract` (codes, libellés, modes, durée) ;
  3. nombres non sourcés : nombres de la sortie absents du texte source ;
  4. codes non sourcés : codes CIM-10 / CCAM de la sortie absents du texte source ;
  5. latence et tokens.
Sorties dans <output_dir>/step2 : <run_name>_par_patient.csv et <run_name>_resume.json.

Attention : `structured_abstract` n'est pas curé et peut contredire le texte.
Les cas signalés `a_relire` doivent être vérifiés à la main avant toute conclusion.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from common import extract_numbers, load_config, normalize_text, output_dir, read_jsonl

CIM10_RE = re.compile(r"\b[A-Z]\d{2}(?:\.?\d{1,2})?\b")
CCAM_RE = re.compile(r"\b[A-Z]{4}\d{3}\b")


def code_norm(code: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


def codes_in_text(text: str) -> set[str]:
    upper = text.upper()
    return {code_norm(c) for c in CIM10_RE.findall(upper) + CCAM_RE.findall(upper)}


def best_similarity(refs: list[str], preds: list[str]) -> float | None:
    """Similarité lexicale maximale (0-1) entre libellés de référence et prédits."""
    refs = [r for r in refs if r]
    if not refs:
        return None
    preds = [p for p in preds if p]
    if not preds:
        return 0.0
    return max(fuzz.token_set_ratio(normalize_text(r), normalize_text(p)) for r in refs for p in preds) / 100


def code_match(refs: list[dict], preds: list[dict], n_chars: int | None = None) -> bool | None:
    """Vrai si un code prédit égale un code de référence (option : n premiers caractères)."""
    cut = (lambda c: c[:n_chars]) if n_chars else (lambda c: c)
    ref_codes = {cut(code_norm(r.get("code"))) for r in refs} - {""}
    if not ref_codes:
        return None
    pred_codes = {cut(code_norm(p.get("code"))) for p in preds} - {""}
    return bool(ref_codes & pred_codes)


def evaluate_record(rec: dict[str, Any], threshold: float) -> dict[str, Any]:
    ref, pred, source = rec["reference"], rec["prediction"], rec["source_text"]
    row: dict[str, Any] = {"patient_id": rec["patient_id"], "specialty": rec["specialty"]}

    for name, ref_key, pred_key in [("diag", "primary_diagnosis", "diagnostic_principal"),
                                    ("acte", "primary_procedure", "acte_principal")]:
        refs, preds = ref[ref_key], pred[pred_key]
        row[f"{name}_code_exact"] = code_match(refs, preds)
        row[f"{name}_code_3car"] = code_match(refs, preds, n_chars=3)
        row[f"{name}_sim"] = best_similarity([r.get("description") for r in refs],
                                             [p.get("description") for p in preds])

    row["entree_sim"] = best_similarity([ref["admission_mode"]], [pred["mode_entree"]])
    row["sortie_sim"] = best_similarity([ref["discharge_mode"]], [pred["mode_sortie"]])

    los = ref.get("length_of_stay") or {}
    if los.get("value") is not None and str(los.get("unit") or "").lower().startswith("j"):
        diff = None if pred["duree_sejour_jours"] is None else abs(pred["duree_sejour_jours"] - los["value"])
        row["duree_exacte"] = diff == 0
        row["duree_plus_moins_1"] = diff is not None and diff <= 1
    else:
        row["duree_exacte"] = row["duree_plus_moins_1"] = None

    # Nombres non sourcés. La durée est exclue : elle peut être calculée à partir des dates.
    pred_text = " ".join(
        [i["description"] for i in pred["diagnostic_principal"] + pred["acte_principal"]]
        + [pred["mode_entree"] or "", pred["mode_sortie"] or "", pred["synthese"]]
        + pred["antecedents"] + pred["traitements_sortie"]
    )
    numbers = extract_numbers(pred_text)
    unsourced = sorted(numbers - extract_numbers(source))
    row["n_nombres"], row["n_nombres_non_sources"] = len(numbers), len(unsourced)
    row["nombres_non_sources"] = ", ".join(unsourced)

    pred_codes = {code_norm(i["code"]) for i in pred["diagnostic_principal"] + pred["acte_principal"]} - {""}
    bad_codes = sorted(pred_codes - codes_in_text(source))
    row["n_codes"], row["n_codes_non_sources"] = len(pred_codes), len(bad_codes)
    row["codes_non_sources"] = ", ".join(bad_codes)

    row["latence_s"] = rec.get("latency_s")
    row["tokens_entree"] = rec.get("usage", {}).get("input_tokens")
    row["tokens_sortie"] = rec.get("usage", {}).get("output_tokens")
    row["a_relire"] = bool(
        unsourced or bad_codes or row["diag_code_exact"] is False
        or (row["diag_sim"] is not None and row["diag_sim"] < threshold)
    )
    return row


def rate(series: pd.Series) -> float | None:
    """Moyenne (en %) en ignorant les cas non applicables (None)."""
    s = series.dropna().astype(float)
    return None if s.empty else round(100 * s.mean(), 1)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    s2 = cfg["step2"]
    out = output_dir(cfg, "step2")
    records = read_jsonl(out / f"{s2['run_name']}.jsonl")
    if not records:
        raise SystemExit("Aucun résultat : lancer step2_generate.py d'abord.")

    statuses = pd.Series([r["status"] for r in records]).value_counts()
    ok = [r for r in records if r["status"] == "ok"]
    df = pd.DataFrame([evaluate_record(r, s2["similarity_threshold"]) for r in ok])
    df.to_csv(out / f"{s2['run_name']}_par_patient.csv", index=False)

    n = len(records)
    summary: dict[str, Any] = {
        "run_name": s2["run_name"],
        "n_patients": n,
        "pct_json_valide": round(100 * statuses.get("ok", 0) / n, 1),
        "pct_sortie_invalide": round(100 * statuses.get("invalid_output", 0) / n, 1),
        "pct_depassement_contexte": round(100 * statuses.get("skipped_context_overflow", 0) / n, 1),
    }
    if not df.empty:
        summary.update({
            "diag_code_exact_pct": rate(df["diag_code_exact"]),
            "diag_code_3car_pct": rate(df["diag_code_3car"]),
            "diag_sim_moyenne": round(df["diag_sim"].dropna().mean(), 3),
            "acte_code_exact_pct": rate(df["acte_code_exact"]),
            "acte_sim_moyenne": round(df["acte_sim"].dropna().mean(), 3) if df["acte_sim"].notna().any() else None,
            "entree_sim_moyenne": round(df["entree_sim"].dropna().mean(), 3) if df["entree_sim"].notna().any() else None,
            "sortie_sim_moyenne": round(df["sortie_sim"].dropna().mean(), 3) if df["sortie_sim"].notna().any() else None,
            "duree_exacte_pct": rate(df["duree_exacte"]),
            "duree_plus_moins_1_pct": rate(df["duree_plus_moins_1"]),
            "pct_nombres_non_sources": round(100 * df["n_nombres_non_sources"].sum() / max(df["n_nombres"].sum(), 1), 1),
            "pct_codes_non_sources": round(100 * df["n_codes_non_sources"].sum() / max(df["n_codes"].sum(), 1), 1),
            "latence_moyenne_s": round(df["latence_s"].mean(), 2),
            "latence_p90_s": round(df["latence_s"].quantile(0.9), 2),
            "tokens_entree_moyen": round(df["tokens_entree"].mean(), 1),
            "tokens_sortie_moyen": round(df["tokens_sortie"].mean(), 1),
            "n_a_relire": int(df["a_relire"].sum()),
        })
    (out / f"{s2['run_name']}_resume.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
