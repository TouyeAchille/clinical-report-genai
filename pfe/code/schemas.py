"""Schémas Pydantic : sortie de l'étape 2 et données structurées de référence (étape 3)."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field


# Bornes de longueur : traduites en maxItems / maxLength dans le schéma JSON, que le
# décodage contraint de vLLM applique (à vérifier dans vllm.log). Elles empêchent les
# boucles de répétition et les recopies du texte source qui coupaient le JSON à 512 tokens.
SHORT = 120        # caractères pour un libellé
SYNTHESE = 700     # caractères pour la synthèse (~4 phrases)
Court = Annotated[str, Field(max_length=SHORT)]   # élément de liste borné


# ------------------------------------------------------------------ Étape 2
class ItemCode(BaseModel):
    code: str | None = Field(..., max_length=10,
                             description="Code CIM-10 ou CCAM uniquement s'il est écrit dans le texte, sinon null")
    description: str = Field(..., max_length=SHORT)


class ResumeSejour(BaseModel):
    """Champs alignés sur `structured_abstract` de PARHAF, plus une synthèse libre."""

    diagnostic_principal: list[ItemCode] = Field(..., max_length=3)
    acte_principal: list[ItemCode] = Field(..., max_length=3)
    mode_entree: str | None = Field(..., max_length=SHORT)
    mode_sortie: str | None = Field(..., max_length=SHORT)
    duree_sejour_jours: int | None
    antecedents: list[Court] = Field(..., max_length=8)
    traitements_sortie: list[Court] = Field(..., max_length=10)
    synthese: str = Field(..., max_length=SYNTHESE,
                          description="3 à 5 phrases, uniquement des faits présents dans les documents")


# ------------------------------------------------------------------ Étape 3
class ResultatBiologique(BaseModel):
    examen: str = Field(..., max_length=SHORT)
    valeur: str = Field(..., max_length=30, description="Valeur telle qu'écrite dans le texte")
    unite: str | None = Field(..., max_length=20)
    date: str | None = Field(..., max_length=30)


class Traitement(BaseModel):
    molecule: str = Field(..., max_length=SHORT)
    posologie: str | None = Field(..., max_length=SHORT)


class DonneesPatient(BaseModel):
    """Données structurées d'un séjour : future entrée du générateur de compte rendu (étape 4)."""

    age: str | None = Field(..., max_length=20)
    sexe: str | None = Field(..., max_length=10)
    motif: str | None = Field(..., max_length=200)
    antecedents: list[Court] = Field(..., max_length=10)
    allergies: list[Court] = Field(..., max_length=5)
    examen_clinique: list[Court] = Field(..., max_length=8)
    biologie: list[ResultatBiologique] = Field(..., max_length=12)
    imagerie: list[Court] = Field(..., max_length=5)
    diagnostics: list[Court] = Field(..., max_length=6)
    actes: list[Court] = Field(..., max_length=6)
    traitements_sortie: list[Traitement] = Field(..., max_length=10)
    suivi: list[Court] = Field(..., max_length=6)


SCALAR_FIELDS = {"age", "sexe", "motif"}


def empty_donnees() -> dict[str, Any]:
    """Gabarit vide, utilisé quand l'extraction automatique échoue (l'annotateur remplit à la main)."""
    return {name: (None if name in SCALAR_FIELDS else []) for name in DonneesPatient.model_fields}
