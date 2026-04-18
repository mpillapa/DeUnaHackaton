"""Scoring y segmentación de churn para el equipo comercial.

Genera `outputs/predictions.csv` con:
    merchant_id, fecha_corte, probabilidad_churn, prob_rank,
    segmento_churn (ALERTA_ROJA / AMARILLA / BAJA / MUY_BAJA),
    segmento_comercial, region, tipo_negocio_desc, tenure_meses,
    tpv_sum_6m, tx_sum_6m, recencia_bucket_0, ejecutivo_cuenta.

La segmentación sigue los percentiles del flujo Databricks de referencia
(95 / 89 / 82), haciéndola independiente del volumen absoluto de la cartera.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

# Permitir ejecución directa y como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PATHS
from model.train_model import split_features


SEGMENTOS = [
    (0.95, "ALERTA_ROJA"),
    (0.89, "ALERTA_AMARILLA"),
    (0.82, "BAJA_PROBABILIDAD"),
    (0.00, "MUY_BAJA_PROBABILIDAD"),
]


def _segmentar(prob_rank: float) -> str:
    for cutoff, label in SEGMENTOS:
        if prob_rank >= cutoff:
            return label
    return "MUY_BAJA_PROBABILIDAD"


def score(mdt_path: str | Path | None = None,
          model_path: str | Path | None = None,
          merchants_path: str | Path | None = None,
          out_path: str | Path | None = None) -> pd.DataFrame:
    mdt_path       = Path(mdt_path)       if mdt_path       else PATHS.MDT
    model_path     = Path(model_path)     if model_path     else PATHS.MODEL_PKL
    merchants_path = Path(merchants_path) if merchants_path else PATHS.DIM_MERCHANTS
    out_path       = Path(out_path)       if out_path       else PATHS.PREDICTIONS

    mdt      = pd.read_parquet(mdt_path)
    pipeline = joblib.load(model_path)
    merchants = pd.read_csv(merchants_path)

    X, _, _, _ = split_features(mdt)
    proba = pipeline.predict_proba(X)[:, 1]

    out = mdt[["merchant_id", "fecha_corte"]].copy()
    out["probabilidad_churn"] = proba
    out["prob_rank"]          = out["probabilidad_churn"].rank(pct=True)
    out["segmento_churn"]     = out["prob_rank"].apply(_segmentar)

    # Enriquecer con variables de negocio para el dashboard
    out = out.merge(
        mdt[["merchant_id", "segmento_comercial", "region", "tipo_negocio_desc",
             "tenure_meses", "tpv_sum_6m", "tx_sum_6m", "recencia_bucket_0"]],
        on="merchant_id",
        how="left",
    )
    out = out.merge(
        merchants[["merchant_id", "nombre_comercio", "ciudad", "ejecutivo_cuenta"]],
        on="merchant_id",
        how="left",
    )

    out = out.sort_values("probabilidad_churn", ascending=False).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nPredicciones guardadas en: {out_path}")
    print("\nDistribución por segmento:")
    print(out["segmento_churn"].value_counts().to_string())
    print("\nTop 10 comercios en riesgo:")
    print(
        out.head(10)[
            ["merchant_id", "nombre_comercio", "probabilidad_churn",
             "segmento_churn", "segmento_comercial", "region", "tpv_sum_6m"]
        ].to_string(index=False)
    )
    return out


if __name__ == "__main__":
    score()
