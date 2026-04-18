"""Explicabilidad del modelo con SHAP.

Genera:
    outputs/figures/shap_summary.png  — importancia global (beeswarm)
    outputs/figures/shap_bar.png      — barra con top features
    outputs/shap_values.parquet       — valores SHAP por comercio (uso individual)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Permitir ejecución directa y como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PATHS
from model.train_model import split_features


def _transform(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pre = pipeline.named_steps["pre"]
    X_trans = pre.transform(X)
    feature_names = list(pre.get_feature_names_out())
    return X_trans, feature_names


def explain(mdt_path: str | Path | None = None,
            model_path: str | Path | None = None,
            fig_dir: str | Path | None = None,
            shap_out: str | Path | None = None,
            sample_size: int = 500) -> pd.DataFrame:
    mdt_path   = Path(mdt_path)   if mdt_path   else PATHS.MDT
    model_path = Path(model_path) if model_path else PATHS.MODEL_PKL
    fig_dir    = Path(fig_dir)    if fig_dir    else PATHS.FIGURES_DIR
    shap_out   = Path(shap_out)   if shap_out   else PATHS.SHAP_VALUES

    mdt      = pd.read_parquet(mdt_path)
    pipeline = joblib.load(model_path)

    X, _, _, _ = split_features(mdt)
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    X_trans, feature_names = _transform(pipeline, X_sample)
    clf = pipeline.named_steps["clf"]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)

    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names,
                       show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names,
                       plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index)
    shap_df["merchant_id"] = mdt.loc[X_sample.index, "merchant_id"].values
    shap_out.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_parquet(shap_out, index=False)

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    top = mean_abs.sort_values(ascending=False).head(15)
    print("\n=== Top 15 variables (|SHAP| medio) ===")
    print(top.round(4).to_string())

    return shap_df


if __name__ == "__main__":
    explain()
