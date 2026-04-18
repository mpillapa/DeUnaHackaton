"""Entrenamiento del modelo de churn con XGBoost.

Pipeline de 3 splits estratificados (60/20/20):

    1. TRAIN (60%)   — ajusta los parámetros del árbol.
    2. VALIDATION    — early stopping + selección de umbral óptimo F1.
    3. TEST (20%)    — métricas finales no sesgadas (no se toca durante ajuste).

Produce:
    outputs/model/churn_model.pkl       — Pipeline preprocesador + XGBoost
    outputs/model/metrics.json          — Métricas por split + umbral
    outputs/model/feature_columns.json  — Columnas numéricas y categóricas
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Permitir ejecución directa y como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CATEGORICAL_COLS,
    ID_COLS,
    PATHS,
    SEED,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


# ════════════════════════════════════════════════════════════════════════════
# Métricas
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SplitMetrics:
    n: int
    churn_rate: float
    auc_roc: float
    auc_pr: float
    precision: float
    recall: float
    f1: float


def _metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    return SplitMetrics(
        n=int(len(y_true)),
        churn_rate=float(np.mean(y_true)),
        auc_roc=float(roc_auc_score(y_true, y_score)),
        auc_pr=float(average_precision_score(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def _pick_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Umbral que maximiza F1 sobre la curva precision–recall."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    if len(thr) == 0:
        return 0.5
    idx = int(np.nanargmax(f1[:-1]))
    return float(thr[idx])


# ════════════════════════════════════════════════════════════════════════════
# Features / preprocesamiento
# ════════════════════════════════════════════════════════════════════════════

def split_features(mdt: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Separa X, y y listas de columnas numéricas/categóricas.

    Se usa también desde predict.py y explain.py para mantener el mismo
    contrato de columnas.
    """
    drop = set(ID_COLS + [TARGET])
    # Excluir también metadatos que no deben llegar al modelo
    drop.update({"nombre_comercio"})

    feature_cols = [c for c in mdt.columns if c not in drop]
    X = mdt[feature_cols].copy()
    y = mdt[TARGET].astype(int) if TARGET in mdt.columns else None

    categorical = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric = [c for c in feature_cols if c not in categorical]

    X[categorical] = X[categorical].astype(str)
    return X, y, numeric, categorical


def _build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


# ════════════════════════════════════════════════════════════════════════════
# Entrenamiento
# ════════════════════════════════════════════════════════════════════════════

def train(mdt_path: str | Path | None = None,
          out_dir: str | Path | None = None) -> dict:
    mdt_path = Path(mdt_path) if mdt_path else PATHS.MDT
    out_dir = Path(out_dir) if out_dir else PATHS.MODEL_DIR

    mdt = pd.read_parquet(mdt_path)
    X, y, numeric, categorical = split_features(mdt)

    # ── Split 60/20/20 estratificado ────────────────────────────────────────
    # 1º corte: hold-out de test (20%)
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    # 2º corte: del 80% restante sacamos val (20% del total → 0.25 de este 80%)
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, stratify=y_rest, random_state=SEED,
    )

    # ── Preprocesamiento ────────────────────────────────────────────────────
    pre = _build_preprocessor(numeric, categorical)
    X_train_t = pre.fit_transform(X_train)
    X_val_t   = pre.transform(X_val)
    X_test_t  = pre.transform(X_test)

    # ── XGBoost con early stopping en VAL ───────────────────────────────────
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # Hiperparámetros conservadores para producir probabilidades granulares.
    # Learning rate bajo + muchos árboles + regularización evitan que el
    # modelo colapse a unos pocos splits triviales y nos da ranking útil.
    clf = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.80,
        colsample_bytree=0.70,
        min_child_weight=8,
        reg_lambda=3.0,
        gamma=0.2,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",      # robusto a desbalance
        tree_method="hist",
        early_stopping_rounds=60,
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    best_iter = int(clf.best_iteration)

    # ── Umbral óptimo F1 sobre VAL ──────────────────────────────────────────
    proba_val  = clf.predict_proba(X_val_t)[:, 1]
    threshold  = _pick_threshold(y_val.values, proba_val)

    # ── Evaluación por split ────────────────────────────────────────────────
    proba_train = clf.predict_proba(X_train_t)[:, 1]
    proba_test  = clf.predict_proba(X_test_t)[:, 1]

    train_m = _metrics(y_train.values, proba_train, (proba_train >= threshold).astype(int))
    val_m   = _metrics(y_val.values,   proba_val,   (proba_val   >= threshold).astype(int))
    test_m  = _metrics(y_test.values,  proba_test,  (proba_test  >= threshold).astype(int))

    # ── Persistir pipeline unificado ────────────────────────────────────────
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "churn_model.pkl")

    metrics = {
        "threshold": threshold,
        "best_iteration": best_iter,
        "scale_pos_weight": pos_weight,
        "splits": {
            "train": asdict(train_m),
            "val":   asdict(val_m),
            "test":  asdict(test_m),
        },
        "n_features_in": int(X_train_t.shape[1]),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump({"numeric": numeric, "categorical": categorical}, f, indent=2)

    # ── Log ─────────────────────────────────────────────────────────────────
    print("\n════════════  RESULTADOS DEL ENTRENAMIENTO  ════════════")
    print(f"Features tras one-hot : {X_train_t.shape[1]}")
    print(f"Iteraciones óptimas   : {best_iter} (early stop sobre VAL)")
    print(f"Umbral F1 óptimo      : {threshold:.4f}")
    print(f"scale_pos_weight      : {pos_weight:.2f}")
    print("\nMétricas por split (stratified):")
    header = f"{'split':<6}{'n':>6}{'churn%':>9}{'AUC-ROC':>10}{'AUC-PR':>10}{'prec':>8}{'rec':>8}{'F1':>8}"
    print(header)
    print("─" * len(header))
    for name, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        print(f"{name:<6}{m.n:>6}{m.churn_rate*100:>8.2f}%{m.auc_roc:>10.4f}"
              f"{m.auc_pr:>10.4f}{m.precision:>8.4f}{m.recall:>8.4f}{m.f1:>8.4f}")

    pred_test = (proba_test >= threshold).astype(int)
    print("\nMatriz de confusión (test):")
    print(confusion_matrix(y_test, pred_test))
    print("\nClassification report (test):")
    print(classification_report(y_test, pred_test, digits=4))

    return metrics


if __name__ == "__main__":
    train()
