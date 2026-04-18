"""
====================================================================
 MODELO DE CHURN PREDICTIVO — Reto Deuna (Interact2Hack 2026)
====================================================================

 Pipeline completo:
   1. Carga de datos y feature engineering
   2. Split train/test estratificado
   3. Entrenamiento de 4 modelos comparativos
      - Regresión Logística (baseline)
      - Random Forest
      - XGBoost
      - LightGBM
   4. Evaluación (AUC, precision, recall, F1, matriz confusión)
   5. Tuning de hiperparámetros para XGBoost (ganador)
   6. Análisis SHAP del modelo final
   7. Generación de fact_churn_predictions.csv (Tabla 3)
   8. Visualizaciones para el pitch

 Requisitos previos (pip install):
   pandas, numpy, scikit-learn, xgboost, lightgbm, shap,
   matplotlib, seaborn

 Uso:
   python entrenar_modelo.py

 Reproducibilidad: semilla fija = 42
====================================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Sklearn
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                       RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, precision_recall_curve)

# Gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Explicabilidad
import shap

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.family"] = "DejaVu Sans"

# Feature engineering propio
from feature_engineering import construir_dataset_features

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
SEED = 42
np.random.seed(SEED)

PATH_MERCHANTS = "data/raw/dim_merchants_con_abandono.csv"
PATH_PERFORMANCE = "data/raw/fact_performance_monthly.csv"
PATH_TICKETS = "data/raw/fact_support_tickets.csv"

OUTPUT_DIR = "outputs_modelo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELO_VERSION = "xgb_v1.0"
FECHA_SNAPSHOT = datetime(2026, 4, 1)   # fecha de la ejecución (para Tabla 3)

# ⚠️ FECHA DE CORTE — CRÍTICO PARA EVITAR LEAKAGE TEMPORAL
# ---------------------------------------------------------------
# El dataset cubre abril 2025 → marzo 2026. Los churners muestran
# decaimiento en los últimos 3-5 meses. Si usamos fecha_corte =
# marzo 2026 como límite de features, el modelo logra AUC=1.0
# "haciendo trampa" (ve el decay completo).
#
# Simulamos estar en un momento del pasado y predecir el futuro:
# el modelo usa datos hasta FECHA_CORTE y debe predecir abandono
# en ~30 días después.
#
# Valores recomendados y AUC esperado:
#   - 2025-11-30 (recomendado): AUC ~0.88-0.93
#   - 2025-12-31 (permisivo):   AUC ~0.95+
#   - 2025-09-30 (estricto):    AUC ~0.80-0.85
# ---------------------------------------------------------------
FECHA_CORTE = datetime(2025, 11, 30)

# ============================================================
# 1. CARGA DE DATOS Y FEATURE ENGINEERING
# ============================================================
print("=" * 70)
print("PASO 1: FEATURE ENGINEERING")
print("=" * 70)
print(f"FECHA_CORTE: {FECHA_CORTE.date()} "
      f"(features usan datos <= esta fecha; target es futuro)")

X, y, df_full = construir_dataset_features(
    PATH_MERCHANTS, PATH_PERFORMANCE, PATH_TICKETS,
    fecha_corte=FECHA_CORTE,
)

merchant_ids = df_full["merchant_id"].values
feature_names = X.columns.tolist()

# ============================================================
# 2. SPLIT TRAIN/TEST ESTRATIFICADO
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: SPLIT TRAIN/TEST ESTRATIFICADO (80/20)")
print("=" * 70)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(X)),
    test_size=0.20, stratify=y, random_state=SEED
)

merchant_ids_train = merchant_ids[idx_train]
merchant_ids_test = merchant_ids[idx_test]

print(f"Train: {len(X_train)} comercios ({y_train.mean()*100:.1f}% churners)")
print(f"Test:  {len(X_test)} comercios ({y_test.mean()*100:.1f}% churners)")

# Calcular scale_pos_weight para manejar desbalance
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# Escalado (solo para LogReg; los modelos basados en árboles no lo necesitan)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ============================================================
# 3. ENTRENAR 4 MODELOS COMPARATIVOS
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: ENTRENANDO 4 MODELOS COMPARATIVOS")
print("=" * 70)

modelos = {}
predicciones = {}
probabilidades = {}

# --- 3.1 Regresión Logística (baseline) ---
print("\n[1/4] Regresión Logística (baseline)...")
lr = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=SEED, C=1.0
)
lr.fit(X_train_sc, y_train)
modelos["LogReg"] = lr
probabilidades["LogReg"] = lr.predict_proba(X_test_sc)[:, 1]
predicciones["LogReg"] = lr.predict(X_test_sc)

# --- 3.2 Random Forest ---
print("[2/4] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=10,
    class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
modelos["RandomForest"] = rf
probabilidades["RandomForest"] = rf.predict_proba(X_test)[:, 1]
predicciones["RandomForest"] = rf.predict(X_test)

# --- 3.3 XGBoost (candidato principal) ---
print("[3/4] XGBoost (sin tuning)...")
xgb_base = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=SEED, n_jobs=-1
)
xgb_base.fit(X_train, y_train)
modelos["XGBoost"] = xgb_base
probabilidades["XGBoost"] = xgb_base.predict_proba(X_test)[:, 1]
predicciones["XGBoost"] = xgb_base.predict(X_test)

# --- 3.4 LightGBM ---
print("[4/4] LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1
)
lgbm.fit(X_train, y_train)
modelos["LightGBM"] = lgbm
probabilidades["LightGBM"] = lgbm.predict_proba(X_test)[:, 1]
predicciones["LightGBM"] = lgbm.predict(X_test)

# ============================================================
# 4. EVALUACIÓN COMPARATIVA
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: EVALUACIÓN COMPARATIVA")
print("=" * 70)

resultados = []
for nombre in modelos.keys():
    y_pred = predicciones[nombre]
    y_proba = probabilidades[nombre]
    resultados.append({
        "Modelo":    nombre,
        "AUC":       roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
    })
df_resultados = pd.DataFrame(resultados).sort_values("AUC", ascending=False)
df_resultados["AUC"] = df_resultados["AUC"].round(4)
df_resultados["Precision"] = df_resultados["Precision"].round(4)
df_resultados["Recall"] = df_resultados["Recall"].round(4)
df_resultados["F1"] = df_resultados["F1"].round(4)

print("\nResultados en test set:")
print(df_resultados.to_string(index=False))

# Guardar tabla comparativa
df_resultados.to_csv(f"{OUTPUT_DIR}/comparacion_modelos.csv", index=False)
print(f"\n✓ Guardado: {OUTPUT_DIR}/comparacion_modelos.csv")

# Verificar umbral del reto
mejor_auc = df_resultados["AUC"].max()
print(f"\nMejor AUC: {mejor_auc:.4f}")
if mejor_auc > 0.75:
    print("✓ Supera el umbral orientativo del reto (AUC > 0.75)")
else:
    print("⚠️ No supera el umbral del reto")

# ============================================================
# 5. TUNING DE HIPERPARÁMETROS DE XGBOOST
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: TUNING DE XGBOOST (RandomizedSearchCV)")
print("=" * 70)

param_grid = {
    "n_estimators":     [200, 300, 500, 700],
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0, 0.1, 0.3, 0.5],
    "reg_alpha":        [0, 0.01, 0.1, 1],
    "reg_lambda":       [1, 1.5, 2],
}

xgb_search = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=SEED, n_jobs=-1,
)

print("Ejecutando RandomizedSearchCV (30 iteraciones, 5-fold CV)...")
search = RandomizedSearchCV(
    estimator=xgb_search,
    param_distributions=param_grid,
    n_iter=30, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="roc_auc",
    n_jobs=-1, random_state=SEED, verbose=0,
)
search.fit(X_train, y_train)

print(f"\n✓ Mejores parámetros encontrados:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"✓ Mejor AUC en CV: {search.best_score_:.4f}")

# Modelo final con hiperparámetros óptimos
xgb_final = search.best_estimator_
y_pred_final = xgb_final.predict(X_test)
y_proba_final = xgb_final.predict_proba(X_test)[:, 1]

print(f"\nEvaluación del modelo tuneado en test:")
print(f"  AUC:       {roc_auc_score(y_test, y_proba_final):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_final):.4f}")

print(f"\nClassification report:")
print(classification_report(y_test, y_pred_final,
                             target_names=["Activo (0)", "Churn (1)"]))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_final)
print(f"\nMatriz de confusión:")
print(f"                Predicción")
print(f"                Activo | Churn")
print(f"  Real Activo:  {cm[0,0]:>6} | {cm[0,1]:>5}")
print(f"  Real Churn:   {cm[1,0]:>6} | {cm[1,1]:>5}")

# ============================================================
# 6. ANÁLISIS SHAP
# ============================================================
print("\n" + "=" * 70)
print("PASO 6: ANÁLISIS SHAP")
print("=" * 70)

print("Calculando valores SHAP...")
explainer = shap.TreeExplainer(xgb_final)
shap_values_test = explainer.shap_values(X_test)
shap_values_all = explainer.shap_values(X)   # para todos los comercios (Tabla 3)

# Top features globales
mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
importancia_global = pd.DataFrame({
    "feature": feature_names,
    "importancia_shap": mean_abs_shap
}).sort_values("importancia_shap", ascending=False)

print(f"\nTop 10 features más influyentes (SHAP global):")
print(importancia_global.head(10).to_string(index=False))
importancia_global.to_csv(f"{OUTPUT_DIR}/importancia_features.csv", index=False)

# ============================================================
# 7. GENERACIÓN DE TABLA 3 (fact_churn_predictions)
# ============================================================
print("\n" + "=" * 70)
print("PASO 7: GENERACIÓN DE TABLA 3 — fact_churn_predictions")
print("=" * 70)

# Probabilidades para TODOS los comercios (no solo test)
probas_all = xgb_final.predict_proba(X)[:, 1]

# --- Clasificación en niveles de riesgo ---
def _nivel_riesgo(p):
    if p >= 0.80: return "Crítico"
    if p >= 0.60: return "Alto"
    if p >= 0.35: return "Medio"
    return "Bajo"

niveles = [_nivel_riesgo(p) for p in probas_all]

# --- Top-3 drivers SHAP por comercio ---
# Usamos los valores SHAP absolutos para ranking
abs_shap = np.abs(shap_values_all)
top3_idx = np.argsort(-abs_shap, axis=1)[:, :3]   # índices de los top 3 por fila

drivers = []
for i, mid in enumerate(merchant_ids):
    top_i = top3_idx[i]
    drivers.append({
        "driver_1_nombre": feature_names[top_i[0]],
        "driver_1_shap":   round(float(shap_values_all[i, top_i[0]]), 4),
        "driver_2_nombre": feature_names[top_i[1]],
        "driver_2_shap":   round(float(shap_values_all[i, top_i[1]]), 4),
        "driver_3_nombre": feature_names[top_i[2]],
        "driver_3_shap":   round(float(shap_values_all[i, top_i[2]]), 4),
    })
df_drivers = pd.DataFrame(drivers)

# --- Next Best Action basada en driver principal ---
def _nba_desde_driver(driver: str, shap_val: float, nivel: str) -> str:
    """Mapea el driver principal a una acción comercial concreta."""
    if nivel == "Bajo":
        return "Monitoreo pasivo"

    # Caídas de actividad
    if "pendiente_count_trx" in driver or "pendiente_tpv" in driver:
        return "Llamada KAM urgente + diagnóstico comercial"
    if "ratio_tpv_3m" in driver or "ratio_count_3m" in driver:
        return "Llamada KAM + análisis de competencia"
    if "max_caida_mensual" in driver:
        return "Llamada KAM + revisar eventos del mes de caída"

    # Recencia
    if "meses_desde_ultimo" in driver or "dias_desde_ultima" in driver:
        return "Campaña de reactivación + incentivo monetario"

    # Soporte
    if "tickets_no_resueltos" in driver or "tickets_30d" in driver or "tickets_90d" in driver:
        return "Soporte técnico prioritario + revisión de SLAs"
    if "severidad" in driver:
        return "Escalamiento a soporte técnico + visita presencial"
    if "tiempo_resolucion" in driver:
        return "Revisión de casos abiertos con coordinador"
    if "ratio_pago_rechazado" in driver:
        return "Revisión técnica de pasarela + test de dispositivo"
    if "ratio_liquidacion_demora" in driver:
        return "Escalamiento al equipo de liquidación/tesorería"

    # Tenure
    if "tenure" in driver or "onboarding" in driver:
        return "Programa de onboarding extendido + capacitación"

    # Rechazo
    if "tasa_rechazo" in driver:
        return "Diagnóstico técnico + ajuste de configuración"

    # Default por nivel
    if nivel == "Crítico":
        return "Contacto inmediato + plan de retención personalizado"
    if nivel == "Alto":
        return "Llamada KAM en próximas 48h"
    return "Seguimiento comercial"

# --- CLV proyectado ---
# Estimado simple: TPV últimos 90d anualizado × margen estimado (1%)
# Si el comercio está en riesgo alto, el CLV se ajusta por probabilidad de retención
MARGEN_SOBRE_TPV = 0.01

# Mapa merchant_id → region (desde dim_merchants)
df_merchants_raw = pd.read_csv(PATH_MERCHANTS, usecols=["merchant_id", "region"])
region_map = dict(zip(df_merchants_raw["merchant_id"], df_merchants_raw["region"]))

df_predicciones = pd.DataFrame({
    "merchant_id":        merchant_ids,
    "fecha_snapshot":     FECHA_SNAPSHOT.date(),
    "modelo_version":     MODELO_VERSION,
    "probabilidad_churn": probas_all.round(4),
    "nivel_riesgo":       niveles,
    "region":             [region_map.get(mid, "") for mid in merchant_ids],
})

# Agregar drivers
df_predicciones = pd.concat([df_predicciones, df_drivers], axis=1)

# CLV: TPV 90d × 4 (para anualizarlo) × margen, ajustado por (1 - prob_churn)
tpv_90d_map = dict(zip(merchant_ids, X["tpv_3m"].values))
df_predicciones["clv_proyectado"] = df_predicciones.apply(
    lambda row: round(tpv_90d_map.get(row["merchant_id"], 0) * 4 * MARGEN_SOBRE_TPV
                      * (1 - row["probabilidad_churn"]), 2),
    axis=1
)

# NBA
df_predicciones["nba_sugerida"] = df_predicciones.apply(
    lambda row: _nba_desde_driver(row["driver_1_nombre"],
                                    row["driver_1_shap"],
                                    row["nivel_riesgo"]),
    axis=1
)

# Score de salud (0-100 inverso al riesgo)
df_predicciones["score_salud"] = ((1 - df_predicciones["probabilidad_churn"]) * 100).round(1)

# Guardar Tabla 3
df_predicciones.to_csv(f"{OUTPUT_DIR}/fact_churn_predictions.csv",
                       index=False, encoding="utf-8")
print(f"\n✓ Guardado: {OUTPUT_DIR}/fact_churn_predictions.csv")
print(f"   {len(df_predicciones)} predicciones generadas")

# Resumen por nivel de riesgo
print("\nDistribución de comercios por nivel de riesgo:")
print(df_predicciones["nivel_riesgo"].value_counts().to_string())

print("\nMuestra de la Tabla 3 (5 comercios en riesgo Crítico):")
criticos = df_predicciones[df_predicciones["nivel_riesgo"] == "Crítico"].head(5)
cols_show = ["merchant_id", "probabilidad_churn", "nivel_riesgo",
             "driver_1_nombre", "nba_sugerida", "clv_proyectado"]
print(criticos[cols_show].to_string(index=False))

# ============================================================
# 8. VISUALIZACIONES PARA EL PITCH
# ============================================================
print("\n" + "=" * 70)
print("PASO 8: VISUALIZACIONES")
print("=" * 70)

# --- 8.1 Comparación de modelos (barras AUC) ---
fig, ax = plt.subplots(figsize=(10, 5))
df_plot = df_resultados.copy()
colors = ['#00D4A6' if m == 'XGBoost' else '#9E9AC8' for m in df_plot["Modelo"]]
ax.barh(df_plot["Modelo"], df_plot["AUC"], color=colors, edgecolor='#2D1B5E')
ax.set_xlabel("AUC-ROC", fontsize=12)
ax.set_title("Comparación de modelos — AUC en test set", fontsize=14, color="#2D1B5E")
ax.axvline(0.75, color='red', linestyle='--', label='Umbral del reto (0.75)')
for i, v in enumerate(df_plot["AUC"]):
    ax.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=11)
ax.set_xlim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_1_comparacion_modelos.png", bbox_inches='tight')
plt.close()

# --- 8.2 Matriz de confusión ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Activo", "Churn"],
            yticklabels=["Activo", "Churn"], ax=ax, cbar=False,
            annot_kws={"size": 16})
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Realidad", fontsize=12)
ax.set_title("Matriz de confusión — XGBoost final", fontsize=14, color="#2D1B5E")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_2_matriz_confusion.png", bbox_inches='tight')
plt.close()

# --- 8.3 Curva ROC ---
fig, ax = plt.subplots(figsize=(7, 6))
for nombre, proba in probabilidades.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})", linewidth=2)

# Modelo tuneado
fpr_f, tpr_f, _ = roc_curve(y_test, y_proba_final)
auc_f = roc_auc_score(y_test, y_proba_final)
ax.plot(fpr_f, tpr_f, label=f"XGBoost Tuned (AUC={auc_f:.3f})",
        linewidth=3, color="#00D4A6")

ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Curvas ROC — todos los modelos", fontsize=14, color="#2D1B5E")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_3_curvas_roc.png", bbox_inches='tight')
plt.close()

# --- 8.4 SHAP summary plot ---
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values_test, X_test, feature_names=feature_names,
                   show=False, max_display=15)
plt.title("SHAP summary — impacto de features sobre la predicción de churn",
          fontsize=13, color="#2D1B5E")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_4_shap_summary.png", bbox_inches='tight')
plt.close()

# --- 8.5 SHAP bar plot (top features globales) ---
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values_test, X_test, feature_names=feature_names,
                   plot_type="bar", show=False, max_display=15)
plt.title("Importancia global de features (SHAP)",
          fontsize=13, color="#2D1B5E")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_5_shap_bar.png", bbox_inches='tight')
plt.close()

# --- 8.6 Distribución de probabilidades por clase ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_proba_final[y_test == 0], bins=30, alpha=0.6,
        label="Activos (test)", color="#9E9AC8", edgecolor='white')
ax.hist(y_proba_final[y_test == 1], bins=30, alpha=0.8,
        label="Churners (test)", color="#E74C3C", edgecolor='white')
ax.set_xlabel("Probabilidad predicha de churn", fontsize=12)
ax.set_ylabel("Cantidad de comercios", fontsize=12)
ax.set_title("Separación de probabilidades entre clases",
             fontsize=14, color="#2D1B5E")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_6_distribucion_probas.png", bbox_inches='tight')
plt.close()

# --- 8.7 Waterfall plot para un caso individual (mejor churner predicho) ---
idx_mejor_churner = np.argsort(-y_proba_final)[0]  # mayor probabilidad en test
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_test[idx_mejor_churner],
        base_values=explainer.expected_value,
        data=X_test.iloc[idx_mejor_churner].values,
        feature_names=feature_names
    ),
    max_display=10, show=False
)
plt.title(f"Explicación individual — comercio con mayor probabilidad de churn\n"
          f"(prob = {y_proba_final[idx_mejor_churner]:.3f})",
          fontsize=12, color="#2D1B5E")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/viz_7_waterfall_churner.png", bbox_inches='tight')
plt.close()

print(f"✓ 7 visualizaciones guardadas en {OUTPUT_DIR}/")

# ============================================================
# 9. RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

print(f"""
✓ Feature engineering: {X.shape[1]} features construidas para {X.shape[0]} comercios
✓ Comparación de 4 modelos completada
✓ Tuning de XGBoost con 30 iteraciones × 5-fold CV
✓ AUC final en test: {roc_auc_score(y_test, y_proba_final):.4f}
✓ SHAP analysis completo
✓ Tabla 3 generada: {len(df_predicciones)} predicciones

Archivos generados en {OUTPUT_DIR}/:
  - comparacion_modelos.csv      Ranking de los 4 modelos
  - importancia_features.csv     Top features según SHAP
  - fact_churn_predictions.csv   Tabla 3 completa
  - viz_1_comparacion_modelos.png
  - viz_2_matriz_confusion.png
  - viz_3_curvas_roc.png
  - viz_4_shap_summary.png
  - viz_5_shap_bar.png
  - viz_6_distribucion_probas.png
  - viz_7_waterfall_churner.png

Próximos pasos:
  1. Revisar la Tabla 3 y validar drivers + NBA con el equipo de negocio
  2. Construir dashboard interactivo usando las 4 tablas finales
  3. Preparar pitch de 5 minutos destacando las visualizaciones generadas
""")