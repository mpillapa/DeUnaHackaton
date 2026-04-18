# Modelo de Churn — Documento Técnico

**Reto 3 · Datos · Interact2Hack 2026 — Deuna**
**Versión:** 1.0 · **Fecha:** 2026-04-18

---

## 1. Propósito

Asignar a cada comercio de Deuna una **probabilidad de abandono en los próximos 30 días** y clasificarlo en un nivel de alerta accionable por el equipo comercial (Roja / Amarilla / Baja / Muy Baja).

El modelo se entrena sobre data sintética que simula 12 meses de historia (2025-04 a 2026-03) para 2 000 comercios con tasa de abandono realista (~13 %).

---

## 2. Insumos y salidas

### Insumos (`data/raw/`)
| Archivo | Contenido | Uso |
|---|---|---|
| `dim_merchants.csv` | Dimensión estática: segmento, región, tipo de negocio, fecha de onboarding, ejecutivo de cuenta | Features estáticos |
| `fact_performance_monthly.csv` | Serie mensual × 12 meses: TPV, trx, tasa de rechazo, días sin operación, tickets | Features temporales (lags + ventanas) |
| `fact_support_tickets.csv` | Detalle individual de tickets: categoría, severidad, estado, tiempo de resolución, satisfacción | Features de experiencia de soporte |
| `churn_labels.csv` | **Solo para entrenar.** `merchant_id`, `abandono_30d` (ground truth) | Target supervisado |

> En producción `churn_labels.csv` se reconstruye desde el warehouse (comercios inactivos observados en los últimos 30 días).

### Salidas (`outputs/`)
| Archivo | Contenido |
|---|---|
| `model/churn_model.pkl` | Pipeline sklearn (ColumnTransformer + XGBoost) |
| `model/metrics.json` | Métricas por split y umbral F1 óptimo |
| `model/feature_columns.json` | Columnas numéricas y categóricas usadas |
| `predictions.csv` | Scoring completo: `merchant_id`, `probabilidad_churn`, `segmento_churn`, variables de negocio |
| `shap_values.parquet` | Valores SHAP por comercio (explicabilidad individual) |
| `figures/shap_summary.png`, `shap_bar.png` | Importancia global de variables |

---

## 3. MDT — Master Data Table

Se construye en [model/feature_engineering.py](../model/feature_engineering.py) una fila por comercio a la **fecha de corte** (2026-01-01, ver §5). Resultado: **2 000 × 131 columnas**.

### 3.1 Bloques de features

| Bloque | Fuente | Ejemplos |
|---|---|---|
| **Estáticos** | `dim_merchants` | `segmento_comercial`, `region`, `tipo_negocio_desc`, `tenure_meses`, `es_auto_gestionado` |
| **Lags 0..4** | `fact_performance_monthly` | `count_trx_{0..4}`, `tpv_mensual_{0..4}`, `tasa_rechazo_{0..4}`, `dias_sin_transaccion_max_{0..4}` |
| **Ventanas 3/6/12 m** | `fact_performance_monthly` | `tx_sum_6m`, `tpv_mean_12m`, `tx_std_6m`, `meses_activos_6m`, `rechazo_mean_6m` |
| **Deltas mes-a-mes** | Derivado | `delta_count_trx_0_1`, `delta_tpv_mensual_1_2`, `delta_tasa_rechazo_0_1` |
| **Ratios** | Derivado | `volatilidad_tx_6m`, `volatilidad_tpv_6m`, `tickets_per_tx_6m`, `tasa_no_resuelto_6m` |
| **Tickets 6 m** | `fact_support_tickets` | `tickets_total_6m`, `tickets_no_resueltos_6m`, `severidad_max_6m`, `tickets_pago_rechazado_6m`, `tickets_liquidacion_demora_6m` |
| **Categórica derivada** | Derivado | `recencia_bucket_0` (sin_registro / menos_5 / 5_10 / 10_20 / 20_30 / mas_30) |

### 3.2 Categóricas
Cuatro columnas entran one-hot: `segmento_comercial`, `region`, `tipo_negocio_desc`, `recencia_bucket_0`. Total features tras one-hot encoding: **171**.

---

## 4. Target y regla de negocio

- **Columna:** `abandono_30d` (binario)
- **Regla de negocio:** comercio que no registra transacciones durante los 30 días posteriores a la fecha de corte.
- **Tasa observada:** 13 % (260 de 2 000 en la data sintética)
- **Fuente:** `churn_labels.csv`. En producción se reconstruye mensualmente desde el warehouse de transacciones.

---

## 5. Fecha de corte — decisión clave

La data sintética cubre 12 meses (2025-04 → 2026-03). Los churners siguen un patrón de **decaimiento de 3 a 5 meses** antes de abandonar. Esto implica:

| Fecha de corte | Problema |
|---|---|
| 2026-03-01 (último mes) | Los churners ya están prácticamente inactivos — señal trivial, AUC = 1.00 |
| **2026-01-01 (elegida)** | El decaimiento recién empieza — el modelo debe detectar señal temprana |
| 2025-10-01 (mes 6) | Aún no ha empezado el decaimiento — el modelo no tiene señal suficiente |

**Decisión:** `FECHA_CORTE = 2026-01-01` en [config/settings.py](../config/settings.py). El modelo dispone de:
- Historia completa hasta 2026-01 (lags 0..4 = meses 2025-09..2026-01)
- Ventana de 6 meses de tickets: 2025-08 a 2026-01
- Target: abandono observado en los 2-3 meses siguientes

Esto replica el patrón real: **anticipar el abandono 30-60 días antes** de que el comercio quede inactivo.

---

## 6. Split train / val / test — tres espacios

Separamos los 2 000 comercios **una sola vez** en tres grupos estratificados por `abandono_30d` (13 % churners en cada split). Los mismos splits se usan en todos los experimentos (`SEED=42`) para asegurar comparabilidad.

| Split | % | N | Churners | Rol |
|---|---|---|---|---|
| **Train** | 60 % | 1 200 | 156 | Ajusta los pesos del modelo — XGBoost construye árboles sobre estos datos. |
| **Validation** | 20 % | 400 | 52 | **Early stopping** (el árbol se detiene cuando AUC-PR deja de mejorar 60 iteraciones) + **selección del umbral F1 óptimo** sobre la curva precision-recall. |
| **Test** | 20 % | 400 | 52 | **Hold-out puro.** Se evalúa UNA SOLA VEZ al final, nunca se usa para tunear nada. Reporta métricas no sesgadas. |

El orden es estricto:

```
                ┌──────────────────┐
  train (60%) ──►   fit trees      │
                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
     val (20%) ──►  early stop     │
                │  threshold F1    │
                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
    test (20%) ──►  métricas finales (auditoría)
                └──────────────────┘
```

Código: [model/train_model.py:83-104](../model/train_model.py#L83-L104)

### Espacio de predicción (producción)

Una vez entrenado, `predict.py` aplica el pipeline a **los 2 000 comercios** — incluidos los de train/val/test — y escribe `outputs/predictions.csv`. En producción este paso recibe el universo completo de comercios activos al cierre de mes.

---

## 7. Algoritmo

**XGBoost** (`XGBClassifier`) por ser estado del arte en datos tabulares B2B y por la interpretabilidad SHAP.

### 7.1 Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 800 | Con early stopping decide cuántos usar realmente |
| `max_depth` | 4 | Árboles superficiales — evita memorizar combinaciones raras |
| `learning_rate` | 0.02 | Bajo para que cada árbol aporte poco → probabilidades más suaves |
| `subsample` | 0.80 | Cada árbol ve 80 % de filas |
| `colsample_bytree` | 0.70 | Cada árbol ve 70 % de features |
| `min_child_weight` | 8 | Requiere ≥ 8 observaciones para hacer un split |
| `reg_lambda` | 3.0 | Regularización L2 sobre pesos de hojas |
| `gamma` | 0.2 | Mejora mínima para permitir un split |
| `scale_pos_weight` | 6.69 (calc) | Igual a `n_neg/n_pos` — compensa desbalance 13/87 |
| `eval_metric` | `aucpr` | Más robusta que AUC-ROC para clases desbalanceadas |
| `early_stopping_rounds` | 60 | Paciencia generosa para que encuentre el óptimo sobre VAL |
| `tree_method` | `hist` | Implementación eficiente para features continuos |
| `random_state` | 42 | Reproducible |

### 7.2 Preprocesamiento

- **Numéricos (167 cols):** passthrough (XGBoost no requiere escalado)
- **Categóricos (4 cols):** One-hot encoding con `handle_unknown="ignore"` (tolera categorías nuevas en predicción)
- Todo envuelto en `sklearn.Pipeline` para evitar leakage del preprocesador al val/test

Código: [model/train_model.py:140-175](../model/train_model.py#L140-L175)

---

## 8. Validación y métricas

### 8.1 Métricas reportadas en cada split

| Métrica | Qué mide | Por qué importa |
|---|---|---|
| **AUC-ROC** | Capacidad de ranking (¿la prob de un churner es mayor que la de un sano al azar?) | Métrica estándar, pero optimista con clases desbalanceadas |
| **AUC-PR** | Área bajo la curva precision-recall | Mejor métrica con 13 % positivos — penaliza falsos positivos fuertemente |
| **Precision** | De los que marco como churn, ¿cuántos lo son? | Evita molestar comercios sanos |
| **Recall** | De los que son churn, ¿cuántos detecto? | Evita perder comercios realmente en riesgo |
| **F1** | Media armónica precision/recall | Métrica de referencia para elegir el umbral |

### 8.2 Criterio de éxito

Orientativo del reto: **AUC > 0.75**. Nuestro resultado en la data sintética:

| Split | N | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Train | 1 200 | 0.9997 | 0.9967 | 1.0000 | 0.9231 | 0.9600 |
| Val | 400 | 0.9997 | 0.9960 | 1.0000 | 0.9615 | 0.9804 |
| **Test** | **400** | **0.9991** | **0.9917** | **1.0000** | **0.9231** | **0.9600** |

**Gap train-test < 0.001 en AUC → sin sobreajuste.** El modelo generaliza correctamente.

> ⚠️ **Nota sobre magnitud.** Las métricas son inusualmente altas porque la data sintética tiene un patrón de decaimiento muy determinístico (los churners tienen ≥ 15 % de tasa de rechazo y caen 40-70 % en transacciones por mes). En producción con data real esperar AUC en el rango **0.80-0.90**. El objetivo de este entrenamiento es validar que el pipeline end-to-end funciona, no presumir el número.

### 8.3 Selección del umbral

Tras el fit, se barre la curva precision-recall sobre el **set de validación** y se elige el umbral que maximiza F1. Este umbral se guarda en `metrics.json` y se usa tanto para reportar métricas en test como para segmentar comercios en producción.

Umbral final: **0.516**.

---

## 9. Segmentación comercial

Sobre `probabilidad_churn` calculamos el **percentil dentro de la cartera** (`prob_rank`) y asignamos nivel de alerta siguiendo la lógica del flujo Databricks de referencia:

| Segmento | Umbral percentil | Acción comercial |
|---|---|---|
| 🔴 `ALERTA_ROJA` | ≥ 95 | Llamada KAM en < 48 h |
| 🟡 `ALERTA_AMARILLA` | 89 – 95 | Email personalizado + oferta |
| 🟢 `BAJA_PROBABILIDAD` | 82 – 89 | Monitoreo, sin acción directa |
| ⚪ `MUY_BAJA_PROBABILIDAD` | < 82 | No intervenir |

Usar percentiles hace la priorización **independiente del volumen absoluto** de la cartera: si la cartera crece 2×, los cutoffs absolutos se mueven pero el tamaño de cada segmento permanece constante.

---

## 10. Supuestos y limitaciones

1. **Data sintética.** El patrón de decaimiento es más limpio que en producción. El modelo funcionará con métricas más modestas sobre data real.
2. **Cross-sectional, no time-series.** Una sola fecha de corte — no capturamos estacionalidad de la misma semana en distintos años.
3. **Probabilidades no calibradas.** `probabilidad_churn` es un score, no una probabilidad absoluta. Para decisiones de ROI usar `prob_rank` o aplicar `CalibratedClassifierCV` (isotonic) como paso posterior.
4. **Universo cerrado de features.** El modelo asume que las 4 categóricas tienen los mismos valores en train y producción (nuevas regiones / CIIU requieren re-entrenamiento).
5. **Sin análisis de supervivencia.** Predecimos "abandono en ventana fija" no "tiempo hasta abandono". Para la V2 migrar a Cox / XGBoost-AFT.

---

## 11. Plan de recalibración

| Cadencia | Disparador | Acción |
|---|---|---|
| Mensual | Cierre de mes en warehouse | Regenerar `churn_labels.csv` con los nuevos 30 días observados + re-scoring completo |
| Trimestral | Fin de trimestre | Re-entrenar desde cero con la ventana móvil de 12 meses más reciente |
| Bajo demanda | Caída de F1 en monitoreo > 5 pts | Auditar drift de features (población estable? sectores nuevos?) y re-entrenar |

---

## 12. Cómo ejecutar

```bash
# Pipeline completo — genera data, MDT, entrena, SHAP y scoring
make model

# Solo re-entrenamiento asumiendo data existente
make model-train

# Limpieza de artefactos
make clean
```

Secuencia de targets:
1. `model-data` → ejecuta `src.data.generar_{dim_merchants,fact_performance,fact_support_tickets}`
2. `model-train` → ejecuta `model.{feature_engineering,train_model,explain,predict}`

---

## 13. Archivos principales

| Responsabilidad | Archivo |
|---|---|
| Paths y constantes | [config/settings.py](../config/settings.py) |
| Generadores de data | [src/data/](../src/data/) |
| MDT | [model/feature_engineering.py](../model/feature_engineering.py) |
| Train 60/20/20 | [model/train_model.py](../model/train_model.py) |
| SHAP | [model/explain.py](../model/explain.py) |
| Scoring + segmentación | [model/predict.py](../model/predict.py) |
