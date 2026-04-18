"""Construcción de la MDT (Master Data Table) para el modelo de churn.

Combina tres fuentes — dim_merchants, fact_performance_monthly y
fact_support_tickets — en una fila por comercio con:

    - Variables estáticas  : segmento, región, tipo de negocio, tenure
    - Lags 0..4            : últimos 5 meses de operación (patrón Databricks)
    - Agregados 3/6/12 m   : suma, media, std, meses activos
    - Deltas mes a mes     : aceleración de deterioro
    - Ratios derivados     : volatilidad, rechazo, tickets por trx
    - Tickets 6 m          : conteo total, no resueltos, severidad, satisfacción,
                              desglose por categoría crítica

La fila de cada comercio lleva la etiqueta `abandono_30d` (0/1) obtenida
del join con churn_labels.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permitir ejecución directa y como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PATHS, FECHA_CORTE


# ── Métricas de performance mensual que se van a "lagear" ──────────────────
PERFORMANCE_METRICS = [
    "count_trx",
    "tpv_mensual",
    "ticket_promedio",
    "tasa_rechazo",
    "dias_sin_transaccion_max",
    "dias_desde_ultima_trx",
    "tickets_soporte_abiertos",
    "tickets_soporte_resueltos",
    "tiempo_resolucion_prom_hrs",
    "severidad_prom_tickets",
]

LAGS = [0, 1, 2, 3, 4]
WINDOWS = (3, 6, 12)

# Categorías de tickets que queremos exponer individualmente (foco en graves)
TICKET_CATEGORIAS_FOCO = [
    "pago_rechazado",
    "liquidacion_demora",
    "app_congelada",
    "facturacion_comisiones",
]


# ════════════════════════════════════════════════════════════════════════════
# Features desde fact_performance_monthly
# ════════════════════════════════════════════════════════════════════════════

def _bucket_recencia(dias: float) -> str:
    if pd.isna(dias):
        return "sin_registro"
    if dias <= 5:
        return "menos_5"
    if dias <= 10:
        return "entre_5_y_10"
    if dias <= 20:
        return "entre_10_y_20"
    if dias <= 30:
        return "entre_20_y_30"
    return "mas_de_30"


def _lag_frame(performance: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Pivot: una fila por merchant con columnas metric_{lag}."""
    wide = None
    for lag in LAGS:
        mes = corte - pd.DateOffset(months=lag)
        snap = performance[performance["mes_reporte"] == mes].copy()
        snap = snap[["merchant_id", *PERFORMANCE_METRICS]]
        rename = {c: f"{c}_{lag}" for c in PERFORMANCE_METRICS}
        snap = snap.rename(columns=rename)
        wide = snap if wide is None else wide.merge(snap, on="merchant_id", how="outer")
    return wide


def _rolling_aggregates(performance: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Agregados sobre ventanas de 3, 6 y 12 meses hacia atrás del corte."""
    out = None
    for window in WINDOWS:
        desde = corte - pd.DateOffset(months=window - 1)
        ventana = performance[
            (performance["mes_reporte"] >= desde)
            & (performance["mes_reporte"] <= corte)
        ]
        agg = ventana.groupby("merchant_id").agg(
            **{
                f"tx_sum_{window}m":            ("count_trx", "sum"),
                f"tx_mean_{window}m":           ("count_trx", "mean"),
                f"tx_std_{window}m":            ("count_trx", "std"),
                f"tpv_sum_{window}m":           ("tpv_mensual", "sum"),
                f"tpv_mean_{window}m":          ("tpv_mensual", "mean"),
                f"tpv_std_{window}m":           ("tpv_mensual", "std"),
                f"rechazo_mean_{window}m":      ("tasa_rechazo", "mean"),
                f"rechazo_max_{window}m":       ("tasa_rechazo", "max"),
                f"tickets_sum_{window}m":       ("tickets_soporte_abiertos", "sum"),
                f"tiempo_res_mean_{window}m":   ("tiempo_resolucion_prom_hrs", "mean"),
                f"meses_activos_{window}m":     ("count_trx", lambda s: int((s > 0).sum())),
            }
        )
        out = agg if out is None else out.join(agg, how="outer")
    return out.reset_index()


def _deltas(wide: pd.DataFrame) -> pd.DataFrame:
    """Deltas mensuales consecutivos sobre métricas clave."""
    df = wide.copy()
    for metric in ["count_trx", "tpv_mensual", "ticket_promedio",
                   "tasa_rechazo", "dias_desde_ultima_trx",
                   "tickets_soporte_abiertos"]:
        for newer, older in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            col_new = f"{metric}_{newer}"
            col_old = f"{metric}_{older}"
            if col_new in df.columns and col_old in df.columns:
                df[f"delta_{metric}_{newer}_{older}"] = df[col_new] - df[col_old]
    return df


# ════════════════════════════════════════════════════════════════════════════
# Features desde fact_support_tickets (ventana 6 meses)
# ════════════════════════════════════════════════════════════════════════════

def _ticket_aggregates(tickets: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Agregados a 6 meses de la tabla granular de tickets."""
    desde = corte - pd.DateOffset(months=5)  # ventana de 6 meses incluyendo el corte
    hasta = corte + pd.DateOffset(months=1)  # hasta cierre del mes de corte

    tk = tickets[
        (tickets["fecha_apertura"] >= desde)
        & (tickets["fecha_apertura"] < hasta)
    ].copy()

    # Flags derivados
    tk["es_no_resuelto"] = tk["estado"].isin(["abierto", "en_proceso", "escalado"]).astype(int)
    tk["es_escalado"]    = (tk["estado"] == "escalado").astype(int)

    agg = tk.groupby("merchant_id").agg(
        tickets_total_6m=       ("ticket_id", "count"),
        tickets_no_resueltos_6m=("es_no_resuelto", "sum"),
        tickets_escalados_6m=   ("es_escalado", "sum"),
        severidad_max_6m=       ("severidad", "max"),
        severidad_mean_6m=      ("severidad", "mean"),
        tiempo_res_max_6m=      ("tiempo_resolucion_hrs", "max"),
        satisfaccion_mean_6m=   ("satisfaccion_post_cierre", "mean"),
    ).reset_index()

    # Desglose por categoría crítica
    for cat in TICKET_CATEGORIAS_FOCO:
        cat_counts = (
            tk[tk["categoria"] == cat]
            .groupby("merchant_id")
            .size()
            .rename(f"tickets_{cat}_6m")
            .reset_index()
        )
        agg = agg.merge(cat_counts, on="merchant_id", how="left")

    return agg


# ════════════════════════════════════════════════════════════════════════════
# Features estáticos de dim_merchants
# ════════════════════════════════════════════════════════════════════════════

def _static_features(merchants: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    df = merchants[[
        "merchant_id", "segmento_comercial", "region",
        "tipo_negocio_ciiu", "tipo_negocio_desc", "ejecutivo_cuenta",
        "fecha_onboarding",
    ]].copy()

    df["fecha_onboarding"] = pd.to_datetime(df["fecha_onboarding"])
    df["tenure_meses"] = ((corte - df["fecha_onboarding"]).dt.days / 30).round(1)
    df["es_auto_gestionado"] = (df["ejecutivo_cuenta"] == "Auto-gestionado").astype(int)

    return df.drop(columns=["fecha_onboarding", "ejecutivo_cuenta", "tipo_negocio_ciiu"])


# ════════════════════════════════════════════════════════════════════════════
# Orquestación
# ════════════════════════════════════════════════════════════════════════════

def build_mdt(
    merchants: pd.DataFrame,
    performance: pd.DataFrame,
    tickets: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    fecha_corte: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Construye la MDT cross-sectional a la fecha de corte.

    Si `labels` es None la MDT no incluye la columna target (modo inferencia).
    """
    performance = performance.copy()
    performance["mes_reporte"] = pd.to_datetime(performance["mes_reporte"])

    tickets = tickets.copy()
    tickets["fecha_apertura"] = pd.to_datetime(tickets["fecha_apertura"])

    corte = (
        pd.Timestamp(fecha_corte) if fecha_corte is not None
        else performance["mes_reporte"].max()
    )

    wide    = _lag_frame(performance, corte)
    rolling = _rolling_aggregates(performance, corte)
    tk_agg  = _ticket_aggregates(tickets, corte)
    static  = _static_features(merchants, corte)

    mdt = static.merge(wide,    on="merchant_id", how="left")
    mdt = mdt.merge(rolling,    on="merchant_id", how="left")
    mdt = mdt.merge(tk_agg,     on="merchant_id", how="left")

    mdt["fecha_corte"] = corte
    mdt["recencia_bucket_0"] = mdt["dias_desde_ultima_trx_0"].apply(_bucket_recencia)

    # Ratios derivados (con guardas contra div/0)
    tx6 = mdt["tx_sum_6m"].replace(0, np.nan)
    mdt["tickets_per_tx_6m"]      = (mdt["tickets_sum_6m"] / tx6).fillna(0)
    mdt["volatilidad_tx_6m"]      = (mdt["tx_std_6m"] / mdt["tx_mean_6m"].replace(0, np.nan)).fillna(0)
    mdt["volatilidad_tpv_6m"]     = (mdt["tpv_std_6m"] / mdt["tpv_mean_6m"].replace(0, np.nan)).fillna(0)
    mdt["tasa_no_resuelto_6m"]    = (
        mdt["tickets_no_resueltos_6m"] / mdt["tickets_total_6m"].replace(0, np.nan)
    ).fillna(0)

    # Deltas (requieren las columnas de lag ya presentes)
    mdt = _deltas(mdt)

    # Relleno para numéricos faltantes (comercios con historia corta,
    # sin tickets, etc.)
    numeric_cols = mdt.select_dtypes(include=[np.number]).columns
    mdt[numeric_cols] = mdt[numeric_cols].fillna(0)

    # Target
    if labels is not None:
        mdt = mdt.merge(labels[["merchant_id", "abandono_30d"]],
                        on="merchant_id", how="left")
        mdt["abandono_30d"] = mdt["abandono_30d"].fillna(0).astype(int)

    return mdt


def main(raw_dir: str | Path | None = None,
         out_dir: str | Path | None = None) -> Path:
    raw_dir = Path(raw_dir) if raw_dir else PATHS.RAW_DIR
    out_dir = Path(out_dir) if out_dir else PATHS.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    merchants   = pd.read_csv(raw_dir / "dim_merchants.csv")
    performance = pd.read_csv(raw_dir / "fact_performance_monthly.csv",
                              parse_dates=["mes_reporte"])
    tickets     = pd.read_csv(raw_dir / "fact_support_tickets.csv",
                              parse_dates=["fecha_apertura", "fecha_cierre"])
    labels      = pd.read_csv(raw_dir / "churn_labels.csv")

    mdt = build_mdt(merchants, performance, tickets, labels,
                    fecha_corte=pd.Timestamp(FECHA_CORTE))

    out_path = out_dir / "mdt_churn.parquet"
    mdt.to_parquet(out_path, index=False)

    print(f"MDT construida: {mdt.shape[0]} filas x {mdt.shape[1]} columnas → {out_path}")
    print(f"Tasa de churn:  {mdt['abandono_30d'].mean():.2%}")
    print(f"Fecha corte:    {mdt['fecha_corte'].iloc[0].date()}")
    return out_path


if __name__ == "__main__":
    main()
