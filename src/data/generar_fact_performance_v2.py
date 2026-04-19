"""
====================================================================
 Generador de Datos Sintéticos — Reto Deuna (Interact2Hack 2026)
 TABLA 2 v2: fact_performance_monthly (Performance Histórica Mensual)
 ** VERSIÓN CON RUIDO GAUSSIANO + CLIPPING PARA AUC ~0.76-0.83 **
====================================================================

 Cambios respecto a la v1 para bajar el AUC a rango realista:

 1. Mayor ruido en salud_latente (σ 0.08 → 0.15): la asignación de
    churn queda menos determinada por las features observables.
 2. Decaimiento menos abrupto: factor_decay (0.40-0.70 → 0.62-0.85).
 3. Señal de tickets suavizada: lambda churner reducida para no ser
    tan discriminante.
 4. Función agregar_ruido_gaussiano(): aplica ruido gaussiano con
    clipping a todas las columnas numéricas del dataframe final.
    Esto mezcla la distribución churner/sano sin crear valores
    aberrantes fuera de rangos plausibles.

 Requisito previo: dim_merchants.csv (Tabla 1)
 Output: fact_performance_monthly.csv + ground truth etiqueta

 Autor: Equipo Hackathon
 Fecha: Abril 2026
 Reproducibilidad: semilla fija = 42
====================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

# ============================================================
# 1. CONFIGURACIÓN GLOBAL Y REPRODUCIBILIDAD
# ============================================================
SEED = 42
np.random.seed(SEED)

FECHA_FIN = datetime(2026, 3, 1)
MESES_HISTORIA = 12
FECHA_INICIO = FECHA_FIN - relativedelta(months=MESES_HISTORIA - 1)  # 2025-04-01

TASA_CHURN_OBJETIVO = 0.13

# ============================================================
# 2. CATÁLOGOS DE CALIBRACIÓN
# ============================================================

ESTACIONALIDAD = {
    1:  0.82,
    2:  0.90,
    3:  1.05,
    4:  1.00,
    5:  1.02,
    6:  1.00,
    7:  0.98,
    8:  1.05,
    9:  1.00,
    10: 1.00,
    11: 1.10,
    12: 1.43,
}

ESCALA_POR_SEGMENTO = {
    "Micro":   {"trx_base": 180,  "trx_sigma": 0.45},
    "Pequeña": {"trx_base": 650,  "trx_sigma": 0.40},
    "Mediana": {"trx_base": 2500, "trx_sigma": 0.35},
    "Grande":  {"trx_base": 8000, "trx_sigma": 0.30},
}

SOPORTE_LAMBDA_SANO = 0.20
SOPORTE_LAMBDA_PROBLEMATICO = 1.20

SEVERIDAD_POR_TIPO = {
    "Restaurantes y picanterías":       3.2,
    "Comida rápida y food trucks":      2.8,
    "Bares y cantinas":                 3.0,
    "Venta y repuestos de motos":       3.5,
    "Farmacias y artículos médicos":    2.5,
    "_default":                         2.2,
}

# ============================================================
# 3. ASIGNACIÓN DE SALUD LATENTE Y ETIQUETA DE ABANDONO
# ============================================================

def calcular_salud_latente(row: pd.Series) -> float:
    score = 0.5

    tenure_dias = (FECHA_FIN - pd.to_datetime(row["fecha_onboarding"])).days
    tenure_meses = tenure_dias / 30
    if tenure_meses > 24:
        score += 0.20
    elif tenure_meses > 12:
        score += 0.10
    elif tenure_meses < 3:
        score -= 0.15

    if row["region"] in ("Pichincha", "Guayas"):
        score += 0.08
    elif row["region"] in ("Galápagos", "Morona Santiago", "Napo",
                           "Pastaza", "Zamora Chinchipe", "Orellana"):
        score -= 0.10

    if row["segmento_comercial"] in ("Grande", "Mediana"):
        score += 0.10
    elif row["segmento_comercial"] == "Micro":
        score -= 0.03

    tipos_resilientes = ["Tiendas de abarrotes y víveres",
                         "Farmacias y artículos médicos",
                         "Panaderías y pastelerías",
                         "Cafeterías y panaderías"]
    tipos_volatiles = ["Boutiques y ropa",
                       "Bares y cantinas",
                       "Venta y repuestos de motos",
                       "Librerías y útiles escolares"]
    if row["tipo_negocio_desc"] in tipos_resilientes:
        score += 0.05
    elif row["tipo_negocio_desc"] in tipos_volatiles:
        score -= 0.05

    # CAMBIO v2: σ aumentado de 0.08 a 0.15 para que la asignación
    # de churn sea menos determinada por las features observables.
    score += np.random.normal(0, 0.15)

    return float(np.clip(score, 0.01, 0.99))


def asignar_abandono(df_merchants: pd.DataFrame) -> pd.DataFrame:
    df = df_merchants.copy()
    np.random.seed(SEED)

    df["_salud_latente"] = df.apply(calcular_salud_latente, axis=1)

    prob_base = (1 - df["_salud_latente"]) ** 3
    n_churners_objetivo = int(len(df) * TASA_CHURN_OBJETIVO)
    prob_norm = prob_base / prob_base.sum()
    churners_idx = np.random.choice(
        df.index, size=n_churners_objetivo, replace=False, p=prob_norm
    )
    df["abandono_30d"] = 0
    df.loc[churners_idx, "abandono_30d"] = 1

    return df


# ============================================================
# 4. GENERACIÓN DE SERIES TEMPORALES POR COMERCIO
# ============================================================

def generar_trayectoria_mensual(merchant_row: pd.Series,
                                 meses_fechas: list) -> list:
    n_meses = len(meses_fechas)
    salud = merchant_row["_salud_latente"]
    es_churner = merchant_row["abandono_30d"] == 1
    segmento = merchant_row["segmento_comercial"]
    tipo_desc = merchant_row["tipo_negocio_desc"]
    fecha_onboard = pd.to_datetime(merchant_row["fecha_onboarding"])

    escala = ESCALA_POR_SEGMENTO[segmento]
    trx_base_mes = escala["trx_base"]
    trx_base_mes *= (0.7 + 0.6 * salud)

    ticket_base = _ticket_base_por_tipo(tipo_desc)

    if es_churner:
        meses_decaimiento = np.random.randint(3, 6)
        inicio_decaimiento = n_meses - meses_decaimiento
        # CAMBIO v2: decay menos abrupto (0.40-0.70 → 0.62-0.85)
        # Esto hace que los churners no caigan tan drásticamente,
        # aumentando el solapamiento con comercios sanos.
        factor_decay = np.random.uniform(0.62, 0.85)

        count_trx_trayectoria = []
        for i in range(n_meses):
            if i < inicio_decaimiento:
                mes_num = meses_fechas[i].month
                factor = ESTACIONALIDAD[mes_num]
                val = trx_base_mes * factor * np.random.lognormal(0, 0.15)
            else:
                meses_dentro_decay = i - inicio_decaimiento + 1
                base_decay = count_trx_trayectoria[inicio_decaimiento - 1]
                val = base_decay * (factor_decay ** meses_dentro_decay)
                val *= np.random.uniform(0.7, 1.1)
            count_trx_trayectoria.append(max(0, val))
    else:
        tenure_meses_inicio = (meses_fechas[0] - fecha_onboard).days / 30
        if tenure_meses_inicio < 6:
            tendencia_mensual = np.random.uniform(1.03, 1.10)
        elif tenure_meses_inicio < 18:
            tendencia_mensual = np.random.uniform(1.01, 1.04)
        else:
            tendencia_mensual = np.random.uniform(0.99, 1.02)

        count_trx_trayectoria = []
        base_actual = trx_base_mes
        for i in range(n_meses):
            mes_num = meses_fechas[i].month
            factor = ESTACIONALIDAD[mes_num]
            val = base_actual * factor * np.random.lognormal(0, 0.18)
            count_trx_trayectoria.append(max(0, val))
            base_actual *= tendencia_mensual

    filas = []
    for i, mes_dt in enumerate(meses_fechas):
        if mes_dt < fecha_onboard.replace(day=1):
            continue

        count_trx = int(round(count_trx_trayectoria[i]))

        factor_ticket_mes = 1 + (ESTACIONALIDAD[mes_dt.month] - 1) * 0.3
        ticket_mes = ticket_base * factor_ticket_mes * np.random.lognormal(0, 0.10)
        tpv_mes = count_trx * ticket_mes

        if es_churner and i >= inicio_decaimiento:
            meses_dentro_decay = i - inicio_decaimiento + 1
            tasa_rechazo_base = 0.04 + 0.04 * meses_dentro_decay
            tasa_rechazo = min(0.30, tasa_rechazo_base + np.random.uniform(0, 0.05))
        else:
            tasa_rechazo = np.random.beta(2, 50)
            if salud < 0.35:
                tasa_rechazo += np.random.uniform(0.02, 0.05)
            tasa_rechazo = min(0.15, tasa_rechazo)

        dias_en_mes = 30
        if count_trx == 0:
            dias_sin_trx_max = dias_en_mes
            dias_desde_ultima = dias_en_mes
        else:
            trx_por_dia = count_trx / dias_en_mes
            lambda_racha = max(trx_por_dia, 0.3)
            dias_sin_trx_max = int(min(dias_en_mes,
                                       np.random.exponential(scale=3/lambda_racha)))
            dias_desde_ultima = int(np.random.uniform(0, max(1, dias_sin_trx_max)))

        if es_churner and i >= inicio_decaimiento:
            meses_dentro_decay = i - inicio_decaimiento + 1
            dias_sin_trx_max = min(30, max(dias_sin_trx_max,
                                           int(5 * meses_dentro_decay + np.random.uniform(0, 5))))
            if i == n_meses - 1:
                dias_desde_ultima = min(30, max(dias_desde_ultima, 15))

        if es_churner:
            if i < inicio_decaimiento - 1:
                lambda_ticket = SOPORTE_LAMBDA_SANO
            elif i < n_meses - 1:
                # CAMBIO v2: pico de tickets reducido (3.0+1.5x → 1.8+0.8x)
                # Así los churners no tienen una señal tan obvia en soporte.
                lambda_ticket = 1.8 + 0.8 * (i - inicio_decaimiento + 1)
            else:
                lambda_ticket = 0.3
        else:
            if salud < 0.4:
                lambda_ticket = SOPORTE_LAMBDA_PROBLEMATICO
            else:
                lambda_ticket = SOPORTE_LAMBDA_SANO

        tickets_abiertos = np.random.poisson(lambda_ticket)
        if es_churner and i >= inicio_decaimiento:
            tasa_resolucion = np.random.uniform(0.30, 0.55)
        else:
            tasa_resolucion = np.random.uniform(0.75, 0.95)
        tickets_resueltos = int(tickets_abiertos * tasa_resolucion)

        if tickets_resueltos > 0:
            if es_churner and i >= inicio_decaimiento:
                tiempo_resolucion = np.random.lognormal(mean=np.log(96), sigma=0.6)
            else:
                tiempo_resolucion = np.random.lognormal(mean=np.log(36), sigma=0.5)
        else:
            tiempo_resolucion = 0.0

        if tickets_abiertos > 0:
            sev_base = SEVERIDAD_POR_TIPO.get(tipo_desc, SEVERIDAD_POR_TIPO["_default"])
            if es_churner and i >= inicio_decaimiento:
                sev_base += np.random.uniform(0.5, 1.2)
            severidad = np.clip(sev_base + np.random.normal(0, 0.5), 1.0, 5.0)
        else:
            severidad = 0.0

        filas.append({
            "merchant_id":                 merchant_row["merchant_id"],
            "mes_reporte":                 mes_dt.date(),
            "tpv_mensual":                 round(tpv_mes, 2),
            "count_trx":                   count_trx,
            "ticket_promedio":             round(tpv_mes / count_trx, 2) if count_trx > 0 else 0.0,
            "tasa_rechazo":                round(tasa_rechazo, 4),
            "dias_sin_transaccion_max":    dias_sin_trx_max,
            "dias_desde_ultima_trx":       dias_desde_ultima,
            "tickets_soporte_abiertos":    tickets_abiertos,
            "tickets_soporte_resueltos":   tickets_resueltos,
            "tiempo_resolucion_prom_hrs":  round(tiempo_resolucion, 1),
            "severidad_prom_tickets":      round(severidad, 2),
        })

    return filas


def _ticket_base_por_tipo(tipo_desc: str) -> float:
    mapping = {
        "Tiendas de abarrotes y víveres":   8,
        "Restaurantes y picanterías":       12,
        "Comida rápida y food trucks":      6,
        "Cafeterías y panaderías":          5,
        "Farmacias y artículos médicos":    18,
        "Peluquerías y salones de belleza": 15,
        "Boutiques y ropa":                 28,
        "Fruterías y mercados":             7,
        "Transporte (taxis, cooperativas)": 4,
        "Bazares y papelerías":             6,
        "Panaderías y pastelerías":         5,
        "Comercio mixto (bazar + abarrote)":9,
        "Bares y cantinas":                 16,
        "Lavanderías y tintorerías":        8,
        "Venta y repuestos de motos":       45,
        "Librerías y útiles escolares":     12,
    }
    return mapping.get(tipo_desc, 10)


# ============================================================
# 5. RUIDO GAUSSIANO CON CLIPPING (pieza central v2)
# ============================================================

def agregar_ruido_gaussiano(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Aplica ruido gaussiano con clipping a todas las columnas numéricas.

    El ruido relativo mezcla las distribuciones churner/sano sin
    producir valores aberrantes gracias al clipping a rangos plausibles.
    Objetivo: AUC final en el rango 0.76–0.83.

    Parámetros de ruido calibrados empíricamente:
    - Métricas de volumen (count_trx, tpv_mensual): ruido relativo 22%
    - tasa_rechazo: ruido absoluto σ=0.012, clip [0, 0.35]
    - días sin transacción: ruido absoluto σ=3 días, clip [0, 30]
    - tickets_soporte_abiertos: ruido absoluto σ=0.7, clip ≥0, entero
    - tickets_soporte_resueltos: recalculado ≤ tickets_abiertos
    - tiempo_resolucion y severidad: ruido relativo 18%
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # --- Volumen transaccional ---
    ruido_rel_volumen = 0.22
    for col in ["count_trx", "tpv_mensual", "ticket_promedio"]:
        ruido = rng.normal(0, ruido_rel_volumen, size=n)
        df[col] = df[col] * (1 + ruido)
        df[col] = np.clip(df[col], 0, None)

    df["count_trx"] = df["count_trx"].round().astype(int)
    df["tpv_mensual"] = df["tpv_mensual"].round(2)
    # Recalcular ticket_promedio coherente con count_trx y tpv
    mask_trx = df["count_trx"] > 0
    df.loc[mask_trx, "ticket_promedio"] = (
        df.loc[mask_trx, "tpv_mensual"] / df.loc[mask_trx, "count_trx"]
    ).round(2)
    df.loc[~mask_trx, "ticket_promedio"] = 0.0

    # --- Tasa de rechazo ---
    ruido_rechazo = rng.normal(0, 0.012, size=n)
    df["tasa_rechazo"] = np.clip(df["tasa_rechazo"] + ruido_rechazo, 0.0, 0.35).round(4)

    # --- Días sin transacción ---
    ruido_dias = rng.normal(0, 3.0, size=n)
    df["dias_sin_transaccion_max"] = np.clip(
        df["dias_sin_transaccion_max"] + ruido_dias, 0, 30
    ).round().astype(int)

    ruido_dias2 = rng.normal(0, 3.0, size=n)
    df["dias_desde_ultima_trx"] = np.clip(
        df["dias_desde_ultima_trx"] + ruido_dias2, 0, 30
    ).round().astype(int)
    # dias_desde_ultima no puede superar dias_sin_transaccion_max
    df["dias_desde_ultima_trx"] = np.minimum(
        df["dias_desde_ultima_trx"], df["dias_sin_transaccion_max"]
    )

    # --- Tickets de soporte ---
    ruido_tickets = rng.normal(0, 0.7, size=n)
    df["tickets_soporte_abiertos"] = np.clip(
        df["tickets_soporte_abiertos"] + ruido_tickets, 0, None
    ).round().astype(int)
    # Resueltos no puede superar abiertos
    df["tickets_soporte_resueltos"] = np.minimum(
        df["tickets_soporte_resueltos"], df["tickets_soporte_abiertos"]
    )
    df["tickets_soporte_resueltos"] = df["tickets_soporte_resueltos"].clip(lower=0)

    # --- Tiempo de resolución ---
    ruido_rel_tiempo = 0.18
    ruido_t = rng.normal(0, ruido_rel_tiempo, size=n)
    df["tiempo_resolucion_prom_hrs"] = np.clip(
        df["tiempo_resolucion_prom_hrs"] * (1 + ruido_t), 0, None
    ).round(1)

    # --- Severidad ---
    ruido_sev = rng.normal(0, 0.25, size=n)
    df["severidad_prom_tickets"] = np.clip(
        df["severidad_prom_tickets"] + ruido_sev, 0.0, 5.0
    ).round(2)
    # Mantener 0.0 para filas sin tickets
    df.loc[df["tickets_soporte_abiertos"] == 0, "severidad_prom_tickets"] = 0.0

    return df


# ============================================================
# 6. ORQUESTADOR
# ============================================================

def generar_fact_performance_monthly(df_merchants: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(SEED)

    meses_fechas = [FECHA_INICIO + relativedelta(months=k)
                    for k in range(MESES_HISTORIA)]

    print(f"Generando {MESES_HISTORIA} meses de historia para {len(df_merchants)} comercios...")
    print(f"   Rango temporal: {meses_fechas[0].date()} → {meses_fechas[-1].date()}")

    todas_filas = []
    for idx, row in df_merchants.iterrows():
        filas_comercio = generar_trayectoria_mensual(row, meses_fechas)
        todas_filas.extend(filas_comercio)

    df_fact = pd.DataFrame(todas_filas)

    print("Aplicando ruido gaussiano con clipping para AUC objetivo 0.76-0.83...")
    df_fact = agregar_ruido_gaussiano(df_fact, seed=SEED + 1)

    return df_fact


# ============================================================
# 7. VALIDACIÓN
# ============================================================

def validar_dataset(df_fact: pd.DataFrame, df_merchants: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"DATASET fact_performance_monthly v2  —  {len(df_fact):,} filas")
    print("=" * 70)

    print(f"\n[1] Comercios cubiertos: {df_fact['merchant_id'].nunique():,} / {len(df_merchants):,}")
    print(f"    Meses únicos:         {df_fact['mes_reporte'].nunique()}")
    print(f"    Filas por comercio (media): {len(df_fact) / df_fact['merchant_id'].nunique():.1f}")

    print("\n[2] Estacionalidad — TPV promedio por mes (esperamos pico en diciembre):")
    df_fact["mes_reporte"] = pd.to_datetime(df_fact["mes_reporte"])
    tpv_mes = df_fact.groupby(df_fact["mes_reporte"].dt.month)["tpv_mensual"].mean()
    for m, v in tpv_mes.items():
        bar = "█" * int(v / tpv_mes.max() * 30)
        print(f"    Mes {m:2d}: ${v:>10,.0f}  {bar}")

    print("\n[3] Distribución de tasa_rechazo (esperamos media ~4%):")
    print(f"    Media:    {df_fact['tasa_rechazo'].mean():.3f}")
    print(f"    Mediana:  {df_fact['tasa_rechazo'].median():.3f}")
    print(f"    P95:      {df_fact['tasa_rechazo'].quantile(0.95):.3f}")

    print("\n[4] Tickets de soporte:")
    print(f"    Comercios sin tickets en ningún mes: "
          f"{(df_fact.groupby('merchant_id')['tickets_soporte_abiertos'].sum() == 0).sum()}")
    print(f"    Tickets abiertos — media: {df_fact['tickets_soporte_abiertos'].mean():.2f}")
    print(f"    Tickets abiertos — max:   {df_fact['tickets_soporte_abiertos'].max()}")

    print("\n[5] Cruce Churner vs Sano — último mes del dataset:")
    ultimo_mes = df_fact["mes_reporte"].max()
    df_ultimo = df_fact[df_fact["mes_reporte"] == ultimo_mes].merge(
        df_merchants[["merchant_id", "abandono_30d"]], on="merchant_id"
    )
    comp = df_ultimo.groupby("abandono_30d").agg(
        tpv_prom=("tpv_mensual", "mean"),
        count_trx_prom=("count_trx", "mean"),
        tasa_rechazo_prom=("tasa_rechazo", "mean"),
        tickets_prom=("tickets_soporte_abiertos", "mean"),
        dias_sin_trx_prom=("dias_sin_transaccion_max", "mean"),
    ).round(2)
    print(comp.to_string())

    print("\n[6] Etiqueta ground truth:")
    tasa_real = df_merchants["abandono_30d"].mean()
    print(f"    Tasa de abandono efectiva: {tasa_real:.3f} (objetivo: {TASA_CHURN_OBJETIVO})")
    print(f"    Churners: {df_merchants['abandono_30d'].sum()} comercios")

    print("\n[7] Checks de integridad:")
    print(f"    Sin nulos:                     {not df_fact.isnull().any().any()}")
    print(f"    TPV no negativo:               {(df_fact['tpv_mensual'] >= 0).all()}")
    print(f"    count_trx entero no negativo:  {(df_fact['count_trx'] >= 0).all()}")
    print(f"    tasa_rechazo en [0, 0.35]:     "
          f"{((df_fact['tasa_rechazo'] >= 0) & (df_fact['tasa_rechazo'] <= 0.35)).all()}")
    print(f"    dias_sin_trx en [0, 30]:       "
          f"{((df_fact['dias_sin_transaccion_max'] >= 0) & (df_fact['dias_sin_transaccion_max'] <= 30)).all()}")

    print("\n[8] Muestra de 5 filas:")
    print(df_fact.head().to_string())

    print("\n[9] Trayectoria ejemplar de un CHURNER (decaimiento suavizado):")
    churner_ejemplo = df_merchants[df_merchants["abandono_30d"] == 1].iloc[0]["merchant_id"]
    traj = df_fact[df_fact["merchant_id"] == churner_ejemplo].sort_values("mes_reporte")
    print(f"    Comercio: {churner_ejemplo}")
    print(traj[["mes_reporte", "count_trx", "tpv_mensual",
                "tasa_rechazo", "tickets_soporte_abiertos"]].to_string(index=False))


# ============================================================
# 8. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    if not os.path.exists("data/raw/dim_merchants.csv"):
        raise FileNotFoundError(
            "No se encuentra data/raw/dim_merchants.csv. "
            "Ejecuta primero generar_dim_merchants.py."
        )
    df_merchants = pd.read_csv("data/raw/dim_merchants.csv")
    print(f"✓ Cargado dim_merchants.csv — {len(df_merchants):,} comercios\n")

    df_merchants = asignar_abandono(df_merchants)
    print(f"✓ Asignada etiqueta de abandono — "
          f"{df_merchants['abandono_30d'].sum()} churners "
          f"({df_merchants['abandono_30d'].mean()*100:.1f}%)\n")

    df_fact = generar_fact_performance_monthly(df_merchants)

    validar_dataset(df_fact, df_merchants)

    df_fact.to_csv("data/raw/fact_performance_monthly.csv", index=False, encoding="utf-8")
    try:
        df_fact.to_parquet("data/raw/fact_performance_monthly.parquet", index=False)
        print("\n✓ Guardado: data/raw/fact_performance_monthly.csv + parquet")
    except Exception as e:
        print(f"\n✓ Guardado: data/raw/fact_performance_monthly.csv (parquet no disponible: {e})")

    df_merchants_out = df_merchants.drop(columns=["_salud_latente"])
    df_merchants_out.to_csv("data/raw/dim_merchants_con_abandono.csv", index=False, encoding="utf-8")
    print("✓ Guardado: data/raw/dim_merchants_con_abandono.csv (con la columna abandono_30d añadida)")
