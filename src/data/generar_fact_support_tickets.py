"""
====================================================================
 Generador de Datos Sintéticos — Reto Deuna (Interact2Hack 2026)
 TABLA 4: fact_support_tickets (Detalle de Tickets de Soporte)
====================================================================

 Genera el detalle individual de cada ticket de soporte, derivado
 de los agregados mensuales de Tabla 2 para garantizar consistencia.

 PRINCIPIO DE DISEÑO:
 Esta tabla se genera COHERENTE con Tabla 2. Si Tabla 2 dice que
 un comercio abrió 5 tickets en enero con severidad promedio 3.2,
 Tabla 4 contendrá exactamente 5 filas con severidades que
 promedien 3.2 y con las demás métricas consistentes.

 Requisitos previos:
 - dim_merchants_con_abandono.csv (Tabla 1 + etiqueta)
 - fact_performance_monthly.csv (Tabla 2)

 Output: fact_support_tickets.csv

 Autor: Equipo Hackathon
 Fecha: Abril 2026
 Reproducibilidad: semilla fija = 42
====================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PATHS

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
SEED = 42
np.random.seed(SEED)

# ============================================================
# 2. CATÁLOGO DE CATEGORÍAS DE TICKETS
# ============================================================

# Categorías basadas en FAQs reales de Deuna y documento de arquitectura.
# Cada categoría tiene su peso y una severidad base característica.
# Ej: "liquidacion_demora" es crítica (afecta flujo de caja); "consulta_general" es leve.
CATEGORIAS = {
    "pago_rechazado":         {"peso": 0.25, "sev_base": 3.2, "sev_sigma": 0.8},
    "liquidacion_demora":     {"peso": 0.18, "sev_base": 3.8, "sev_sigma": 0.7},
    "app_congelada":          {"peso": 0.15, "sev_base": 2.8, "sev_sigma": 0.9},
    "configuracion_qr":       {"peso": 0.10, "sev_base": 2.0, "sev_sigma": 0.6},
    "cierre_caja":            {"peso": 0.08, "sev_base": 2.5, "sev_sigma": 0.7},
    "facturacion_comisiones": {"peso": 0.07, "sev_base": 3.0, "sev_sigma": 0.8},
    "consulta_general":       {"peso": 0.07, "sev_base": 1.5, "sev_sigma": 0.5},
    "problema_qr_fisico":     {"peso": 0.05, "sev_base": 2.2, "sev_sigma": 0.6},
    "roles_vendedor":         {"peso": 0.05, "sev_base": 2.0, "sev_sigma": 0.5},
}

# Normalizar pesos
_total = sum(c["peso"] for c in CATEGORIAS.values())
for c in CATEGORIAS.values():
    c["peso"] /= _total

CATEGORIAS_LISTA = list(CATEGORIAS.keys())
CATEGORIAS_PESOS = [CATEGORIAS[k]["peso"] for k in CATEGORIAS_LISTA]

# Estados posibles
ESTADOS_NO_RESUELTOS = ["abierto", "en_proceso", "escalado"]
PESOS_NO_RESUELTOS = [0.50, 0.35, 0.15]  # abierto + en_proceso dominan, escalado es raro

# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def generar_ticket_id(merchant_id: str, mes: str, i: int) -> str:
    """ID único determinístico por (comercio, mes, índice)."""
    raw = f"{merchant_id}_{mes}_{i}_{SEED}"
    h = hashlib.md5(raw.encode()).hexdigest()[:10].upper()
    return f"TKT-{h}"

def asignar_categoria_con_contexto(es_churner: bool,
                                    mes_es_decay: bool,
                                    tipo_negocio: str) -> str:
    """
    Los churners en fase de decay reportan más quejas de tipo crítico
    (liquidacion_demora, pago_rechazado). Los sanos tienen distribución
    más balanceada, con más consultas generales y configuración.
    """
    if es_churner and mes_es_decay:
        # Pesos sesgados hacia problemas graves
        pesos_adj = {
            "pago_rechazado":         0.35,
            "liquidacion_demora":     0.28,
            "app_congelada":          0.15,
            "facturacion_comisiones": 0.10,
            "cierre_caja":            0.05,
            "configuracion_qr":       0.02,
            "consulta_general":       0.02,
            "problema_qr_fisico":     0.02,
            "roles_vendedor":         0.01,
        }
    else:
        # Distribución normal según el catálogo
        pesos_adj = {k: CATEGORIAS[k]["peso"] for k in CATEGORIAS_LISTA}

    # Ajuste por tipo de negocio: restaurantes tienen más app_congelada
    # (POS integrado), transporte tiene más configuracion_qr (QR físico dañado).
    if "Restaurantes" in tipo_negocio or "Comida" in tipo_negocio:
        pesos_adj["app_congelada"] *= 1.5
    if "Transporte" in tipo_negocio:
        pesos_adj["problema_qr_fisico"] *= 2.5
    if "Farmacias" in tipo_negocio:
        pesos_adj["facturacion_comisiones"] *= 1.4

    # Renormalizar y elegir
    total = sum(pesos_adj.values())
    pesos_norm = [pesos_adj[k] / total for k in CATEGORIAS_LISTA]
    return np.random.choice(CATEGORIAS_LISTA, p=pesos_norm)

def generar_severidades_consistentes(n_tickets: int,
                                      severidad_objetivo: float,
                                      categorias: list) -> list:
    """
    Genera n severidades que promedien aproximadamente la severidad_objetivo.
    Estrategia: muestrear desde distribuciones por categoría, luego ajustar
    para hit el promedio objetivo.
    """
    if n_tickets == 0 or severidad_objetivo == 0:
        return []

    # Paso 1: muestrear severidades base por categoría
    sevs = []
    for cat in categorias:
        info = CATEGORIAS[cat]
        s = np.random.normal(info["sev_base"], info["sev_sigma"])
        sevs.append(np.clip(s, 1.0, 5.0))
    sevs = np.array(sevs)

    # Paso 2: ajustar al promedio objetivo (shift + clip)
    shift = severidad_objetivo - sevs.mean()
    sevs_adj = np.clip(sevs + shift, 1.0, 5.0)

    # Redondear a enteros (severidad es escala 1-5 entera)
    sevs_int = np.round(sevs_adj).astype(int)
    sevs_int = np.clip(sevs_int, 1, 5)
    return sevs_int.tolist()

def generar_tiempos_resolucion_consistentes(n_resueltos: int,
                                             tiempo_objetivo: float) -> list:
    """
    Genera n tiempos de resolución que promedien tiempo_objetivo (en horas).
    Usa distribución lognormal que refleja la realidad (colas largas).
    """
    if n_resueltos == 0 or tiempo_objetivo == 0:
        return []

    # Muestrear lognormal con sigma fija y ajustar la media
    sigma = 0.55
    # Para que una lognormal tenga media tiempo_objetivo: mu = ln(media) - sigma^2/2
    mu = np.log(max(tiempo_objetivo, 0.5)) - (sigma ** 2) / 2
    tiempos = np.random.lognormal(mean=mu, sigma=sigma, size=n_resueltos)

    # Ajustar para asegurar que la media sea exactamente el objetivo
    factor = tiempo_objetivo / tiempos.mean() if tiempos.mean() > 0 else 1.0
    tiempos *= factor
    tiempos = np.clip(tiempos, 0.5, 720)  # entre 30 min y 30 días

    return [round(t, 1) for t in tiempos]

def generar_satisfaccion(es_churner: bool, severidad: int,
                         tiempo_resolucion: float) -> int:
    """
    Satisfacción 1-5 tras resolución. Solo se captura en ~60% de casos
    (no todos responden la encuesta).
    Churners y casos graves/lentos dan puntajes más bajos.
    """
    if np.random.random() > 0.60:
        return None  # no respondió la encuesta

    if es_churner:
        base = 2.2
    else:
        base = 4.0

    # Penalización por severidad alta
    penalty_sev = (severidad - 2) * 0.3
    # Penalización por tiempo largo (>48h)
    penalty_tiempo = max(0, (tiempo_resolucion - 48) / 48) * 0.4

    score = base - penalty_sev - penalty_tiempo + np.random.normal(0, 0.5)
    return int(np.clip(round(score), 1, 5))

# ============================================================
# 4. ORQUESTADOR: DERIVAR TABLA 4 DE TABLA 2
# ============================================================

def generar_fact_support_tickets(df_performance: pd.DataFrame,
                                  df_merchants: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fila de Tabla 2 con tickets_soporte_abiertos > 0,
    genera las filas individuales de tickets en Tabla 4.
    """
    np.random.seed(SEED)

    # Merge para tener contexto del comercio en cada fila
    df = df_performance.merge(
        df_merchants[["merchant_id", "abandono_30d", "tipo_negocio_desc",
                      "fecha_onboarding"]],
        on="merchant_id", how="left"
    )
    df["mes_reporte"] = pd.to_datetime(df["mes_reporte"])

    # Filtrar solo filas con tickets
    df_con_tickets = df[df["tickets_soporte_abiertos"] > 0].copy()
    print(f"   Filas (comercio × mes) con tickets: {len(df_con_tickets):,}")
    print(f"   Total tickets esperados: {df_con_tickets['tickets_soporte_abiertos'].sum():,}")

    # Para identificar si un mes está en la "fase de decay" del churner,
    # vamos a considerar como decay los últimos 4 meses de cada churner
    fecha_fin = df["mes_reporte"].max()
    meses_decay_churner = pd.date_range(
        end=fecha_fin, periods=4, freq="MS"
    )

    todas_filas = []
    for _, row in df_con_tickets.iterrows():
        n_abiertos  = int(row["tickets_soporte_abiertos"])
        n_resueltos = int(row["tickets_soporte_resueltos"])
        sev_obj     = row["severidad_prom_tickets"]
        tiempo_obj  = row["tiempo_resolucion_prom_hrs"]
        mes_dt      = row["mes_reporte"]
        mid         = row["merchant_id"]
        es_churner  = row["abandono_30d"] == 1
        mes_decay   = es_churner and mes_dt in meses_decay_churner
        tipo_neg    = row["tipo_negocio_desc"]

        # 1. Asignar categorías para cada ticket
        categorias_ticket = [
            asignar_categoria_con_contexto(es_churner, mes_decay, tipo_neg)
            for _ in range(n_abiertos)
        ]

        # 2. Generar severidades consistentes con el promedio
        severidades = generar_severidades_consistentes(n_abiertos, sev_obj,
                                                        categorias_ticket)

        # 3. Decidir qué tickets están resueltos (los primeros N_resueltos)
        #    El orden no importa semánticamente — sólo garantiza el conteo
        estados = ["resuelto"] * n_resueltos + \
                  list(np.random.choice(ESTADOS_NO_RESUELTOS,
                                        size=n_abiertos - n_resueltos,
                                        p=PESOS_NO_RESUELTOS,
                                        replace=True))
        np.random.shuffle(estados)

        # 4. Generar tiempos de resolución para los resueltos
        tiempos_res = generar_tiempos_resolucion_consistentes(n_resueltos, tiempo_obj)

        # 5. Para cada ticket, generar la fila
        tiempos_iter = iter(tiempos_res)
        dias_en_mes = 30  # usamos 30 como aproximación
        for i in range(n_abiertos):
            # Fecha de apertura: dentro del mes, uniforme
            hora = np.random.randint(7, 22)      # Los tickets se abren de 7am a 10pm
            minuto = np.random.randint(0, 60)
            dia_mes = np.random.randint(1, dias_en_mes + 1)
            try:
                fecha_apertura = mes_dt.replace(day=dia_mes, hour=hora, minute=minuto)
            except ValueError:
                # Por si el mes no tiene ese día (ej. febrero día 30)
                fecha_apertura = mes_dt.replace(day=28, hour=hora, minute=minuto)

            categoria = categorias_ticket[i]
            severidad = severidades[i]
            estado = estados[i]

            # Tiempo de resolución y fecha de cierre
            if estado == "resuelto":
                tiempo_res = next(tiempos_iter)
                fecha_cierre = fecha_apertura + timedelta(hours=tiempo_res)
                satisfaccion = generar_satisfaccion(es_churner, severidad, tiempo_res)
            else:
                tiempo_res = None
                fecha_cierre = None
                satisfaccion = None

            todas_filas.append({
                "ticket_id":                generar_ticket_id(mid, str(mes_dt.date()), i),
                "merchant_id":              mid,
                "fecha_apertura":           fecha_apertura,
                "fecha_cierre":             fecha_cierre,
                "categoria":                categoria,
                "severidad":                severidad,
                "estado":                   estado,
                "tiempo_resolucion_hrs":    tiempo_res,
                "satisfaccion_post_cierre": satisfaccion,
            })

    df_tickets = pd.DataFrame(todas_filas)
    # Ordenar por merchant_id y fecha_apertura para una tabla prolija
    df_tickets = df_tickets.sort_values(["merchant_id", "fecha_apertura"]).reset_index(drop=True)
    return df_tickets


# ============================================================
# 5. VALIDACIÓN
# ============================================================

def validar_dataset(df_tickets: pd.DataFrame,
                    df_performance: pd.DataFrame,
                    df_merchants: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"DATASET fact_support_tickets  —  {len(df_tickets):,} tickets")
    print("=" * 70)

    print("\n[1] Volumen general:")
    print(f"    Total tickets:           {len(df_tickets):,}")
    print(f"    Comercios con tickets:   {df_tickets['merchant_id'].nunique():,}")
    print(f"    Comercios sin tickets:   "
          f"{len(df_merchants) - df_tickets['merchant_id'].nunique():,}")
    print(f"    Tickets promedio/comercio con tickets: "
          f"{len(df_tickets) / df_tickets['merchant_id'].nunique():.2f}")

    print("\n[2] Distribución por categoría:")
    dist_cat = df_tickets["categoria"].value_counts(normalize=True).round(3)
    for cat, pct in dist_cat.items():
        bar = "█" * int(pct * 100)
        print(f"    {cat:25} {pct:.3f}  {bar}")

    print("\n[3] Distribución por estado:")
    print(df_tickets["estado"].value_counts(normalize=True).round(3).to_string())

    print("\n[4] Severidad:")
    print(f"    Media:    {df_tickets['severidad'].mean():.2f}")
    print(f"    Mediana:  {df_tickets['severidad'].median():.1f}")
    print("    Distribución 1-5:")
    print(df_tickets["severidad"].value_counts().sort_index().to_string())

    print("\n[5] Tiempo de resolución (solo resueltos):")
    resueltos = df_tickets[df_tickets["estado"] == "resuelto"]["tiempo_resolucion_hrs"]
    print(f"    Media:    {resueltos.mean():.1f}h")
    print(f"    Mediana:  {resueltos.median():.1f}h")
    print(f"    P90:      {resueltos.quantile(0.90):.1f}h")

    print("\n[6] Satisfacción (solo tickets resueltos con respuesta):")
    sat = df_tickets["satisfaccion_post_cierre"].dropna()
    print(f"    Tickets con encuesta respondida: {len(sat):,} "
          f"({len(sat)/len(df_tickets)*100:.1f}%)")
    print(f"    Satisfacción media: {sat.mean():.2f}")
    print("    Distribución:")
    print(sat.value_counts().sort_index().to_string())

    print("\n[7] Consistencia con Tabla 2 (CRÍTICO):")
    # Agregar tickets de Tabla 4 por mes y comparar con Tabla 2
    df_tickets["mes_reporte"] = pd.to_datetime(df_tickets["fecha_apertura"]).dt.to_period("M").dt.to_timestamp()

    agg_t4 = df_tickets.groupby(["merchant_id", "mes_reporte"]).agg(
        tickets_t4=("ticket_id", "count"),
        resueltos_t4=("estado", lambda x: (x == "resuelto").sum()),
        severidad_t4=("severidad", "mean"),
    ).reset_index()

    df_perf_copy = df_performance.copy()
    df_perf_copy["mes_reporte"] = pd.to_datetime(df_perf_copy["mes_reporte"])

    merged = df_perf_copy.merge(agg_t4, on=["merchant_id", "mes_reporte"], how="left").fillna(0)

    # Comparación
    dif_tickets = (merged["tickets_soporte_abiertos"] - merged["tickets_t4"]).abs()
    dif_resueltos = (merged["tickets_soporte_resueltos"] - merged["resueltos_t4"]).abs()
    print(f"    Desviación conteo tickets (debería ser 0): max={dif_tickets.max():.0f}, "
          f"media={dif_tickets.mean():.3f}")
    print(f"    Desviación conteo resueltos (debería ser 0): max={dif_resueltos.max():.0f}, "
          f"media={dif_resueltos.mean():.3f}")

    # Severidad (solo donde hay tickets)
    con_tickets = merged[merged["tickets_soporte_abiertos"] > 0]
    dif_sev = (con_tickets["severidad_prom_tickets"] - con_tickets["severidad_t4"]).abs()
    print(f"    Desviación severidad promedio: max={dif_sev.max():.2f}, "
          f"media={dif_sev.mean():.3f} (aceptable <1.0)")

    print("\n[8] Cruce con etiqueta de abandono — tickets por churner vs sano:")
    df_tk_m = df_tickets.merge(df_merchants[["merchant_id", "abandono_30d"]],
                                on="merchant_id")
    comp = df_tk_m.groupby("abandono_30d").agg(
        total_tickets=("ticket_id", "count"),
        severidad_media=("severidad", "mean"),
        pct_resueltos=("estado", lambda x: (x == "resuelto").mean()),
    ).round(3)
    comp["tickets_por_comercio"] = (comp["total_tickets"] /
                                     df_merchants.groupby("abandono_30d").size()).round(2)
    print(comp.to_string())

    print("\n[9] Checks de integridad:")
    print(f"    ticket_id únicos:              {df_tickets['ticket_id'].is_unique}")
    print(f"    Todo merchant_id existe:        "
          f"{df_tickets['merchant_id'].isin(df_merchants['merchant_id']).all()}")
    print(f"    fecha_cierre >= fecha_apertura (donde aplica): "
          f"{(df_tickets[df_tickets['estado']=='resuelto']['fecha_cierre'] >= df_tickets[df_tickets['estado']=='resuelto']['fecha_apertura']).all()}")
    print(f"    Severidad en [1, 5]:            "
          f"{df_tickets['severidad'].between(1, 5).all()}")
    print(f"    Resueltos tienen tiempo/cierre: "
          f"{df_tickets[df_tickets['estado']=='resuelto']['tiempo_resolucion_hrs'].notna().all()}")
    print(f"    No-resueltos no tienen cierre:  "
          f"{df_tickets[df_tickets['estado']!='resuelto']['fecha_cierre'].isna().all()}")

    print("\n[10] Muestra de 5 tickets:")
    cols_show = ["ticket_id", "merchant_id", "fecha_apertura", "categoria",
                 "severidad", "estado", "tiempo_resolucion_hrs", "satisfaccion_post_cierre"]
    print(df_tickets[cols_show].head().to_string(index=False))


# ============================================================
# 6. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    # 1. Cargar tablas previas
    merchants_path = PATHS.RAW_DIR / "dim_merchants_con_abandono.csv"
    performance_path = PATHS.RAW_DIR / "fact_performance_monthly.csv"

    if not merchants_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {merchants_path}. "
            "Ejecuta primero generar_fact_performance.py."
        )
    if not performance_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {performance_path}. "
            "Ejecuta primero generar_fact_performance.py."
        )

    df_merchants = pd.read_csv(merchants_path)
    df_performance = pd.read_csv(performance_path)
    print(f"✓ Cargado {merchants_path.name}: {len(df_merchants):,} comercios")
    print(f"✓ Cargado {performance_path.name}:   {len(df_performance):,} filas\n")

    # 2. Generar Tabla 4
    print("Generando tickets de soporte individuales...")
    df_tickets = generar_fact_support_tickets(df_performance, df_merchants)
    print(f"✓ Generados {len(df_tickets):,} tickets individuales\n")

    # 3. Validar
    validar_dataset(df_tickets, df_performance, df_merchants)

    # 4. Guardar (sin la columna auxiliar mes_reporte que añadimos para validar)
    output_dir = PATHS.RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tickets_csv_path = output_dir / "fact_support_tickets.csv"
    tickets_parquet_path = output_dir / "fact_support_tickets.parquet"

    df_tickets_out = df_tickets.drop(columns=["mes_reporte"], errors="ignore")
    df_tickets_out.to_csv(tickets_csv_path, index=False, encoding="utf-8")
    try:
        df_tickets_out.to_parquet(tickets_parquet_path, index=False)
        print(f"\n✓ Guardado: {tickets_csv_path.name} + parquet")
    except Exception as e:
        print(f"\n✓ Guardado: {tickets_csv_path.name} (parquet no disponible: {e})")