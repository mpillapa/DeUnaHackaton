"""
====================================================================
 Generador de Datos Sintéticos — Reto Deuna (Interact2Hack 2026)
 TABLA 2: fact_performance_monthly (Performance Histórica Mensual)
====================================================================

 Genera 12 meses de historia transaccional para los 2000 comercios
 creados en Tabla 1. Es el corazón del dataset porque aquí se define:

 - La "salud latente" (variable oculta que nunca se expone)
 - La etiqueta ground truth de abandono (~15% de los comercios)
 - Decaimiento exponencial realista para los churners
 - Estacionalidad anclada a datos reales del SRI Ecuador:
     * Diciembre +43% (décimo tercer sueldo)
     * Enero -20% (contracción post-navideña)
     * Agosto/Marzo +5% (décimo cuarto sueldo)
 - Tickets de soporte correlacionados con fricción técnica
 - Tasa de rechazo calibrada a datos reales de QR Ecuador/LatAm

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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PATHS

# ============================================================
# 1. CONFIGURACIÓN GLOBAL Y REPRODUCIBILIDAD
# ============================================================
SEED = 42
np.random.seed(SEED)

# Ventana del dataset: 12 meses terminando en marzo 2026 (cierre de mes más reciente)
FECHA_FIN = datetime(2026, 3, 1)          # último mes reportado
MESES_HISTORIA = 12
FECHA_INICIO = FECHA_FIN - relativedelta(months=MESES_HISTORIA - 1)  # 2025-04-01

# Tasa objetivo de abandono (ground truth): entre 10% y 15%
# El enunciado del reto orienta a esta magnitud.
TASA_CHURN_OBJETIVO = 0.13

# ============================================================
# 2. CATÁLOGOS DE CALIBRACIÓN (anclados a datos reales)
# ============================================================

# --- Estacionalidad Ecuador (multiplicadores por mes) ---
# Fuente: SRI — ventas mensuales promedio $12.7B vs diciembre $18.25B (+43%).
# Enero con contracción del 20% ("ya no hay dinero" - economistas ecuatorianos).
# Décimo cuarto sueldo: Costa en marzo, Sierra/Amazonía en agosto.
ESTACIONALIDAD = {
    1:  0.82,   # Enero  — contracción post-navideña fuerte
    2:  0.90,   # Febrero — aún contraído
    3:  1.05,   # Marzo — décimo cuarto en Costa
    4:  1.00,   # Abril — base
    5:  1.02,   # Mayo — Día de la Madre
    6:  1.00,   # Junio — base
    7:  0.98,   # Julio — vacaciones sierra
    8:  1.05,   # Agosto — décimo cuarto en Sierra/Amazonía
    9:  1.00,   # Septiembre — regreso a clases
    10: 1.00,   # Octubre — base
    11: 1.10,   # Noviembre — Black Friday, pre-navidad
    12: 1.43,   # Diciembre — décimo tercer sueldo (dato real SRI)
}

# --- Escala transaccional por segmento ---
# Un Micro típico hace ~100-300 trx/mes; un Grande ~5000-15000.
# Fuente: cifras públicas de Deuna (~520k comercios) y ticket promedio del sector.
ESCALA_POR_SEGMENTO = {
    "Micro":   {"trx_base": 180,  "trx_sigma": 0.45},
    "Pequeña": {"trx_base": 650,  "trx_sigma": 0.40},
    "Mediana": {"trx_base": 2500, "trx_sigma": 0.35},
    "Grande":  {"trx_base": 8000, "trx_sigma": 0.30},
}

# --- Parámetros de tickets de soporte ---
# Modelados como Poisson. Comercios sanos: lambda ~0.2/mes (la mayoría = 0 tickets).
# Comercios con fricción técnica: lambda 0.8-1.5/mes.
# Churners en meses finales: lambda 3-5/mes luego cae a 0 (dejan de reportar).
SOPORTE_LAMBDA_SANO = 0.20
SOPORTE_LAMBDA_PROBLEMATICO = 1.20

# --- Categorías de tickets por sector (para severidad promedio) ---
# Algunos sectores tienen incidencias técnicas más severas (ej: integraciones
# complejas en restaurantes con sistemas POS vs tiendas con QR impreso).
SEVERIDAD_POR_TIPO = {
    "Restaurantes y picanterías":       3.2,
    "Comida rápida y food trucks":      2.8,
    "Bares y cantinas":                 3.0,
    "Venta y repuestos de motos":       3.5,
    "Farmacias y artículos médicos":    2.5,
    # Default para el resto
    "_default":                         2.2,
}

# ============================================================
# 3. ASIGNACIÓN DE SALUD LATENTE Y ETIQUETA DE ABANDONO
# ============================================================

def calcular_salud_latente(row: pd.Series) -> float:
    """
    Calcula una variable oculta de 'salud' del comercio (0 a 1)
    basada en características de dim_merchants.
    Esta variable nunca se expone en el dataset final — es solo
    el ancla generativa (como dice el documento de arquitectura).

    Factores:
    - Tenure (antigüedad): comercios maduros = más salud (correlación inversa con churn)
    - Región: Pichincha/Guayas con mejor infraestructura → más salud
    - Segmento: Mediana/Grande son más estables que Micro
    - Tipo de negocio: alimentos/farmacia son más resilientes que moda
    """
    score = 0.5  # base

    # Tenure: meses desde onboarding hasta FECHA_FIN
    tenure_dias = (FECHA_FIN - pd.to_datetime(row["fecha_onboarding"])).days
    tenure_meses = tenure_dias / 30
    if tenure_meses > 24:
        score += 0.20
    elif tenure_meses > 12:
        score += 0.10
    elif tenure_meses < 3:
        score -= 0.15    # fase crítica de deserción temprana

    # Región
    if row["region"] in ("Pichincha", "Guayas"):
        score += 0.08
    elif row["region"] in ("Galápagos", "Morona Santiago", "Napo",
                           "Pastaza", "Zamora Chinchipe", "Orellana"):
        score -= 0.10   # infraestructura digital más débil

    # Segmento
    if row["segmento_comercial"] in ("Grande", "Mediana"):
        score += 0.10
    elif row["segmento_comercial"] == "Micro":
        score -= 0.03

    # Tipo de negocio (algunos son más resilientes)
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

    # Ruido aleatorio para no hacer el churn 100% determinístico
    score += np.random.normal(0, 0.08)

    return float(np.clip(score, 0.01, 0.99))


def asignar_abandono(df_merchants: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna etiqueta 'abandono_30d' respetando la tasa objetivo (~13%).
    Los comercios con menor salud latente son los candidatos.
    """
    df = df_merchants.copy()
    np.random.seed(SEED)

    # 1. Calcular salud latente para cada comercio
    df["_salud_latente"] = df.apply(calcular_salud_latente, axis=1)

    # 2. Los comercios con salud más baja son los más probables de abandonar.
    #    Usamos la salud invertida (1 - salud) como probabilidad base,
    #    y calibramos para hit la tasa objetivo.
    prob_base = (1 - df["_salud_latente"]) ** 3  # elevamos al cubo para acentuar cola

    # Normalizar para que la suma sea exactamente TASA_CHURN_OBJETIVO * N
    n_churners_objetivo = int(len(df) * TASA_CHURN_OBJETIVO)
    # Muestreamos SIN reemplazo usando prob_base como peso
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
    """
    Genera las 12 filas mensuales para un comercio específico.
    La lógica diverge fuertemente según sea churner (abandono_30d=1) o sano.
    """
    n_meses = len(meses_fechas)
    salud = merchant_row["_salud_latente"]
    es_churner = merchant_row["abandono_30d"] == 1
    segmento = merchant_row["segmento_comercial"]
    tipo_desc = merchant_row["tipo_negocio_desc"]
    fecha_onboard = pd.to_datetime(merchant_row["fecha_onboarding"])

    # --- Parámetros base del comercio ---
    escala = ESCALA_POR_SEGMENTO[segmento]
    trx_base_mes = escala["trx_base"]

    # Los comercios con mejor salud tienen ligeramente más trx base
    trx_base_mes *= (0.7 + 0.6 * salud)

    # Ticket promedio desde catálogo CIIU de Tabla 1
    # Como no tenemos el ticket_base aquí, lo derivamos del tipo_desc
    ticket_base = _ticket_base_por_tipo(tipo_desc)

    # --- Generar trayectoria de count_trx para los 12 meses ---
    inicio_decaimiento = n_meses
    if es_churner:
        # Decaimiento exponencial en los últimos 3-5 meses
        meses_decaimiento = np.random.randint(3, 6)   # entre 3 y 5 meses de caída
        inicio_decaimiento = n_meses - meses_decaimiento
        # Factor de decaimiento por mes (entre 0.4 y 0.7 por mes = caída fuerte)
        factor_decay = np.random.uniform(0.40, 0.70)

        count_trx_trayectoria = []
        for i in range(n_meses):
            if i < inicio_decaimiento:
                # Aún activo y con variación estacional normal
                mes_num = meses_fechas[i].month
                factor = ESTACIONALIDAD[mes_num]
                val = trx_base_mes * factor * np.random.lognormal(0, 0.15)
            else:
                # Decaimiento: cada mes es factor_decay del mes anterior
                meses_dentro_decay = i - inicio_decaimiento + 1
                base_decay = count_trx_trayectoria[inicio_decaimiento - 1]
                val = base_decay * (factor_decay ** meses_dentro_decay)
                # Añadir algo de ruido para que no sea matemáticamente perfecto
                val *= np.random.uniform(0.7, 1.1)
            count_trx_trayectoria.append(max(0, val))
    else:
        # Comercio sano: estacionalidad + crecimiento ligero + ruido
        # Tendencia de crecimiento: los comercios sanos jóvenes crecen más rápido
        tenure_meses_inicio = (meses_fechas[0] - fecha_onboard).days / 30
        if tenure_meses_inicio < 6:
            # Comercio en ramp-up fuerte
            tendencia_mensual = np.random.uniform(1.03, 1.10)
        elif tenure_meses_inicio < 18:
            tendencia_mensual = np.random.uniform(1.01, 1.04)
        else:
            tendencia_mensual = np.random.uniform(0.99, 1.02)  # maduro, estable

        count_trx_trayectoria = []
        base_actual = trx_base_mes
        for i in range(n_meses):
            mes_num = meses_fechas[i].month
            factor = ESTACIONALIDAD[mes_num]
            val = base_actual * factor * np.random.lognormal(0, 0.18)
            count_trx_trayectoria.append(max(0, val))
            base_actual *= tendencia_mensual

    # --- Para cada mes, derivar las demás métricas ---
    filas = []
    for i, mes_dt in enumerate(meses_fechas):
        # Si el comercio aún no estaba onboardeado ese mes, no generar fila
        if mes_dt < fecha_onboard.replace(day=1):
            continue

        count_trx = int(round(count_trx_trayectoria[i]))

        # --- TPV mensual ---
        # Ticket promedio del mes varía alrededor del ticket_base del sector
        # (ligeramente más alto en diciembre por regalos, más bajo en enero)
        factor_ticket_mes = 1 + (ESTACIONALIDAD[mes_dt.month] - 1) * 0.3
        ticket_mes = ticket_base * factor_ticket_mes * np.random.lognormal(0, 0.10)
        tpv_mes = count_trx * ticket_mes

        # --- Tasa de rechazo ---
        # Sanos: beta(2, 50) → media ~4%, rango realista 0-15%
        # Churners en meses finales: sube progresivamente hasta 20-25%
        if es_churner and i >= inicio_decaimiento:
            meses_dentro_decay = i - inicio_decaimiento + 1
            tasa_rechazo_base = 0.04 + 0.04 * meses_dentro_decay
            tasa_rechazo = min(0.30, tasa_rechazo_base + np.random.uniform(0, 0.05))
        else:
            # Comercios sanos: tasa baja con variación natural
            tasa_rechazo = np.random.beta(2, 50)  # media ~3.8%
            # Comercios con mala salud sin llegar a churner: tasa más alta
            if salud < 0.35:
                tasa_rechazo += np.random.uniform(0.02, 0.05)
            tasa_rechazo = min(0.15, tasa_rechazo)

        # --- Días sin transacción (racha máxima dentro del mes) ---
        dias_en_mes = 30
        if count_trx == 0:
            dias_sin_trx_max = dias_en_mes
            dias_desde_ultima = dias_en_mes
        else:
            # Frecuencia promedio de transacciones por día
            trx_por_dia = count_trx / dias_en_mes
            # La racha máxima sigue una distribución exponencial
            # que depende de la frecuencia: cuanto más alta la frecuencia, menos racha
            lambda_racha = max(trx_por_dia, 0.3)
            dias_sin_trx_max = int(min(dias_en_mes,
                                       np.random.exponential(scale=3/lambda_racha)))
            # Días desde última: entre 0 y racha_max
            dias_desde_ultima = int(np.random.uniform(0, max(1, dias_sin_trx_max)))

        # Para churners en últimos meses, forzar que los días sin trx suban mucho
        if es_churner and i >= inicio_decaimiento:
            meses_dentro_decay = i - inicio_decaimiento + 1
            dias_sin_trx_max = min(30, max(dias_sin_trx_max,
                                           int(5 * meses_dentro_decay + np.random.uniform(0, 5))))
            if i == n_meses - 1:  # último mes del churner
                dias_desde_ultima = min(30, max(dias_desde_ultima, 15))

        # --- Tickets de soporte ---
        if es_churner:
            # En la fase pre-decay, soporte normal
            # En los meses de decay, pico de tickets → luego silencio
            if i < inicio_decaimiento - 1:
                lambda_ticket = SOPORTE_LAMBDA_SANO
            elif i < n_meses - 1:
                # Pico de quejas justo antes de darse de baja
                lambda_ticket = 3.0 + 1.5 * (i - inicio_decaimiento + 1)
            else:
                # Último mes: silencio (ya se resignaron)
                lambda_ticket = 0.3
        else:
            # Comercios sanos: bajo nivel, un poco más si salud baja
            if salud < 0.4:
                lambda_ticket = SOPORTE_LAMBDA_PROBLEMATICO
            else:
                lambda_ticket = SOPORTE_LAMBDA_SANO

        tickets_abiertos = np.random.poisson(lambda_ticket)
        # Los resueltos son un % de los abiertos (en churners, se resuelven menos)
        if es_churner and i >= inicio_decaimiento:
            tasa_resolucion = np.random.uniform(0.30, 0.55)
        else:
            tasa_resolucion = np.random.uniform(0.75, 0.95)
        tickets_resueltos = int(tickets_abiertos * tasa_resolucion)

        # Tiempo de resolución (horas) — lognormal
        if tickets_resueltos > 0:
            if es_churner and i >= inicio_decaimiento:
                # SLAs largos en churners (falta de atención)
                tiempo_resolucion = np.random.lognormal(mean=np.log(96), sigma=0.6)
            else:
                tiempo_resolucion = np.random.lognormal(mean=np.log(36), sigma=0.5)
        else:
            tiempo_resolucion = 0.0

        # Severidad promedio de los tickets
        if tickets_abiertos > 0:
            sev_base = SEVERIDAD_POR_TIPO.get(tipo_desc, SEVERIDAD_POR_TIPO["_default"])
            if es_churner and i >= inicio_decaimiento:
                sev_base += np.random.uniform(0.5, 1.2)   # se quejan de cosas más graves
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
    """Mapeo inverso de descripción → ticket base (debe coincidir con Tabla 1)."""
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
# 5. ORQUESTADOR
# ============================================================

def generar_fact_performance_monthly(df_merchants: pd.DataFrame) -> pd.DataFrame:
    """Genera la tabla completa de performance mensual."""
    np.random.seed(SEED)

    # Lista de meses desde FECHA_INICIO hasta FECHA_FIN
    meses_fechas = [FECHA_INICIO + relativedelta(months=k)
                    for k in range(MESES_HISTORIA)]

    print(f"Generando {MESES_HISTORIA} meses de historia para {len(df_merchants)} comercios...")
    print(f"   Rango temporal: {meses_fechas[0].date()} → {meses_fechas[-1].date()}")

    todas_filas = []
    for idx, row in df_merchants.iterrows():
        filas_comercio = generar_trayectoria_mensual(row, meses_fechas)
        todas_filas.extend(filas_comercio)

    df_fact = pd.DataFrame(todas_filas)
    return df_fact


# ============================================================
# 6. VALIDACIÓN
# ============================================================

def validar_dataset(df_fact: pd.DataFrame, df_merchants: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"DATASET fact_performance_monthly  —  {len(df_fact):,} filas")
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
    print(f"    Ticket promedio coherente:     "
          f"{((df_fact['ticket_promedio'] >= 0) & (df_fact['ticket_promedio'] <= 500)).all()}")
    print(f"    tasa_rechazo en [0, 0.35]:     "
          f"{((df_fact['tasa_rechazo'] >= 0) & (df_fact['tasa_rechazo'] <= 0.35)).all()}")

    print("\n[8] Muestra de 5 filas:")
    print(df_fact.head().to_string())

    print("\n[9] Trayectoria ejemplar de un CHURNER (mostrando decaimiento):")
    churner_ejemplo = df_merchants[df_merchants["abandono_30d"] == 1].iloc[0]["merchant_id"]
    traj = df_fact[df_fact["merchant_id"] == churner_ejemplo].sort_values("mes_reporte")
    print(f"    Comercio: {churner_ejemplo}")
    print(traj[["mes_reporte", "count_trx", "tpv_mensual",
                "tasa_rechazo", "tickets_soporte_abiertos"]].to_string(index=False))


# ============================================================
# 7. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    # 1. Cargar Tabla 1
    input_merchants_path = PATHS.RAW_DIR / "dim_merchants.csv"
    if not input_merchants_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {input_merchants_path}. "
            "Ejecuta primero generar_dim_merchants.py."
        )
    df_merchants = pd.read_csv(input_merchants_path)
    print(f"✓ Cargado {input_merchants_path.name} — {len(df_merchants):,} comercios\n")

    # 2. Asignar salud latente y etiqueta de abandono
    df_merchants = asignar_abandono(df_merchants)
    print(f"✓ Asignada etiqueta de abandono — "
          f"{df_merchants['abandono_30d'].sum()} churners "
          f"({df_merchants['abandono_30d'].mean()*100:.1f}%)\n")

    # 3. Generar Tabla 2
    df_fact = generar_fact_performance_monthly(df_merchants)

    # 4. Validar
    validar_dataset(df_fact, df_merchants)

    # 5. Guardar
    output_dir = PATHS.RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fact_csv_path = output_dir / "fact_performance_monthly.csv"
    fact_parquet_path = output_dir / "fact_performance_monthly.parquet"
    merchants_out_path = output_dir / "dim_merchants_con_abandono.csv"

    df_fact.to_csv(fact_csv_path, index=False, encoding="utf-8")
    try:
        df_fact.to_parquet(fact_parquet_path, index=False)
        print(f"\n✓ Guardado: {fact_csv_path.name} + parquet")
    except Exception as e:
        print(f"\n✓ Guardado: {fact_csv_path.name} (parquet no disponible: {e})")

    # 6. Guardar también dim_merchants actualizado con la etiqueta ground truth
    #    (quitamos la variable oculta _salud_latente: nunca debe llegar al modelo)
    df_merchants_out = df_merchants.drop(columns=["_salud_latente"])
    df_merchants_out.to_csv(merchants_out_path, index=False, encoding="utf-8")
    print(f"✓ Guardado: {merchants_out_path.name} (con la columna abandono_30d añadida)")