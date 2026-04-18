"""
====================================================================
 Generador de Datos Sintéticos — Reto Deuna (Interact2Hack 2026)
 TABLA 1: dim_merchants (Dimensión Maestra de Comercios)
====================================================================

 Objetivo: generar 2000 comercios sintéticos con distribuciones
 ancladas a la realidad del mercado ecuatoriano (INEC, SRI, Deuna).

 Autor: Equipo Hackathon
 Fecha: Abril 2026
 Reproducibilidad: semilla fija = 42
====================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
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

N_COMERCIOS = 2000
FECHA_CORTE = datetime(2026, 3, 31)     # último día del último mes del dataset
FECHA_LANZAMIENTO_DEUNA = datetime(2020, 4, 1)  # lanzamiento aproximado

# ============================================================
# 2. CATÁLOGOS MAESTROS (basados en datos reales de Ecuador)
# ============================================================

# --- Segmento SRI/Superintendencia de Compañías ---
# Distribución basada en INEC: ~90% Micro, ~7% Pequeña, ~2.5% Mediana, ~0.5% Grande
SEGMENTOS = {
    "Micro":    0.900,   # Ventas hasta $100k, 1-9 trabajadores
    "Pequeña":  0.070,   # Ventas $100k-$1M, 10-49 trabajadores
    "Mediana":  0.025,   # Ventas $1M-$5M, 50-199 trabajadores
    "Grande":   0.005,   # Ventas > $5M, 200+ trabajadores
}

# --- Provincias de Ecuador con peso por dinamismo comercial ---
# Los pesos están calibrados al mix de empresas según DIEE/INEC.
# Pichincha + Guayas concentran más del 50% de comercios formales.
PROVINCIAS = {
    "Pichincha":        {"peso": 0.300, "ciudad": "Quito",       "lat": -0.1807, "lon": -78.4678},
    "Guayas":           {"peso": 0.250, "ciudad": "Guayaquil",   "lat": -2.1709, "lon": -79.9224},
    "Azuay":            {"peso": 0.080, "ciudad": "Cuenca",      "lat": -2.9001, "lon": -79.0059},
    "Manabí":           {"peso": 0.070, "ciudad": "Portoviejo",  "lat": -1.0546, "lon": -80.4525},
    "Tungurahua":       {"peso": 0.050, "ciudad": "Ambato",      "lat": -1.2490, "lon": -78.6167},
    "El Oro":           {"peso": 0.040, "ciudad": "Machala",     "lat": -3.2581, "lon": -79.9554},
    "Imbabura":         {"peso": 0.035, "ciudad": "Ibarra",      "lat":  0.3517, "lon": -78.1223},
    "Loja":             {"peso": 0.030, "ciudad": "Loja",        "lat": -4.0079, "lon": -79.2113},
    "Chimborazo":       {"peso": 0.025, "ciudad": "Riobamba",    "lat": -1.6635, "lon": -78.6547},
    "Los Ríos":         {"peso": 0.025, "ciudad": "Babahoyo",    "lat": -1.8019, "lon": -79.5342},
    "Santo Domingo":    {"peso": 0.025, "ciudad": "Santo Domingo","lat": -0.2524, "lon": -79.1713},
    "Esmeraldas":       {"peso": 0.020, "ciudad": "Esmeraldas",  "lat":  0.9592, "lon": -79.6539},
    "Cotopaxi":         {"peso": 0.020, "ciudad": "Latacunga",   "lat": -0.9354, "lon": -78.6155},
    "Santa Elena":      {"peso": 0.015, "ciudad": "Santa Elena", "lat": -2.2267, "lon": -80.8583},
    "Cañar":            {"peso": 0.010, "ciudad": "Azogues",     "lat": -2.7386, "lon": -78.8482},
    "Carchi":           {"peso": 0.008, "ciudad": "Tulcán",      "lat":  0.8120, "lon": -77.7181},
    "Bolívar":          {"peso": 0.008, "ciudad": "Guaranda",    "lat": -1.5905, "lon": -79.0030},
    "Sucumbíos":        {"peso": 0.007, "ciudad": "Nueva Loja",  "lat":  0.0844, "lon": -76.8889},
    "Orellana":         {"peso": 0.006, "ciudad": "Coca",        "lat": -0.4672, "lon": -76.9857},
    "Morona Santiago":  {"peso": 0.005, "ciudad": "Macas",       "lat": -2.3085, "lon": -78.1198},
    "Napo":             {"peso": 0.005, "ciudad": "Tena",        "lat": -0.9935, "lon": -77.8152},
    "Pastaza":          {"peso": 0.004, "ciudad": "Puyo",        "lat": -1.4836, "lon": -77.9928},
    "Zamora Chinchipe": {"peso": 0.004, "ciudad": "Zamora",      "lat": -4.0683, "lon": -78.9567},
    "Galápagos":        {"peso": 0.003, "ciudad": "Puerto Ayora","lat": -0.7436, "lon": -90.3134},
}

# Normalizar pesos por si no suman exactamente 1 (por redondeo)
_total_peso = sum(p["peso"] for p in PROVINCIAS.values())
for p in PROVINCIAS.values():
    p["peso"] /= _total_peso

# --- Tipos de negocio CIIU ---
# Foco en sectores que realmente usan pagos QR en Ecuador.
# Las descripciones son simplificadas; los códigos son reales de CIIU Rev.4 del INEC.
TIPOS_NEGOCIO = {
    "G4711.01": {"desc": "Tiendas de abarrotes y víveres",    "peso": 0.22, "ticket_base": 8,    "volatilidad": 0.15},
    "I5610.01": {"desc": "Restaurantes y picanterías",         "peso": 0.18, "ticket_base": 12,   "volatilidad": 0.20},
    "I5610.02": {"desc": "Comida rápida y food trucks",        "peso": 0.08, "ticket_base": 6,    "volatilidad": 0.15},
    "I5630.01": {"desc": "Cafeterías y panaderías",            "peso": 0.07, "ticket_base": 5,    "volatilidad": 0.12},
    "G4774.01": {"desc": "Farmacias y artículos médicos",      "peso": 0.06, "ticket_base": 18,   "volatilidad": 0.25},
    "S9602.01": {"desc": "Peluquerías y salones de belleza",   "peso": 0.06, "ticket_base": 15,   "volatilidad": 0.20},
    "G4771.01": {"desc": "Boutiques y ropa",                   "peso": 0.05, "ticket_base": 28,   "volatilidad": 0.35},
    "G4721.01": {"desc": "Fruterías y mercados",               "peso": 0.05, "ticket_base": 7,    "volatilidad": 0.18},
    "H4922.01": {"desc": "Transporte (taxis, cooperativas)",   "peso": 0.04, "ticket_base": 4,    "volatilidad": 0.30},
    "G4773.01": {"desc": "Bazares y papelerías",               "peso": 0.04, "ticket_base": 6,    "volatilidad": 0.22},
    "C1071.01": {"desc": "Panaderías y pastelerías",           "peso": 0.04, "ticket_base": 5,    "volatilidad": 0.10},
    "G4719.01": {"desc": "Comercio mixto (bazar + abarrote)",  "peso": 0.03, "ticket_base": 9,    "volatilidad": 0.20},
    "I5610.03": {"desc": "Bares y cantinas",                   "peso": 0.03, "ticket_base": 16,   "volatilidad": 0.30},
    "S9601.01": {"desc": "Lavanderías y tintorerías",          "peso": 0.02, "ticket_base": 8,    "volatilidad": 0.15},
    "G4541.01": {"desc": "Venta y repuestos de motos",         "peso": 0.02, "ticket_base": 45,   "volatilidad": 0.50},
    "G4761.01": {"desc": "Librerías y útiles escolares",       "peso": 0.01, "ticket_base": 12,   "volatilidad": 0.40},
}

# Normalizar pesos de tipos de negocio
_total_tn = sum(t["peso"] for t in TIPOS_NEGOCIO.values())
for t in TIPOS_NEGOCIO.values():
    t["peso"] /= _total_tn

# --- Ejecutivos comerciales (KAM/CSM) ---
# Simulamos un equipo de 15 ejecutivos. Los comercios Grandes y Medianos
# tienden a tener KAMs asignados; los Micro muchas veces son auto-gestionados.
EJECUTIVOS = [
    "Andrea Vásconez", "Carlos Mendoza", "Diana Cárdenas", "Esteban Ruiz",
    "Fernanda Ortiz", "Gabriel Aguilar", "Hilda Proaño", "Iván Moreno",
    "Jimena Salazar", "Kevin Jaramillo", "Lorena Dávila", "Mateo Pérez",
    "Natalia Rosales", "Oswaldo Ñacato", "Paola Intriago",
]

# ============================================================
# 3. FUNCIONES DE SÍNTESIS
# ============================================================

def generar_merchant_id(i: int) -> str:
    """Hash determinístico corto a partir del índice. Formato: DEU-XXXXXXXX."""
    h = hashlib.md5(f"merchant_{i}_{SEED}".encode()).hexdigest()[:8].upper()
    return f"DEU-{h}"

def generar_nombre_comercio(tipo_desc: str, i: int) -> str:
    """
    Nombre plausible combinando prefijos comunes en Ecuador con el tipo de negocio.
    No usamos Faker aquí para mantener el flavor local.
    """
    prefijos_tienda = ["Don", "Doña", "Mi", "La", "El", "Tía", "Tío", "Super"]
    nombres_persona = [
        "Juanito", "Carmita", "Lucho", "Rosita", "Pedrito", "Manuelito",
        "Martita", "Segundo", "Anita", "Pancho", "Chabela", "Kléber",
        "Jorgito", "Nelly", "Beto", "Fanny", "Mayra", "Gladys",
    ]
    sufijos = ["Express", "Plus", "", "", "", "Center", "Mini", "Market"]

    # Elección pseudo-aleatoria determinística por índice
    rng = np.random.default_rng(i + 1000)
    if "Restaurante" in tipo_desc or "Cafetería" in tipo_desc or "Bar" in tipo_desc:
        base = f"{rng.choice(prefijos_tienda)} {rng.choice(nombres_persona)}"
    elif "Boutique" in tipo_desc or "ropa" in tipo_desc.lower():
        base = f"Boutique {rng.choice(nombres_persona)}"
    elif "Farmacia" in tipo_desc:
        base = f"Farmacia {rng.choice(['Sana', 'Vida', 'Salud', 'Central', 'del Barrio'])}"
    elif "Transporte" in tipo_desc:
        base = f"Coop. {rng.choice(['Trans Vencedores', 'Los Andes', '24 de Mayo', 'Reina del Camino'])}"
    else:
        base = f"{rng.choice(prefijos_tienda)} {rng.choice(nombres_persona)}"

    suf = rng.choice(sufijos)
    return f"{base} {suf}".strip()

def asignar_segmento_con_correlacion(region: str) -> str:
    """
    La distribución del segmento depende ligeramente de la provincia:
    Pichincha y Guayas tienen mayor proporción de Pequeñas/Medianas/Grandes;
    provincias amazónicas e insulares son casi 100% Micro.
    """
    base = list(SEGMENTOS.values())  # [Micro, Pequeña, Mediana, Grande]

    if region in ("Pichincha", "Guayas"):
        # Aumenta la probabilidad de segmentos superiores
        pesos = [0.830, 0.115, 0.045, 0.010]
    elif region in ("Azuay", "Manabí", "Tungurahua", "El Oro"):
        pesos = [0.880, 0.085, 0.030, 0.005]
    elif region in ("Galápagos", "Morona Santiago", "Napo", "Pastaza",
                    "Zamora Chinchipe", "Orellana", "Sucumbíos"):
        pesos = [0.970, 0.025, 0.005, 0.000]
    else:
        pesos = base

    pesos = np.array(pesos) / sum(pesos)
    return np.random.choice(list(SEGMENTOS.keys()), p=pesos)

def asignar_fecha_onboarding() -> datetime:
    """
    Distribución sesgada hacia años recientes. Deuna creció fuerte en 2023-2025.
    - 2020-2021: ~10% del parque actual (early adopters)
    - 2022: ~15%
    - 2023: ~25%
    - 2024: ~30%
    - 2025-2026-Q1: ~20%
    """
    r = np.random.random()
    if r < 0.10:
        start = datetime(2020, 4, 1); end = datetime(2021, 12, 31)
    elif r < 0.25:
        start = datetime(2022, 1, 1); end = datetime(2022, 12, 31)
    elif r < 0.50:
        start = datetime(2023, 1, 1); end = datetime(2023, 12, 31)
    elif r < 0.80:
        start = datetime(2024, 1, 1); end = datetime(2024, 12, 31)
    else:
        start = datetime(2025, 1, 1); end = datetime(2026, 3, 1)

    delta_dias = (end - start).days
    return start + timedelta(days=int(np.random.random() * delta_dias))

def asignar_ejecutivo(segmento: str) -> str:
    """
    Segmentos grandes tienen KAM asignado; los Micro frecuentemente no.
    Devolvemos cadena vacía para Micro que son auto-gestionados (~70% de los Micro).
    """
    if segmento == "Micro" and np.random.random() < 0.70:
        return "Auto-gestionado"
    return np.random.choice(EJECUTIVOS)

def ruido_geografico(lat: float, lon: float, segmento: str) -> tuple:
    """
    Agregamos ruido a las coordenadas para dispersar los comercios
    dentro del área metropolitana. Los Micro se dispersan más (barrios periféricos),
    los Grandes tienden al centro comercial.
    """
    if segmento == "Micro":
        sigma = 0.08
    elif segmento == "Pequeña":
        sigma = 0.05
    else:
        sigma = 0.025
    return (
        round(lat + np.random.normal(0, sigma), 6),
        round(lon + np.random.normal(0, sigma), 6),
    )

# ============================================================
# 4. GENERACIÓN DE LA TABLA
# ============================================================

def generar_dim_merchants(n: int = N_COMERCIOS) -> pd.DataFrame:
    """Genera el DataFrame dim_merchants con n filas."""
    np.random.seed(SEED)  # re-fijar por si se llama múltiples veces

    # Pre-calcular arreglos de selección ponderada
    provincias_keys = list(PROVINCIAS.keys())
    provincias_pesos = [PROVINCIAS[k]["peso"] for k in provincias_keys]

    tipos_keys = list(TIPOS_NEGOCIO.keys())
    tipos_pesos = [TIPOS_NEGOCIO[k]["peso"] for k in tipos_keys]

    registros = []
    for i in range(n):
        # 1. Región (muestreo ponderado)
        region = np.random.choice(provincias_keys, p=provincias_pesos)
        meta_region = PROVINCIAS[region]

        # 2. Segmento (condicionado por región)
        segmento = asignar_segmento_con_correlacion(region)

        # 3. Tipo de negocio
        ciiu = np.random.choice(tipos_keys, p=tipos_pesos)
        tipo_desc = TIPOS_NEGOCIO[ciiu]["desc"]

        # 4. Coordenadas con ruido
        lat, lon = ruido_geografico(meta_region["lat"], meta_region["lon"], segmento)

        # 5. Onboarding
        fecha_on = asignar_fecha_onboarding()

        # 6. Fecha última transacción — por ahora, simulamos que todos transaccionaron
        #    al menos una vez en el mes de corte. La Tabla 2 ajustará esto realistamente.
        #    Provisional: fecha aleatoria entre hace 90 días y la fecha de corte.
        dias_atras = np.random.exponential(scale=15)
        dias_atras = min(int(dias_atras), 150)
        fecha_ultima = FECHA_CORTE - timedelta(days=dias_atras)
        if fecha_ultima < fecha_on:
            fecha_ultima = fecha_on + timedelta(days=int(np.random.random() * 30))

        # 7. Ejecutivo
        ejecutivo = asignar_ejecutivo(segmento)

        # 8. Nombre
        nombre = generar_nombre_comercio(tipo_desc, i)

        # 9. Merchant ID
        mid = generar_merchant_id(i)

        registros.append({
            "merchant_id":              mid,
            "nombre_comercio":          nombre,
            "segmento_comercial":       segmento,
            "tipo_negocio_ciiu":        ciiu,
            "tipo_negocio_desc":        tipo_desc,
            "fecha_onboarding":         fecha_on.date(),
            "fecha_ultima_transaccion": fecha_ultima.date(),
            "region":                   region,
            "ciudad":                   meta_region["ciudad"],
            "latitud":                  lat,
            "longitud":                 lon,
            "ejecutivo_cuenta":         ejecutivo,
        })

    df = pd.DataFrame(registros)
    return df

# ============================================================
# 5. VALIDACIÓN DE CALIDAD DEL DATASET
# ============================================================

def validar_dataset(df: pd.DataFrame) -> None:
    """Imprime chequeos de sanidad para verificar que el dataset es realista."""
    print("=" * 70)
    print(f"DATASET dim_merchants  —  {len(df):,} comercios generados")
    print("=" * 70)

    print("\n[1] Distribución por Segmento Comercial (esperado ~90% Micro):")
    print(df["segmento_comercial"].value_counts(normalize=True).round(3).to_string())

    print("\n[2] Top 10 provincias (esperado Pichincha + Guayas ~55%):")
    print(df["region"].value_counts(normalize=True).head(10).round(3).to_string())

    print("\n[3] Top 8 tipos de negocio (esperado dominio de abarrotes + restaurantes):")
    print(df["tipo_negocio_desc"].value_counts(normalize=True).head(8).round(3).to_string())

    print("\n[4] Distribución de onboarding por año:")
    df_tmp = df.copy()
    df_tmp["anio_onboarding"] = pd.to_datetime(df_tmp["fecha_onboarding"]).dt.year
    print(df_tmp["anio_onboarding"].value_counts().sort_index().to_string())

    print("\n[5] Cruce Segmento × Región (top provincias):")
    top_regiones = df["region"].value_counts().head(5).index
    tabla = pd.crosstab(
        df[df["region"].isin(top_regiones)]["region"],
        df[df["region"].isin(top_regiones)]["segmento_comercial"],
        normalize="index",
    ).round(3)
    print(tabla.to_string())

    print("\n[6] Checks de integridad:")
    print(f"    - merchant_id únicos:            {df['merchant_id'].is_unique}")
    print(f"    - Sin nulos:                     {not df.isnull().any().any()}")
    print(f"    - fecha_ultima >= fecha_onboard: "
          f"{(pd.to_datetime(df['fecha_ultima_transaccion']) >= pd.to_datetime(df['fecha_onboarding'])).all()}")
    print(f"    - Latitudes en rango Ecuador:    "
          f"{(df['latitud'].between(-5.0, 1.5)).all()}")
    print(f"    - Longitudes en rango Ecuador:   "
          f"{(df['longitud'].between(-92, -75)).all()}")

    print("\n[7] Primeros 5 registros de muestra:")
    print(df.head().to_string())

# ============================================================
# 6. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    df = generar_dim_merchants()
    validar_dataset(df)

    # Guardar a CSV y Parquet dentro de data/raw
    output_dir = PATHS.RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dim_merchants.csv"
    parquet_path = output_dir / "dim_merchants.parquet"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"\n✓ Archivos guardados: {csv_path} y {parquet_path}")
    except Exception as e:
        print(f"\n✓ Archivo guardado: {csv_path} (parquet no disponible: {e})")