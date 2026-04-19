import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import textwrap
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="D'Una Churn Intelligence", 
    page_icon="💸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores para niveles de riesgo (Ajustada para vibrar en fondos oscuros)
RISK_COLORS = {
    "Crítico": "#EF4444",   # Rojo vibrante
    "Alto": "#F97316",      # Naranja de alto contraste
    "Medio": "#FBBF24",     # Amarillo brillante
    "Bajo": "#10B981"       # Verde esmeralda
}

# CSS Personalizado (Estilo Dark Mode Minimalista con Acentos Naranjas)
st.markdown("""
<style>
    /* Ocultar elementos predeterminados de Streamlit para un look más inmersivo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Fondo principal y tipografía general */
    .stApp {
        background-color: #0B0F19; /* Fondo oscuro profundo */
        font-family: 'Inter', sans-serif;
    }
    
    /* Estilo de las tarjetas de métricas nativas (si usas st.metric) */
    div[data-testid="metric-container"] {
        background-color: #1E293B; /* Tarjeta oscura elegante */
        border: 1px solid #334155;
        border-left: 4px solid #F97316; /* Acento naranja lateral */
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important; /* Texto secundario en gris platinado */
        font-weight: 600;
        font-size: 13px;
    }
    
    div[data-testid="metric-container"] div {
        color: #F8FAFC !important; /* Valor numérico en blanco brillante */
    }

    /* Títulos globales limpios */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Personalización avanzada de los Tabs (Pestañas) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0F172A;
        padding: 10px 10px 0 10px;
        border-radius: 12px 12px 0 0;
        border-bottom: 1px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        color: #94A3B8; /* Color de pestaña inactiva */
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
        font-size: 14px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    
    /* Pestaña Activa con acento naranja */
    .stTabs [aria-selected="true"] {
        color: #F97316 !important; /* Letra naranja */
        border-bottom: 3px solid #F97316 !important; /* Subrayado naranja */
        background-color: #1E293B; /* Fondo ligeramente más claro */
    }

    /* Estilización del Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Textos del sidebar unificados */
    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)



# ==========================================
# 2. CARGA DE DATOS (MOCKUP / REAL)
# ==========================================
import streamlit as st
import pandas as pd

@st.cache_data  # 🚀 Esto guarda el CSV en memoria para que los filtros vuelen
def load_data():
    # 1. Cargamos el CSV usando pathlib para que sea independiente del sistema
    data_path = Path(__file__).resolve().parent.parent.parent / "outputs_modelo" / "fact_churn_predictions.csv"
    df = pd.read_csv(data_path, sep=None, engine='python')
    
    # 2. Limpiamos los nombres de las columnas por si tienen espacios en blanco invisibles al inicio o al final
    df.columns = df.columns.str.strip()
    
    return df

    #     n = 2000
        
    #     # Generar IDs
    #     merchant_ids = [f"DEU-{str(i).zfill(8)}" for i in range(1, n + 1)]
        
    #     # Variables estáticas
    #     regiones = np.random.choice(["Pichincha", "Guayas", "Azuay", "Manabí", "Resto"], n, p=[0.30, 0.25, 0.08, 0.07, 0.30])
    #     segmentos = np.random.choice(["Microempresa", "Pequeña", "Mediana", "Grande"], n, p=[0.90, 0.07, 0.025, 0.005])
    #     sectores = np.random.choice(["Comercio al por menor", "Restaurantes", "Servicios personales", "Alimentos/bebidas", "Transporte"], n)
        
    #     # Coordenadas base (Quito y Guayaquil aproximadas) con ruido
    #     lats = np.where(regiones == "Pichincha", -0.22 + np.random.normal(0, 0.05, n), -2.18 + np.random.normal(0, 0.05, n))
    #     lons = np.where(regiones == "Pichincha", -78.52 + np.random.normal(0, 0.05, n), -79.88 + np.random.normal(0, 0.05, n))
        
    #     # Fechas de onboarding (cohorte)
    #     today = datetime(2026, 3, 1)
    #     fechas_onboarding = [today - timedelta(days=np.random.randint(30, 1000)) for _ in range(n)]
    #     meses_onboarding = [f"{d.year}-{str(d.month).zfill(2)}" for d in fechas_onboarding]

    #     # Features Temporales
    #     tpv = np.where(segmentos=="Microempresa", np.random.uniform(100, 2000, n), np.random.uniform(2000, 50000, n))
        
    #     # Salida del modelo (Predicciones)
    #     prob_churn = np.random.beta(0.5, 3.5, n) # Mayoría sana, cola larga de riesgo
        
    #     niveles = []
    #     for p in prob_churn:
    #         if p > 0.75: niveles.append("Crítico")
    #         elif p > 0.50: niveles.append("Alto")
    #         elif p > 0.25: niveles.append("Medio")
    #         else: niveles.append("Bajo")
            
    #     clv = tpv * np.random.uniform(6, 24, n) # LTV proyectado simple
        
    #     drivers_opciones = ["ratio_tpv_3m_vs_12m", "tasa_rechazo_max_90d", "tiempo_resolucion_prom_90d", "dias_desde_ultima_trx", "tickets_no_resueltos"]
    #     nba_opciones = ["Llamada KAM Inmediata", "Descuento en comisión 30d", "Revisión Técnica QR", "Campaña SMS Reactivación", "Visita de campo"]

    #     df = pd.DataFrame({
    #         "merchant_id": merchant_ids,
    #         "nombre_comercio": [f"Comercio {i}" for i in range(1, n+1)],
    #         "region": regiones,
    #         "segmento_comercial": segmentos,
    #         "tipo_negocio_desc": sectores,
    #         "latitud": lats,
    #         "longitud": lons,
    #         "mes_onboarding": meses_onboarding,
    #         "tpv_mensual_promedio": tpv,
    #         "probabilidad_churn": prob_churn,
    #         "nivel_riesgo": niveles,
    #         "clv_proyectado": clv,
    #         "score_salud": (1 - prob_churn) * 100,
    #         "driver_1_nombre": np.random.choice(drivers_opciones, n),
    #         "driver_1_shap": np.random.uniform(0.1, 0.5, n),
    #         "driver_2_nombre": np.random.choice(drivers_opciones, n),
    #         "driver_2_shap": np.random.uniform(0.05, 0.2, n),
    #         "nba_sugerida": np.random.choice(nba_opciones, n)
    #     })
    return df

df = load_data()

# ==========================================
# 3. BARRA LATERAL (SIDEBAR) & FILTROS
# ==========================================
with st.sidebar:
    # Contenedor HTML para centrar el logo visualmente
    st.markdown("<div style='text-align: center; padding: 10px 0 20px 0;'>", unsafe_allow_html=True)
    st.image("https://res.cloudinary.com/doy9vd3pj/image/upload/q_auto/f_auto/v1776540402/unnamed_ooahqu.png", width=140)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # st.divider es mucho más limpio que st.markdown("---")
    st.divider()
    
    st.markdown("### 🎛️ Filtros Globales")
    st.markdown("<p style='color: #94A3B8; font-size: 13px; margin-bottom: 20px;'>Ajusta los parámetros del portafolio</p>", unsafe_allow_html=True)

    # Filtros nativos (Streamlit maneja muy bien el modo oscuro aquí por defecto)
    selected_region = st.multiselect(
        "📍 Región", 
        options=df['region'].unique(), 
        default=df['region'].unique(),
        help="Segmenta los comercios por ubicación geográfica"
    )
    
    selected_segmento = st.multiselect(
        "🏢 Segmento Comercial", 
        options=df['segmento_comercial'].unique(), 
        default=df['segmento_comercial'].unique()
    )
    
    selected_risk = st.multiselect(
        "🚨 Nivel de Riesgo", 
        options=list(RISK_COLORS.keys()), 
        default=list(RISK_COLORS.keys())
    )

    st.divider()
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 12px;'>D'Una Churn Intelligence v1.0</p>", unsafe_allow_html=True)

# Lógica de filtrado
df_filtered = df[
    (df['region'].isin(selected_region)) & 
    (df['segmento_comercial'].isin(selected_segmento)) &
    (df['nivel_riesgo'].isin(selected_risk))
]

# ==========================================
# 4. TABS (VISTAS PRINCIPALES)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen", 
    "📅 Cohortes", 
    "💼 Inteligencia", 
    "🗺️ Geoespacial", 
    "🔍 Perfil Profundo"
])


# ------------------------------------------
# TAB 1: RESUMEN EJECUTIVO (MODO OSCURO / FINTECH STYLE)
# ------------------------------------------
with tab1:
    st.markdown("### 📊 Panel Ejecutivo de Riesgo de Abandono (Próximos 30 días)")
    st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Visión general de la salud del portafolio y exposición al riesgo.</p>", unsafe_allow_html=True)
    
    # KPIs Top
    col1, col2, col3, col4 = st.columns(4)
    total_merchants = len(df_filtered)
    merchants_at_risk = len(df_filtered[df_filtered['nivel_riesgo'].isin(['Alto', 'Crítico'])])
    churn_rate = (merchants_at_risk / total_merchants) * 100 if total_merchants > 0 else 0
    tpv_at_risk = df_filtered[df_filtered['nivel_riesgo'].isin(['Alto', 'Crítico'])]['clv_proyectado'].sum()
    score_promedio = df_filtered['score_salud'].mean() if total_merchants > 0 else 0
    
    # El CSS del paso anterior estilizará automáticamente estos st.metric
    col1.metric("Total Comercios", f"{total_merchants:,}")
    col2.metric("Comercios en Riesgo", f"{merchants_at_risk:,}", delta=f"{churn_rate:.1f}% Tasa Riesgo", delta_color="inverse")
    col3.metric("CLV en Riesgo (USD)", f"${tpv_at_risk:,.0f}", delta="Proyectado a 30 días", delta_color="off")
    col4.metric("Salud del Portafolio", f"{score_promedio:.1f} / 100", delta="Score Promedio", delta_color="normal")

    st.divider() # Usamos st.divider() que es más moderno que st.markdown("---")

    # Gráficos
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        risk_counts = df_filtered['nivel_riesgo'].value_counts().reset_index()
        risk_counts.columns = ['Nivel de Riesgo', 'Cantidad']
        
        fig_pie = px.pie(
            risk_counts, 
            values='Cantidad', 
            names='Nivel de Riesgo', 
            color='Nivel de Riesgo', 
            color_discrete_map=RISK_COLORS, 
            hole=0.7 # Anillo más delgado y elegante
        )
        
        # Ajustes estéticos del Donut Chart
        fig_pie.update_traces(
            textposition='outside', 
            textinfo='percent+label',
            textfont=dict(color='#E2E8F0', size=14),
            marker=dict(line=dict(color='#0B0F19', width=3)), # Borde del color del fondo principal para separar tajadas
            hoverinfo="label+percent+value"
        )
        
        fig_pie.update_layout(
            title=dict(text="Distribución por Nivel de Riesgo", font=dict(size=18, color='#F8FAFC')),
            showlegend=False, # Ocultamos la leyenda porque ya está en los labels
            paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente
            plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            margin=dict(t=50, b=20, l=20, r=20),
            # Texto central del donut
            annotations=[dict(text=f'{total_merchants:,}<br>Comercios', x=0.5, y=0.5, font_size=16, font_color='#94A3B8', showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        ciiu_risk = df_filtered.groupby('tipo_negocio_desc')['probabilidad_churn'].mean().reset_index()
        ciiu_risk = ciiu_risk.sort_values('probabilidad_churn', ascending=True)
        
        fig_bar = px.bar(
            ciiu_risk, 
            x='probabilidad_churn', 
            y='tipo_negocio_desc', 
            orientation='h',
            text='probabilidad_churn' # Mostrar el valor en la barra
        )
        
        # Ajustes estéticos del Bar Chart
        fig_bar.update_traces(
            marker_color='#F97316', # Usamos el color de acento naranja principal
            marker_line_width=0, # Sin bordes en las barras
            texttemplate='%{text:.1%}', # Formato de porcentaje con 1 decimal
            textposition='outside', # Texto fuera de la barra para legibilidad
            textfont=dict(color='#E2E8F0', size=13)
        )
        
        fig_bar.update_layout(
            title=dict(text="Riesgo Promedio por Segmento", font=dict(size=18, color='#F8FAFC')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            xaxis=dict(showgrid=False, visible=False), # Ocultamos completamente el eje X (ya tenemos los labels)
            yaxis=dict(showgrid=False, title="", tickfont=dict(size=13)),
            margin=dict(t=50, b=20, l=10, r=40) # Margen derecho un poco más amplio para que quepan los porcentajes
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# TAB 2: MATRIZ DE COHORTES (MODO OSCURO / FINTECH STYLE)
# ------------------------------------------
with tab2:
    st.markdown("### 📅 Análisis de Cohortes")
    st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Evolución de la probabilidad de abandono según el mes de onboarding y segmento comercial.</p>", unsafe_allow_html=True)
    
    # Preparación de datos
    cohort_data = df_filtered.groupby(['mes_onboarding', 'segmento_comercial'])['probabilidad_churn'].mean().reset_index()
    cohort_pivot = cohort_data.pivot(index='mes_onboarding', columns='segmento_comercial', values='probabilidad_churn').fillna(0)
    
    # Definir una escala de colores personalizada para Dark Mode (Azul oscuro -> Naranja -> Rojo)
    custom_color_scale = [
        [0.0, '#1E293B'],   # Riesgo bajo (se funde casi con el fondo)
        [0.5, '#F97316'],   # Riesgo medio-alto (Acento naranja)
        [1.0, '#EF4444']    # Riesgo crítico (Rojo vibrante)
    ]

    # Crear el heatmap con texto automático y la nueva escala
    fig_heatmap = px.imshow(
        cohort_pivot, 
        labels=dict(x="", y="", color="Probabilidad Churn"),
        x=cohort_pivot.columns, 
        y=cohort_pivot.index,
        color_continuous_scale=custom_color_scale, 
        aspect="auto",
        text_auto=".1%"  # Muestra el porcentaje directamente en la celda
    )
    
    # Ajustes estéticos de alto nivel
    fig_heatmap.update_traces(
        xgap=4, # Espacio entre columnas (crea el efecto de "tarjetas" individuales)
        ygap=4, # Espacio entre filas
        hovertemplate="<b>Mes:</b> %{y}<br><b>Segmento:</b> %{x}<br><b>Riesgo:</b> %{z:.1%}<extra></extra>",
        textfont=dict(family='Inter', color='#F8FAFC', size=12) # Letra blanca y limpia
    )
    
    fig_heatmap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter'),
        margin=dict(t=10, b=30, l=10, r=10),
        xaxis=dict(
            side="bottom", 
            tickfont=dict(size=13, color="#94A3B8"),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=13, color="#94A3B8"),
            showgrid=False,
            autorange="reversed" # Para que el mes más antiguo o reciente empiece arriba lógicamente
        ),
        coloraxis_showscale=False # Ocultamos la barra lateral de colores para un diseño más limpio
    )
    
    # Contenedor para darle un margen y fondo sutil a la gráfica entera
    st.markdown("<div style='background-color: #0F172A; padding: 20px; border-radius: 12px; border: 1px solid #1E293B;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: INTELIGENCIA DE CUENTAS (MODO OSCURO / FINTECH)
# ------------------------------------------
with tab3:
    # Layout del encabezado con botón de exportación alineado a la derecha
    head_col1, head_col2 = st.columns([3, 1])
    
    with head_col1:
        st.markdown("### 💼 Workspace de Retención")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Lista priorizada de comercios con acciones de retención sugeridas (Next Best Action).</p>", unsafe_allow_html=True)
    
    # Preparación de datos y asignación rápida de ejecutivos
    if 'ejecutivo_cuenta' not in df_filtered.columns:
        df_filtered['ejecutivo_cuenta'] = np.where(df_filtered['segmento_comercial'] == "Microempresa", "Auto-gestionado", "KAM Asignado")
         
    display_cols = ['merchant_id', 'nombre_comercio', 'nivel_riesgo', 'probabilidad_churn', 'clv_proyectado', 'nba_sugerida', 'ejecutivo_cuenta']
    table_df = df_filtered[display_cols].sort_values(by='probabilidad_churn', ascending=False).copy()
    
    # Truco visual: Reemplazar el nivel de riesgo por texto enriquecido con emojis para no pelear con el CSS oscuro
    def format_risk_icon(val):
        icons = {"Crítico": "🔴 Crítico", "Alto": "🟠 Alto", "Medio": "🟡 Medio", "Bajo": "🟢 Bajo"}
        return icons.get(val, val)
        
    table_df['nivel_riesgo'] = table_df['nivel_riesgo'].apply(format_risk_icon)
    
    with head_col2:
        st.markdown("<br>", unsafe_allow_html=True) # Espaciado para alinear con el título
        csv = table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Exportar Plan (CSV)",
            data=csv,
            file_name="plan_retencion_deuna.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Renderizado Moderno de Tabla Interactiva
    st.dataframe(
        table_df,
        column_config={
            "merchant_id": st.column_config.TextColumn(
                "ID Comercio", 
                width="small"
            ),
            "nombre_comercio": st.column_config.TextColumn(
                "Nombre", 
                width="medium"
            ),
            "nivel_riesgo": st.column_config.TextColumn(
                "Alerta", 
                width="small"
            ),
            "probabilidad_churn": st.column_config.ProgressColumn(
                "Riesgo de Fuga",
                help="Probabilidad calculada por el modelo predictivo (0 a 1)",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "clv_proyectado": st.column_config.NumberColumn(
                "CLV en Riesgo",
                help="Valor proyectado a 30 días",
                format="$ %.0f",
            ),
            "nba_sugerida": st.column_config.TextColumn(
                "Next Best Action (Sugerencia)",
                width="large"
            ),
            "ejecutivo_cuenta": st.column_config.TextColumn(
                "Ejecutivo"
            )
        },
        hide_index=True, # Ocultamos el índice numérico que no aporta valor
        use_container_width=True,
        height=450
    )

# ------------------------------------------
# TAB 4: MAPA GEOESPACIAL (CORREGIDO)
# ------------------------------------------
with tab4:
    st.markdown("### 🗺️ Focos Geográficos de Riesgo")
    st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Identifica zonas con alta concentración de riesgo para enfocar visitas de campo y campañas locales.</p>", unsafe_allow_html=True)
    
    # Dividir en Mapa y Panel de Insights
    col_map, col_geo_kpi = st.columns([3, 1])
    
    with col_map:
        fig_map = px.scatter_mapbox(
            df_filtered, 
            lat="latitud", 
            lon="longitud", 
            color="nivel_riesgo",
            size="clv_proyectado", 
            color_discrete_map=RISK_COLORS,
            hover_name="nombre_comercio", 
            hover_data={"nivel_riesgo": True, "probabilidad_churn": ":.1%", "clv_proyectado": ":$,.0f", "latitud": False, "longitud": False},
            zoom=5.5, 
            center={"lat": -1.8312, "lon": -78.1834}, 
            mapbox_style="carto-darkmatter"
        )
        
        # EL FIX: Solo opacity y sizemin, sin la propiedad 'line'
        fig_map.update_traces(
            marker=dict(opacity=0.85, sizemin=5) 
        )
        
        # Diseño limpio sin márgenes y leyenda flotante sobre el mapa
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                title=dict(text=""),
                yanchor="top", y=0.98, 
                xanchor="left", x=0.02,
                bgcolor="rgba(15, 23, 42, 0.85)", 
                font=dict(color="#E2E8F0", size=12),
                bordercolor="#334155",
                borderwidth=1,
                itemsizing="constant"
            )
        )
        
        # Envolvemos el mapa en un contenedor oscuro
        st.markdown("<div style='border: 1px solid #1E293B; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_geo_kpi:
        st.markdown("<h4 style='font-size: 16px; color: #F8FAFC; margin-top: 5px; margin-bottom: 15px;'>📍 Top Zonas en Alerta</h4>", unsafe_allow_html=True)
        
        # Filtrar solo comercios en riesgo (Alto o Crítico)
        df_risk_geo = df_filtered[df_filtered['nivel_riesgo'].isin(['Alto', 'Crítico'])]
        
        if not df_risk_geo.empty:
            # Agrupar por región
            risk_by_region = df_risk_geo.groupby('region').agg(
                comercios=('merchant_id', 'count'),
                clv_peligro=('clv_proyectado', 'sum')
            ).reset_index().sort_values('comercios', ascending=False).head(4)
            
            # Generar tarjetas de KPI geográfico
            for _, row in risk_by_region.iterrows():
                st.markdown(f"""
                <div style='background-color: #1E293B; padding: 16px; border-radius: 12px; margin-bottom: 12px; border-left: 4px solid #EF4444; border-right: 1px solid #334155; border-top: 1px solid #334155; border-bottom: 1px solid #334155;'>
                    <p style='margin: 0; font-size: 13px; color: #94A3B8; font-weight: 600; text-transform: uppercase;'>{row['region']}</p>
                    <p style='margin: 4px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 800;'>{row['comercios']} <span style='font-size: 13px; color: #EF4444; font-weight:600;'>locales</span></p>
                    <p style='margin: 2px 0 0 0; font-size: 12px; color: #64748B;'>Riesgo: ${row['clv_peligro']:,.0f} USD</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No hay comercios en riesgo Alto/Crítico en la selección actual.")
            
        st.markdown("<br><p style='font-size: 12px; color: #64748B; line-height: 1.4;'>*El panel muestra las regiones con mayor volumen de locales categorizados en riesgo <b>Alto</b> o <b>Crítico</b>.</p>", unsafe_allow_html=True)

# ------------------------------------------
# TAB 5: PERFIL PROFUNDO (CORREGIDO)
# ------------------------------------------
with tab5:
    st.markdown("### 🔍 Análisis a Nivel de Cuenta (Explainable AI)")
    st.markdown("<p style='color: #94A3B8; margin-bottom: 20px;'>Radiografía individual para entender los conductores de fuga.</p>", unsafe_allow_html=True)
    
    formated_merchants = df_filtered['merchant_id'] + " - " + df_filtered['nombre_comercio']
    selected_option = st.selectbox("Busca o Selecciona un Comercio:", formated_merchants.tolist())
    
    if selected_option:
        selected_merchant_id = selected_option.split(" - ")[0]
        merchant_data = df_filtered[df_filtered['merchant_id'] == selected_merchant_id].iloc[0]
        
        col_prof1, col_prof2 = st.columns([1, 1.5])
        
        with col_prof1:
            risk_color = RISK_COLORS.get(merchant_data['nivel_riesgo'], '#F8FAFC')
            
            # Usamos textwrap.dedent para asegurar que no haya espacios al inicio que activen el modo código
            profile_html = textwrap.dedent(f"""
                <div style='background-color: #1E293B; padding: 24px; border-radius: 12px; border: 1px solid #334155;'>
                    <h3 style='margin: 0; color: #F8FAFC; font-size: 22px;'>{merchant_data['nombre_comercio']}</h3>
                    <p style='color: #94A3B8; font-size: 13px; margin-bottom: 20px; font-family: monospace;'>ID: {merchant_data['merchant_id']}</p>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 10px;'>
                        <span style='color: #94A3B8; font-size: 14px;'>Segmento</span>
                        <span style='color: #F8FAFC; font-weight: 600;'>{merchant_data['segmento_comercial']}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 10px;'>
                        <span style='color: #94A3B8; font-size: 14px;'>Región</span>
                        <span style='color: #F8FAFC; font-weight: 600;'>{merchant_data['region']}</span>
                    </div>
                    <div style='background-color: #0F172A; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid {risk_color}; margin: 20px 0;'>
                        <p style='margin: 0; color: #94A3B8; font-size: 12px;'>PROBABILIDAD DE FUGA</p>
                        <h2 style='margin: 5px 0; color: {risk_color}; font-size: 32px;'>{merchant_data['probabilidad_churn']*100:.1f}%</h2>
                    </div>
                    <div style='background-color: rgba(249, 115, 22, 0.1); border: 1px dashed #F97316; padding: 16px; border-radius: 8px;'>
                        <p style='margin: 0; color: #F97316; font-size: 12px; font-weight: 800;'>💡 NEXT BEST ACTION</p>
                        <p style='margin: 6px 0 0 0; color: #F8FAFC; font-size: 14px;'>{merchant_data['nba_sugerida']}</p>
                    </div>
                </div>
            """)
            st.markdown(profile_html, unsafe_allow_html=True)

        with col_prof2:
            # Agrupamos todo en un contenedor para evitar el "espacio vacío" superior
            with st.container():
                st.markdown(textwrap.dedent(f"""
                    <div style='background-color: #0F172A; padding: 20px; border-radius: 12px; border: 1px solid #1E293B;'>
                        <h4 style='color: #F8FAFC; margin: 0;'>Principales Drivers de Abandono</h4>
                        <p style='color: #64748B; font-size: 13px;'>Impacto de variables en el modelo (SHAP).</p>
                    </div>
                """), unsafe_allow_html=True)
                
                driver_1 = str(merchant_data['driver_1_nombre']).replace('_', ' ').title()
                driver_2 = str(merchant_data['driver_2_nombre']).replace('_', ' ').title()
                
                shap_df = pd.DataFrame({
                    "Driver": [driver_1, driver_2],
                    "Impacto": [merchant_data['driver_1_shap'], merchant_data['driver_2_shap']]
                }).sort_values(by="Impacto")
                
                fig_shap = px.bar(shap_df, x="Impacto", y="Driver", orientation='h', text_auto='.3f')
                fig_shap.update_traces(marker_color='#F97316', width=0.4, textposition='outside', cliponaxis=False)

                fig_shap.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"),
                    xaxis=dict(showgrid=True, gridcolor="#1E293B"),
                    yaxis=dict(showgrid=False, title=""),
                    # 2. Aumentamos la 'r' (right) de 50 a 100
                    margin=dict(t=20, b=20, l=10, r=100), 
                    height=300
                )
                st.plotly_chart(fig_shap, use_container_width=True)