
import streamlit as st
import pandas as pd
import numpy as np
import umap
from scipy.stats import gaussian_kde, spearmanr
from scipy.interpolate import griddata
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y HOOK DE AUTENTICACIÓN (PREPARADO PARA SAAS)
# ==============================================================================
st.set_page_config(
    page_title="Topos Graph-3D | Revelador Topológico",
    page_icon="🗺️",
    layout="wide"
)

def verificar_autenticacion_y_suscripcion():
    """
    Middleware de control de acceso para monetización futura.
    - Modo Demo Actual: Permite acceso directo (retorna True).
    - Modo SaaS Futuro: Valida tokens JWT/Cookies de Stripe, Supabase o Auth0.
    """
    if "usuario_autenticado" not in st.session_state:
        # Cambiar a False para activar la barrera de cobro por suscripción
        st.session_state["usuario_autenticado"] = True
        st.session_state["plan_suscripcion"] = "Pro_Demo"

    if not st.session_state["usuario_autenticado"]:
        st.title("🔒 Acceso Restringido - Topos Graph-3D Pro")
        st.warning("Se requiere una suscripción activa para analizar tableros topológicos.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔑 Iniciar Sesión")
        with col2:
            st.button("💳 Suscribirme por $29/mes")
        return False
    return True

# Barrera de control
if not verificar_autenticacion_y_suscripcion():
    st.stop()

# ==============================================================================
# 2. AGENTE 4: TRADUCTOR NARRATIVO DETERMINISTA (UMBRALES ESTRICTOS DEL MANUAL)
# ==============================================================================
def generar_reporte_narrativo(vce_df, numeric_df):
    """
    Traduce métricas geométricas a lenguaje humano para TODAS las variables
    mediante mapeo directo condicional respetando estrictamente los umbrales.
    """
    corr_matrix = numeric_df.corr(method='spearman')
    reportes = []

    for _, row in vce_df.iterrows():
        var_name = row['Variable']
        topologia = row['Topologia_Asociada']
        corr_z = row['Correlacion_Densidad_Z']
        corr_grad = row['Correlacion_Gradiente']
        impacto = row['Impacto_Estructural']

        # 1. Ubicación (Umbrales 0.35 y 0.30)
        if corr_grad > 0.35:
            ubicacion = f"Ubicada sobre la ladera de alta pendiente descendente del paisaje (Gradiente: {corr_grad:.2f})."
        elif corr_grad < -0.35:
            ubicacion = f"Ubicada sobre la ladera de alta pendiente ascendente (Gradiente: {corr_grad:.2f})."
        elif corr_z >= 0.30:
            ubicacion = f"Situada en la cima del pico de mayor concentración de datos (Correlación Z: {corr_z:.2f})."
        elif corr_z <= -0.30:
            ubicacion = f"Asentada en las profundidades de un valle de baja densidad (Correlación Z: {corr_z:.2f})."
        else:
            ubicacion = "Extendida a lo largo de una meseta plana con densidad uniforme y bajo gradiente."

        # 2. Relevancia (Umbrales 0.40 y 0.25)
        if impacto >= 0.40:
            relevancia = f"CRÍTICA (Impacto Estructural: {impacto:.2f}). Pilar geométrico fundamental. Variaciones en su valor fuerzan al sistema a cruzar umbrales críticos."
        elif impacto >= 0.25:
            relevancia = f"MODERADA (Impacto Estructural: {impacto:.2f}). Contribuye significativamente a delinear las pendientes y separar agrupamientos."
        else:
            relevancia = f"BAJA (Impacto Estructural: {impacto:.2f}). Se comporta como dimensión ortogonal o ruidosa. Su variación no altera la forma del paisaje."

        # 3. Vinculación (Umbrales 0.70 y 0.35)
        other_vars = corr_matrix[var_name].drop(var_name)
        if not other_vars.empty:
            most_linked_var = other_vars.abs().idxmax()
            linked_corr_val = other_vars[most_linked_var]

            if abs(linked_corr_val) >= 0.70:
                vinculo = f"Fuerte acoplamiento geométrico con **'{most_linked_var}'** (Spearman: {linked_corr_val:.2f}). Ambas co-evolucionan sobre el mismo eje."
            elif abs(linked_corr_val) >= 0.35:
                vinculo = f"Acoplamiento moderado con **'{most_linked_var}'** (Spearman: {linked_corr_val:.2f})."
            else:
                vinculo = "Comportamiento geométrico autónomo sin acoplamientos fuertes."
        else:
            vinculo = "Variable única."

        reporte_txt = f"""
### Variable: **{var_name}**
* **1. Rol Topológico:** {topologia}
* **2. ¿Dónde está en el paisaje?:** {ubicacion}
* **3. ¿Por qué es relevante?:** {relevancia}
* **4. ¿Con qué otras variables se vincula?:** {vinculo}
---
"""
        reportes.append(reporte_txt)

    return "\n".join(reportes)

# ==============================================================================
# 3. PIPELINE MOTOR TOPOLÓGICO (AGENTES 1, 2 Y 3)
# ==============================================================================
def ejecutar_pipeline_topologico(df):
    # Agente 1: Sanitización
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all').dropna()

    if numeric_df.shape[1] < 2:
        st.error("⚠️ El dataset requiere al menos 2 columnas numéricas válidas.")
        return None, None, None

    # Varianza cero
    variances = numeric_df.var()
    non_zero_var_cols = variances[variances > 0].index
    if len(non_zero_var_cols) < 2:
        st.error("⚠️ El dataset no posee suficientes columnas numéricas con variación.")
        return None, None, None

    numeric_df = numeric_df[non_zero_var_cols]

    # Filas mínimas
    if len(numeric_df) < 10:
        st.error("⚠️ El dataset contiene muy pocas observaciones (menos de 10 filas) para UMAP.")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    feature_names = numeric_df.columns.tolist()

    # Agente 2: UMAP & KDE 3D
    n_neighbors = min(15, max(2, len(numeric_df) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(X_scaled)
    x_emb, y_emb = coords_2d[:, 0], coords_2d[:, 1]
    xy_sample = np.vstack([x_emb, y_emb])

    try:
        kde = gaussian_kde(xy_sample)
        z_emb = kde(xy_sample)
    except Exception as e:
        st.error(f"⚠️ Error al calcular la densidad del paisaje topológico: {e}")
        return None, None, None

    grid_x, grid_y = np.mgrid[x_emb.min():x_emb.max():100j, y_emb.min():y_emb.max():100j]
    grid_z = griddata((x_emb, y_emb), z_emb, (grid_x, grid_y), method='cubic', fill_value=0)

    # Agente 3: Gradientes y Curvatura
    gy, gx = np.gradient(grid_z)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    laplacian = gxx + gyy

    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    point_grad = griddata(points, gradient_mag.ravel(), (x_emb, y_emb), method='nearest')
    point_lapl = griddata(points, laplacian.ravel(), (x_emb, y_emb), method='nearest')

    vce_scores = []
    for col in feature_names:
        vals = numeric_df[col].values
        corr_z, _ = spearmanr(vals, z_emb)
        corr_grad, _ = spearmanr(vals, point_grad)
        corr_lapl, _ = spearmanr(vals, point_lapl)

        impact = np.abs(corr_z) * 0.4 + np.abs(corr_grad) * 0.4 + np.abs(corr_lapl) * 0.2

        if abs(corr_grad) > 0.35:
            topo = "Caída Abrupta / Borde Crítico"
        elif corr_z > 0.30:
            topo = "Pico Topológico (Alta Densidad)"
        elif corr_z < -0.30:
            topo = "Valle / Depresión Estructural"
        else:
            topo = "Meseta / Zona de Estancamiento"

        vce_scores.append({
            'Variable': col,
            'Topologia_Asociada': topo,
            'Impacto_Estructural': round(impact, 4),
            'Correlacion_Densidad_Z': round(corr_z, 4),
            'Correlacion_Gradiente': round(corr_grad, 4),
            'Correlacion_Curvatura': round(corr_lapl, 4)
        })

    vce_df = pd.DataFrame(vce_scores).sort_values(by='Impacto_Estructural', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=grid_x[:,0], y=grid_y[0,:], z=grid_z,
        colorscale='Viridis', opacity=0.85,
        name='Superficie Topológica',
        colorbar=dict(title='Densidad Z', len=0.6, x=0.9)
    ))
    fig.add_trace(go.Scatter3d(
        x=x_emb, y=y_emb, z=z_emb,
        mode='markers',
        marker=dict(
            size=3,
            color=point_grad,
            colorscale='Magma',
            colorbar=dict(title='||∇Z|| (Pendiente)', len=0.6, x=1.1),
            opacity=0.9
        ),
        text=[f"Obs #{i}<br>Pendiente ||∇Z||: {g:.3f}" for i, g in enumerate(point_grad)],
        hoverinfo='text',
        name='Observaciones'
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Manifold Dim 1'),
            yaxis=dict(title='Manifold Dim 2'),
            zaxis=dict(title='Densidad Z')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=650
    )

    return fig, vce_df, numeric_df

# ==============================================================================
# 4. INTERFAZ DE USUARIO COMPLETA (STREAMLIT)
# ==============================================================================
st.title("🗺️ Topos Graph-3D: Revelador Geométrico de Datos")
st.markdown("Cargue un archivo **.csv** o **.xlsx** para proyectar el espacio latente 3D, detectar Variables Clave Emergentes (VCE) y generar el reporte narrativo dinámico.")

st.info("ℹ️ **Nota de rendimiento:** La primera vez que proceses un dataset, la inicialización del motor UMAP puede demorar unos segundos mientras compila los componentes en segundo plano.")

st.sidebar.header("📂 Ingesta de Datos")
uploaded_file = st.sidebar.file_uploader("Subir dataset (.csv o .xlsx)", type=["csv", "xlsx"])
use_demo = st.sidebar.checkbox("Usar Dataset Demo Sintético")

df_input = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
        st.sidebar.success(f"📁 Archivo cargado: '{uploaded_file.name}' ({df_input.shape[0]} filas, {df_input.shape[1]} columnas)")
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo: {e}")

elif use_demo:
    np.random.seed(42)
    n = 1000
    t = np.random.uniform(0, 4*np.pi, n)
    df_input = pd.DataFrame({
        'Ingreso_Mensual': t * np.cos(t) * 1000 + 5000 + np.random.normal(0, 200, n),
        'Frecuencia_Compra': t * np.sin(t) * 10 + 25 + np.random.normal(0, 2, n),
        'Score_Fidelidad': np.sin(2*t) * 50 + 50 + np.random.normal(0, 5, n),
        'Dias_Inactivo': np.random.exponential(scale=15.0, size=n),
        'Gasto_Acumulado': (t * np.cos(t) * 800) + (t * np.sin(t) * 1200) + np.random.normal(0, 100, n)
    })
    st.sidebar.info("🧪 Ejecutando con Dataset Demo comercial.")

if df_input is not None:
    with st.spinner("🔄 Procesando geometría no lineal y proyectando manifold 3D..."):
        fig, vce_df, numeric_df = ejecutar_pipeline_topologico(df_input)

        # CHEQUEO DE SEGURIDAD: evitar AttributeError si ocurre un error en el pipeline
        if fig is not None:
            st.subheader("1. Paisaje Topológico 3D Explorable")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("2. Matriz de Variables Clave Emergentes (VCE)")
                st.dataframe(vce_df, use_container_width=True)
                csv_data = vce_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Tabla VCE (CSV)", csv_data, "vce_ranking.csv", "text/csv")

            with col2:
                st.subheader("3. Reporte Narrativo Traducido a Lenguaje Humano")
                narrativa_md = generar_reporte_narrativo(vce_df, numeric_df)
                with st.container(height=400):
                    st.markdown(narrativa_md)
                st.download_button("📥 Descargar Reporte (.md)", narrativa_md.encode('utf-8'), "reporte_topologico.md", "text/markdown")
else:
    st.info("👈 Cargue un archivo CSV/XLSX en el panel izquierdo o active 'Usar Dataset Demo Sintético' para comenzar.")
