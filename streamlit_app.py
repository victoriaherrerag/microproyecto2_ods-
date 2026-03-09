"""
Clasificador de Textos según los ODS
Microproyecto 2 - ML No Supervisado
Victoria Herrera - Andrés García
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Clasificador ODS",
    page_icon="🇺🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constantes
NOMBRES_ODS = {
    1:  "Fin de la pobreza",
    2:  "Hambre cero",
    3:  "Salud y bienestar",
    4:  "Educación de calidad",
    5:  "Igualdad de género",
    6:  "Agua limpia y saneamiento",
    7:  "Energía asequible y no contaminante",
    8:  "Trabajo decente y crecimiento económico",
    9:  "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

# Colores oficiales ODS de la ONU
COLORES_ODS = {
    1:  "#E5243B", 2:  "#DDA63A", 3:  "#4C9F38", 4:  "#C5192D",
    5:  "#FF3A21", 6:  "#26BDE2", 7:  "#FCC30B", 8:  "#A21942",
    9:  "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A",
}

# Emojis representativos por ODS
EMOJIS_ODS = {
    1: "🏚️", 2: "🌾", 3: "❤️", 4: "📚", 5: "⚧️",
    6: "💧", 7: "⚡", 8: "💼", 9: "🏭", 10: "⚖️",
    11: "🏙️", 12: "♻️", 13: "🌡️", 14: "🐋", 15: "🌿",
    16: "🕊️", 17: "🤝",
}

# Carga del modelo
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_pipeline():
    """Carga el pipeline entrenado."""
    ruta_base = os.path.dirname(__file__)
    ruta_modelo = os.path.join(ruta_base, "resources", "models", "pipeline.joblib")
    if not os.path.exists(ruta_modelo):
        return None, ruta_modelo
    pipeline = joblib.load(ruta_modelo)
    return pipeline, ruta_modelo


# Función de predicción
def predecir(texto: str, pipeline):
    """
    Clasifica un texto y devuelve el ODS predicho,
    la confianza y las probabilidades de todos los ODS.
    """
    ods = int(pipeline.predict([texto])[0])
    probs = pipeline.predict_proba([texto])[0]
    clases = pipeline.classes_
    confianza = float(max(probs))
    prob_dict = {int(c): float(p) for c, p in zip(clases, probs)}
    return ods, confianza, prob_dict

# Gráfico de probabilidades

def grafico_probabilidades(prob_dict: dict, ods_predicho: int):
    """Genera un gráfico de barras horizontales con las top-5 probabilidades."""
    top_n = 5
    ordenado = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ods_ids = [item[0] for item in ordenado]
    probs   = [item[1] * 100 for item in ordenado]
    labels  = [f"ODS {o}: {NOMBRES_ODS.get(o, '')}" for o in ods_ids]
    colores = [COLORES_ODS.get(o, "#888888") for o in ods_ids]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels[::-1], probs[::-1], color=colores[::-1], edgecolor="white", height=0.6)

    for bar, prob in zip(bars, probs[::-1]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%", va="center", fontsize=9, color="#333333"
        )

    ax.set_xlabel("Probabilidad (%)", fontsize=9)
    ax.set_xlim(0, max(probs) * 1.2)
    ax.set_title("Top 5 ODS más probables", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


# CSS personalizado
st.markdown("""
<style>
    .ods-card {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    .ods-numero { font-size: 3rem; font-weight: 900; line-height: 1; }
    .ods-nombre { font-size: 1.3rem; font-weight: 600; margin-top: 0.3rem; }
    .ods-confianza { font-size: 1rem; opacity: 0.9; margin-top: 0.3rem; }
    .stTextArea textarea { font-size: 1rem; }
    .metric-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/UN_emblem_blue.svg/1280px-UN_emblem_blue.svg.png",
        width=80,
    )
    st.title("🌐 Clasificador ODS🇺🇳")
    st.markdown(
        "**Microproyecto 2** — ML No Supervisado  \n"
        "Victoria Herrera · Andrés García"
    )
    st.divider()
    st.markdown("### ¿Cómo funciona?")
    st.markdown(
        "1. **Escribe** un texto en español de una problematica a de tu comunidad \n"
        "2. Presiona **Clasificar**  \n"
        "3. El modelo identifica el ODS más relacionado  \n\n"
    )
    st.divider()
    st.markdown("### 17 Objetivos de Desarrollo Sostenible")
    for num, nombre in NOMBRES_ODS.items():
        color = COLORES_ODS[num]
        emoji = EMOJIS_ODS[num]
        st.markdown(
            f'<span style="background:{color};color:white;'
            f'border-radius:4px;padding:2px 6px;font-size:0.75rem;">'
            f'ODS {num}</span> {emoji} {nombre}',
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────
# Header principal
# ──────────────────────────────────────────────────────────────
st.title("Clasificador Automático de Textos según los ODS 🇺🇳")
st.markdown(
    "Ingresa un texto relacionado con problemáticas sociales, ambientales o económicas "
    "y el modelo lo clasificará en uno de los **17 Objetivos de Desarrollo Sostenible** "
    "de la Agenda 2030 de la ONU."
)

# Cargar pipeline
pipeline, ruta_modelo = cargar_pipeline()

if pipeline is None:
    st.error(
        f"⚠️ No se encontró el modelo en:\n\n`{ruta_modelo}`\n\n"
        "Asegúrate de que el archivo `pipeline.joblib` esté en la carpeta "
        "`resources/models/` dentro del proyecto."
    )
    st.stop()

# ── Pestañas: texto individual vs. lote ──────────────────────
tab1, tab2 = st.tabs(["📝 Texto individual", "📋 Clasificación por lote"])

# ─── TAB 1: Texto individual ──────────────────────────────────
with tab1:
    # Ejemplos predefinidos
    ejemplos = {
        "— Selecciona un ejemplo —": "",
        "ODS 4 · Educación": (
            "Los niños de zonas rurales no tienen acceso a escuelas primarias. "
            "Necesitamos más profesores y materiales educativos para garantizar su aprendizaje."
        ),
        "ODS 6 · Agua": (
            "Las comunidades indígenas no cuentan con agua potable ni sistemas de "
            "saneamiento básico, lo que genera enfermedades gastrointestinales en los niños."
        ),
        "ODS 13 · Clima": (
            "El aumento de la temperatura global está derritiendo los glaciares y causando "
            "fenómenos meteorológicos extremos como huracanes más intensos."
        ),
        "ODS 5 · Género": (
            "Las mujeres en áreas rurales siguen siendo excluidas del mercado laboral formal "
            "y no tienen acceso a financiamiento para emprender sus propios negocios."
        ),
        "ODS 1 · Pobreza": (
            "Millones de personas no pueden cubrir sus necesidades básicas de alimentación, "
            "vivienda y salud debido a la falta de empleo y programas sociales efectivos."
        ),
    }

    col_sel, _ = st.columns([2, 1])
    with col_sel:
        seleccion = st.selectbox("Cargar ejemplo:", list(ejemplos.keys()))

    texto_inicial = ejemplos[seleccion]
    texto_input = st.text_area(
        "✍️ Ingresa tu texto aquí:",
        value=texto_inicial,
        height=160,
        placeholder=(
            "Escribe o pega aquí un texto en español relacionado con "
            "problemáticas sociales, económicas o ambientales…"
        ),
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        clasificar = st.button("🔍 Clasificar", type="primary", use_container_width=True)
    with col_clear:
        limpiar = st.button("🗑️ Limpiar", use_container_width=False)

    if limpiar:
        st.rerun()

    if clasificar:
        texto_limpio = texto_input.strip()
        if not texto_limpio:
            st.warning("Por favor ingresa un texto antes de clasificar.")
        elif len(texto_limpio.split()) < 3:
            st.warning("El texto es muy corto. Intenta con al menos una oración completa.")
        else:
            with st.spinner("Procesando texto…"):
                ods_pred, confianza, prob_dict = predecir(texto_limpio, pipeline)

            color = COLORES_ODS.get(ods_pred, "#444444")
            nombre = NOMBRES_ODS.get(ods_pred, "Desconocido")
            emoji = EMOJIS_ODS.get(ods_pred, "🌍")

            st.divider()
            st.subheader("📊 Resultado de la clasificación")

            col_res, col_graf = st.columns([1, 1.6])

            with col_res:
                # Tarjeta principal
                st.markdown(
                    f'<div class="ods-card" style="background:{color};">'
                    f'<div class="ods-numero">{emoji} ODS {ods_pred}</div>'
                    f'<div class="ods-nombre">{nombre}</div>'
                    f'<div class="ods-confianza">Confianza: {confianza:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # Indicador de confianza
                nivel = "Alta ✅" if confianza >= 0.6 else ("Media ⚠️" if confianza >= 0.35 else "Baja ❗")
                st.markdown(
                    f'<div class="metric-box">'
                    f'<b>Nivel de confianza:</b> {nivel}<br>'
                    f'<b>Probabilidad:</b> {confianza:.2%}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_graf:
                fig = grafico_probabilidades(prob_dict, ods_pred)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("")
            with st.expander("📄 Ver texto analizado"):
                st.write(texto_limpio)


# ─── TAB 2: Lote ─────────────────────────────────────────────
with tab2:
    st.markdown(
        "Pega varios textos (uno por línea) para clasificarlos todos a la vez."
    )

    textos_lote = st.text_area(
        "📋 Textos a clasificar (uno por línea):",
        height=200,
        placeholder=(
            "Texto 1 sobre pobreza y desigualdad…\n"
            "Texto 2 sobre cambio climático y glaciares…\n"
            "Texto 3 sobre acceso al agua potable…"
        ),
    )

    if st.button("🔍 Clasificar lote", type="primary"):
        lineas = [l.strip() for l in textos_lote.splitlines() if l.strip()]
        if not lineas:
            st.warning("Ingresa al menos un texto.")
        else:
            resultados = []
            barra = st.progress(0, text="Clasificando textos…")
            for i, texto in enumerate(lineas):
                ods_p, conf, _ = predecir(texto, pipeline)
                resultados.append({
                    "Texto": texto[:120] + ("…" if len(texto) > 120 else ""),
                    "ODS": ods_p,
                    "Nombre ODS": NOMBRES_ODS.get(ods_p, ""),
                    "Confianza": f"{conf:.1%}",
                })
                barra.progress((i + 1) / len(lineas), text=f"Procesando {i+1}/{len(lineas)}…")
            barra.empty()

            df = pd.DataFrame(resultados)
            st.success(f"✅ {len(df)} textos clasificados")
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "ODS": st.column_config.NumberColumn("ODS #", width="small"),
                    "Confianza": st.column_config.TextColumn("Confianza", width="small"),
                },
            )

            # Distribución
            st.markdown("### Distribución de ODS en el lote")
            conteo = df["ODS"].value_counts().sort_index()
            fig2, ax2 = plt.subplots(figsize=(9, 3))
            colors_bar = [COLORES_ODS.get(o, "#888") for o in conteo.index]
            ax2.bar(
                [f"ODS {o}" for o in conteo.index],
                conteo.values,
                color=colors_bar,
                edgecolor="white",
            )
            ax2.set_ylabel("Cantidad de textos")
            ax2.set_title("Distribución por ODS")
            ax2.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            # Descarga CSV
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ Descargar resultados CSV",
                data=csv,
                file_name="clasificacion_ods.csv",
                mime="text/csv",
            )

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "Microproyecto 2 · Maestría en Inteligencia Artificial · ML No Supervisado<br>"
    "Victoria Herrera · Andrés García · 2025"
    "</div>",
    unsafe_allow_html=True,
)
