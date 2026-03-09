"""
Clasificador de Textos según los ODS
Microproyecto 2 - ML No Supervisado
Victoria Herrera - Andrés Rueda
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Descargar recursos solo si no existen
try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# Objetos usados en el pipeline
_STOP_WORDS = set(stopwords.words("spanish"))
_STEMMER = SnowballStemmer("spanish")
_TOKENIZER = RegexpTokenizer(r"\w+")


def preprocesar_serie(textos):
    """
    Preprocesamiento usado durante el entrenamiento del modelo.
    """
    resultado = []

    for texto in textos:
        tokens = _TOKENIZER.tokenize(str(texto).lower())
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
        tokens = [_STEMMER.stem(t) for t in tokens]
        resultado.append(" ".join(tokens))

    return resultado


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

# Colores oficiales ODS
COLORES_ODS = {
    1:"#E5243B",2:"#DDA63A",3:"#4C9F38",4:"#C5192D",
    5:"#FF3A21",6:"#26BDE2",7:"#FCC30B",8:"#A21942",
    9:"#FD6925",10:"#DD1367",11:"#FD9D24",12:"#BF8B2E",
    13:"#3F7E44",14:"#0A97D9",15:"#56C02B",16:"#00689D",
    17:"#19486A",
}

# Emojis
EMOJIS_ODS = {
    1:"🏚️",2:"🌾",3:"❤️",4:"📚",5:"⚧️",
    6:"💧",7:"⚡",8:"💼",9:"🏭",10:"⚖️",
    11:"🏙️",12:"♻️",13:"🌡️",14:"🐋",15:"🌿",
    16:"🕊️",17:"🤝",
}


# Cargar modelo
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_pipeline():

    ruta_base = os.path.dirname(__file__)
    ruta_modelo = os.path.join(ruta_base, "resources", "models", "pipeline.joblib")

    if not os.path.exists(ruta_modelo):
        return None, ruta_modelo

    pipeline = joblib.load(ruta_modelo)

    return pipeline, ruta_modelo


# Predicción
def predecir(texto: str, pipeline):

    ods = int(pipeline.predict([texto])[0])
    probs = pipeline.predict_proba([texto])[0]

    clases = pipeline.classes_
    confianza = float(max(probs))

    prob_dict = {int(c): float(p) for c, p in zip(clases, probs)}

    return ods, confianza, prob_dict


# Gráfico
def grafico_probabilidades(prob_dict: dict, ods_predicho: int):

    top_n = 5
    ordenado = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    ods_ids = [item[0] for item in ordenado]
    probs   = [item[1] * 100 for item in ordenado]

    labels  = [f"ODS {o}: {NOMBRES_ODS.get(o,'')}" for o in ods_ids]
    colores = [COLORES_ODS.get(o,"#888") for o in ods_ids]

    fig, ax = plt.subplots(figsize=(7,3))

    bars = ax.barh(labels[::-1], probs[::-1], color=colores[::-1], edgecolor="white")

    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{prob:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Probabilidad (%)")
    ax.set_title("Top 5 ODS más probables")
    ax.spines[["top","right","left"]].set_visible(False)

    fig.tight_layout()

    return fig


# CSS
st.markdown("""
<style>
.ods-card{
border-radius:12px;
padding:1.5rem 2rem;
color:white;
text-align:center;
}
.ods-numero{font-size:3rem;font-weight:900}
.ods-nombre{font-size:1.3rem;font-weight:600}
.metric-box{
background:#f0f2f6;
border-radius:8px;
padding:0.8rem;
text-align:center;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:

    st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/UN_emblem_blue.svg/1280px-UN_emblem_blue.svg.png",
    width=80)

    st.title("🌐 Clasificador ODS 🇺🇳")

    st.markdown(
    "**Microproyecto 2** — ML No Supervisado  \n"
    "Victoria Herrera · Andrés Rueda")

    st.divider()

    st.markdown("### ¿Cómo funciona?")

    st.markdown(
    "1. Escribe un texto\n"
    "2. Presiona Clasificar\n"
    "3. El modelo identifica el ODS")

    st.divider()

    for num,nombre in NOMBRES_ODS.items():

        color=COLORES_ODS[num]
        emoji=EMOJIS_ODS[num]

        st.markdown(
        f'<span style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:0.75rem;">ODS {num}</span> {emoji} {nombre}',
        unsafe_allow_html=True)


# Header
st.title("Clasificador Automático de Textos según los ODS 🇺🇳")

st.markdown(
"Ingresa un texto sobre problemáticas sociales, ambientales o económicas "
"y el modelo lo clasificará en uno de los **17 ODS**."
"Ejemplo de textos"
"Los niños de zonas rurales no tienen acceso a escuelas primarias, necesitamos más profesores y materiales educativos para garantizar su aprendizaje",
"Las comunidades indígenas no cuentan con agua potable ni sistemas de saneamiento básico, lo que genera enfermedades gastrointestinales en los niños",
)

pipeline, ruta_modelo = cargar_pipeline()

if pipeline is None:

    st.error(
    f"No se encontró el modelo en:\n\n{ruta_modelo}\n\n"
    "Verifica que exista en resources/models/")

    st.stop()


tab1, tab2 = st.tabs(["Texto individual","Clasificación por lote"])


# TAB 1
with tab1:

    texto_input = st.text_area(
    "Ingresa tu texto:",
    height=160)

    if st.button("Clasificar",type="primary"):

        if texto_input.strip()=="":
            st.warning("Ingresa un texto")

        else:

            ods_pred,confianza,prob_dict = predecir(texto_input,pipeline)

            color=COLORES_ODS[ods_pred]
            nombre=NOMBRES_ODS[ods_pred]
            emoji=EMOJIS_ODS[ods_pred]

            st.markdown(
            f'<div class="ods-card" style="background:{color};">'
            f'<div class="ods-numero">{emoji} ODS {ods_pred}</div>'
            f'<div class="ods-nombre">{nombre}</div>'
            f'<div>Confianza: {confianza:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True)

            fig=grafico_probabilidades(prob_dict,ods_pred)

            st.pyplot(fig)


# TAB 2
with tab2:

    textos_lote = st.text_area(
    "Textos (uno por línea):",
    height=200)

    if st.button("Clasificar lote"):

        lineas=[l.strip() for l in textos_lote.splitlines() if l.strip()]

        resultados=[]

        for texto in lineas:

            ods_p,conf,_=predecir(texto,pipeline)

            resultados.append({
            "Texto":texto,
            "ODS":ods_p,
            "Nombre":NOMBRES_ODS[ods_p],
            "Confianza":f"{conf:.1%}"
            })

        df=pd.DataFrame(resultados)

        st.dataframe(df)

        csv=df.to_csv(index=False)

        st.download_button(
        "Descargar CSV",
        data=csv,
        file_name="clasificacion_ods.csv",
        mime="text/csv")


st.divider()

st.markdown(
"<div style='text-align:center;color:#888;font-size:0.8rem;'>"
"Microproyecto 2 · Maestría IA · 2026"
"</div>",
unsafe_allow_html=True)
