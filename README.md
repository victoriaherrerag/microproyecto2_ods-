# 🇺🇳 Clasificador ODS — Microproyecto 2🇺🇳

**Clasificación automática de textos según los 17 Objetivos de Desarrollo Sostenible**  
Victoria Herrera · Andrés García · Maestría en IA — ML No Supervisado

---

## Estructura del proyecto

```
microproyecto2_ods/
├── streamlit_app.py          
├── requirements.txt          
├── README.md
└── resources/
    ├── batch/
    │   └── streamlit.bat    
    └── models/
        └── pipeline.joblib   
```
---

## 🌐 Despliegue en Streamlit 

1. Se sube el repositorio a GitHub (incluye `pipeline.joblib` en `resources/models/`)
2. En [share.streamlit.io](https://share.streamlit.io)
3. Se conecta tu repositorio
4. En **Main file path** se pone `streamlit_app.py`
5. Y finalmente **Deploy**

---

## Pipeline del modelo

```
Texto 
    ↓  NLP: tokenización + stopwords + stemming (español)
    ↓  TF-IDF: vectorización (max_features=5000)
    ↓  TruncatedSVD: reducción de dimensionalidad (n_components=100)
    ↓  LogisticRegression: clasificación (class_weight='balanced')
    ↓
ODS predicho (1–17)
```

---

## ✨ Funcionalidades de la app

- **Texto individual**: escribe o pega un texto y para obtener el ODS predicho con nivel de confianza y gráfico de probabilidades Top-5
- **Clasificación por lote**: procesa múltiples textos a la vez con exportación a CSV
- **Sidebar**: referencia visual de los 17 ODS con colores oficiales de la ONU
