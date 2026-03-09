@echo off
echo ============================================
echo  Clasificador ODS - Microproyecto 2
echo  Victoria Herrera - Andres Garcia
echo ============================================
echo.

:: Establecer la raiz del proyecto como PYTHONPATH
SET "PROJECT_ROOT=%~dp0..\.."
SET "PYTHONPATH=%PROJECT_ROOT%"

echo Iniciando aplicacion Streamlit...
echo Ruta del proyecto: %PROJECT_ROOT%
echo.

:: Ejecutar streamlit desde la raiz del proyecto
streamlit run "%PROJECT_ROOT%\streamlit_app.py" --server.port 8501

pause
