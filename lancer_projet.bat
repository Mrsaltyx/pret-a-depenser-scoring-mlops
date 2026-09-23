@echo off
chcp 65001 >nul
title Pret a Depenser - Lancement du projet
cd /d "%~dp0"

REM Python gere (contient fastapi, uvicorn, streamlit, lightgbm, mlflow...)
set "PY=C:\Users\Salty\AppData\Roaming\kimi-desktop\daimon-share\daimon\runtime\python\.venv\Scripts\python.exe"

:menu
cls
echo ============================================================
echo   PRET A DEPENSER - Modele de scoring en production
echo ============================================================
echo.
echo   [1] Lancer l'API + le dashboard (demo soutenance)
echo   [2] Lancer les 21 tests automatises (pytest)
echo   [3] Rejouer la simulation de production (2000 requetes)
echo   [4] Lancer l'analyse de data drift (Evidently)
echo   [5] Ouvrir l'interface MLflow (partie 1)
echo   [6] Tout arreter
echo   [0] Quitter
echo.
set /p choix="  Ton choix : "

if "%choix%"=="1" goto demo
if "%choix%"=="2" goto tests
if "%choix%"=="3" goto simulation
if "%choix%"=="4" goto drift
if "%choix%"=="5" goto mlflow
if "%choix%"=="6" goto stop
if "%choix%"=="0" goto fin
goto menu

:demo
echo.
echo [1/3] Demarrage de l'API FastAPI ......... http://127.0.0.1:8000
start "API-scoring" "%PY%" -m uvicorn api.main:app --port 8000
echo [2/3] Demarrage du dashboard Streamlit ... http://127.0.0.1:8501
start "Dashboard-scoring" "%PY%" -m streamlit run monitoring/dashboard.py --server.headless true
echo [3/3] Ouverture du navigateur (Swagger + dashboard) ...
timeout /t 8 >nul
start http://127.0.0.1:8000/docs
start http://127.0.0.1:8501
echo.
echo L'API et le dashboard tournent dans leurs propres fenetres.
echo Choix [6] du menu pour tout arreter proprement.
echo.
pause
goto menu

:tests
echo.
echo Execution de la suite pytest ...
"%PY%" -m pytest tests/ -v
echo.
pause
goto menu

:simulation
echo.
echo Simulation de 2000 requetes contre l'API (l'API doit tourner, choix [1]) ...
"%PY%" monitoring/simulate_traffic.py --url http://127.0.0.1:8000
echo.
pause
goto menu

:drift
echo.
echo Analyse Evidently des donnees de production ...
"%PY%" monitoring/drift_analysis.py
echo Rapport : monitoring\drift_report.html
echo.
pause
goto menu

:mlflow
echo.
echo Interface MLflow sur http://127.0.0.1:5000 ...
start "MLflow-UI" "%PY%" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
timeout /t 6 >nul
start http://127.0.0.1:5000
echo.
pause
goto menu

:stop
echo.
echo Arret de l'API, du dashboard et de MLflow ...
taskkill /FI "WINDOWTITLE eq API-scoring*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Dashboard-scoring*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLflow-UI*" /F >nul 2>&1
echo Tout est arrete.
echo.
pause
goto menu

:fin
