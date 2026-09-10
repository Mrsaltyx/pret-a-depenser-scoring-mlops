@echo off
REM Lance l'interface web MLflow du projet "Pret a depenser"
cd /d "%~dp0"
set "PYTHON=C:\Users\Salty\AppData\Roaming\kimi-desktop\daimon-share\daimon\runtime\python\.venv\Scripts\python.exe"

echo Demarrage de l'UI MLflow...
echo Ouvre ton navigateur sur : http://127.0.0.1:5000
echo (Ctrl+C dans cette fenetre pour arreter)
start "" http://127.0.0.1:5000
"%PYTHON%" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
pause
