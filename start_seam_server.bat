@echo off
REM Starts the Seam model API on this machine and exposes it through ngrok.
REM ngrok reuses the account's free dev domain, so the public URL stays the same.
REM One-time setup: ngrok config add-authtoken <your token>  (never commit the token)

cd /d "%~dp0"
start "Seam API" cmd /k python -m uvicorn serve.app:app --host 127.0.0.1 --port 7860
start "Seam tunnel" cmd /k "%USERPROFILE%\ngrok\ngrok.exe" http 7860
echo Seam is starting. The model takes about 20 seconds to load.
echo Close both windows to take the site offline.
