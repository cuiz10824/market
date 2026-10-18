@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if exist ".venv-daily\Scripts\python.exe" goto check
where py >nul 2>nul
if errorlevel 1 goto usepython
py -3 -m venv .venv-daily
if errorlevel 1 goto failed
goto check
:usepython
python -m venv .venv-daily
if errorlevel 1 goto failed
:check
".venv-daily\Scripts\python.exe" -c "import numpy, lightgbm" >nul 2>nul
if not errorlevel 1 goto run
echo 首次运行，正在安装每日预测依赖，请保持联网。建议使用 Python 3.12。
".venv-daily\Scripts\python.exe" -m pip install --disable-pip-version-check -r "html\requirements.txt"
if errorlevel 1 goto failed
:run
".venv-daily\Scripts\python.exe" %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo 运行失败。请安装 Python 3.12，检查网络和报错信息；服务端口为8765。
pause
exit /b 1
