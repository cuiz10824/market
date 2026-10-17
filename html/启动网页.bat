@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
  py -3 "..\code\server.py"
) else (
  python "..\code\server.py"
)
if errorlevel 1 (
  echo 启动失败。请确认已安装 Python 3.10 或更新版本，且 8765 端口未被占用。
  pause
)
