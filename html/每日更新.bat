@echo off
chcp 65001 >nul
call "%~dp0运行Python.bat" "%~dp0predict.py" --refresh
