@echo off
chcp 65001 >nul
call "%~dp0运行Python.bat" --version
if errorlevel 1 exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0自动更新任务.ps1"
if errorlevel 1 goto failed
echo 已安装后台更新任务。保持电脑开机并登录Windows，网页关闭后仍会更新。
pause
exit /b 0
:failed
echo 任务安装失败，请查看上面的Windows错误信息。
pause
exit /b 1
