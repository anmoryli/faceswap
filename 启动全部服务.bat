@echo off
chcp 65001 >nul
title FaceSwap - 换脸服务

echo ========================================
echo   FaceSwap 一键启动脚本
echo ========================================
echo.

set DIR=%~dp0

REM 先杀掉旧进程
echo 清理旧进程...
taskkill /F /FI "WINDOWTITLE eq FaceSwap WebUI*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq FaceSwap frpc*" >nul 2>&1
timeout /t 2 /nobreak >nul

echo [1/2] 启动换脸 WebUI...
start "FaceSwap WebUI" cmd /k "cd /d %DIR% && python app.py"

echo 等待 WebUI 启动（约15秒）...
timeout /t 15 /nobreak >nul

echo [2/2] 启动 frp 内网穿透...
start "FaceSwap frpc" cmd /k "%DIR%frp\frpc.exe -c %DIR%frp\frpc.toml"

timeout /t 4 /nobreak >nul

echo.
echo ========================================
echo   服务已启动！
echo   外网访问: http://175.24.205.213:7860
echo   本地访问: http://127.0.0.1:7860
echo ========================================
echo.
echo 两个服务窗口会保持打开，请勿关闭。
echo 关闭服务：直接关闭 FaceSwap WebUI 和 FaceSwap frpc 窗口即可。
echo.
pause
