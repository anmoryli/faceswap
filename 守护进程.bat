@echo off
chcp 65001 >nul
title FaceSwap 守护进程 (请勿关闭)
echo ========================================
echo   FaceSwap 守护进程已启动
echo   外网访问: http://175.24.205.213:7860
echo   本地访问: http://127.0.0.1:7860
echo   此窗口最小化即可，请勿关闭
echo ========================================
D:\miniconda3\python.exe d:\Downloads\faceswap\daemon.py
pause
