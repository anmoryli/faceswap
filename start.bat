@echo off
chcp 65001 >nul
echo ========================================
echo   本地换脸工具启动脚本
echo ========================================
echo.
echo 正在启动换脸 WebUI，请稍候...
echo 启动后浏览器会自动打开 http://127.0.0.1:7860
echo 按 Ctrl+C 可以停止服务
echo.
cd /d "%~dp0"
python app.py
pause
