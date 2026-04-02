@echo off
chcp 65001 >nul
echo ========================================
echo   frp 客户端启动脚本
echo   本地换脸工具公网穿透
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
set FRP_DIR=%SCRIPT_DIR%frp

REM 检查frpc.exe是否存在
if not exist "%FRP_DIR%\frpc.exe" (
    echo [错误] 未找到 frpc.exe
    echo 请先下载 frp Windows版本：
    echo https://github.com/fatedier/frp/releases/download/v0.61.1/frp_0.61.1_windows_amd64.zip
    echo 解压后将 frpc.exe 放到：%FRP_DIR%\
    echo.
    pause
    exit /b 1
)

echo [提示] 确保换脸WebUI已经启动（python app.py）
echo [提示] 启动后可通过以下地址访问：
echo.
echo   公网地址: http://175.24.205.213:7860
echo.
echo 正在连接到 175.24.205.213...
echo 按 Ctrl+C 断开连接
echo.

"%FRP_DIR%\frpc.exe" -c "%FRP_DIR%\frpc.toml"
pause
