@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist "data\index.json" (
  echo 首次运行，先建索引...
  python -m scripts.build_index
)
echo.
echo 服务启动后，浏览器打开 http://127.0.0.1:8000
echo 关闭这个窗口即停止服务。
echo.
python server.py 8000
pause
