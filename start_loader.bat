@echo off
chcp 65001 >nul

:: Launch the GUI without a persistent console window.
:: pythonw.exe runs Python without allocating a console, so no black window stays open.
:: Logs are written to %APPDATA%\llamacpp-loader\app.log.
::
:: Requires pythonw (or the `py` launcher) on PATH. The llamacpp_loader package
:: must be importable — install with `pip install -e .` from the project root.

set "PYW_EXE="
where pythonw.exe >nul 2>nul && set "PYW_EXE=pythonw.exe"
if not defined PYW_EXE (
    where py.exe >nul 2>nul && set "PYW_EXE=py -3"
)

if not defined PYW_EXE (
    echo ERROR: pythonw.exe / py launcher not found on PATH.
    echo Install Python (https://www.python.org) or add it to PATH, then re-run.
    pause
    exit /b 1
)

start "" %PYW_EXE% -m llamacpp_loader.main
