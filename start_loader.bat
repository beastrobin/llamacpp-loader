@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Launch the GUI without a persistent console window.
:: pythonw.exe runs Python without allocating a console, so no black window stays open.
:: Logs are written to %APPDATA%\llamacpp-loader\app.log.
::
:: We must verify the chosen interpreter can actually import llamacpp_loader:
:: WorkBuddy's bundled pythonw sits ahead on PATH but does NOT have the package
:: installed, so trusting `where` order alone would launch it, fail with a silent
:: ModuleNotFoundError (swallowed by pythonw), and the window would never appear.

set "PYW_EXE="
set "PYW_ARGS="

:: 1) Any pythonw.exe on PATH, in PATH order — use the first that imports the package.
for /f "delims=" %%I in ('where pythonw.exe 2^>nul') do (
    if not defined PYW_EXE (
        "%%I" -c "import llamacpp_loader" >nul 2>nul
        if not errorlevel 1 (
            set "PYW_EXE=%%I"
        )
    )
)

:: 2) Fall back to the py launcher (-3 picks the latest Python 3).
if not defined PYW_EXE (
    for /f "delims=" %%I in ('where py.exe 2^>nul') do (
        if not defined PYW_EXE (
            "%%I" -3 -c "import llamacpp_loader" >nul 2>nul
            if not errorlevel 1 (
                set "PYW_EXE=%%I"
                set "PYW_ARGS=-3"
            )
        )
    )
)

if not defined PYW_EXE (
    echo ERROR: No Python interpreter with llamacpp-loader installed was found.
    echo Install it from the project root with:  pip install -e .
    echo Then re-run this script.
    pause
    exit /b 1
)

if defined PYW_ARGS (
    start "" "%PYW_EXE%" %PYW_ARGS% -m llamacpp_loader.main
) else (
    start "" "%PYW_EXE%" -m llamacpp_loader.main
)
