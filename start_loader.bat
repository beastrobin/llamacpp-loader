@echo off
setlocal

:: Lock cwd to the project root so `-m llamacpp_loader.main` resolves.
pushd "%~dp0"

:: Locate a Python interpreter on PATH (no hard-coded user paths - keeps the
:: script portable and avoids leaking the local username into a public repo).
:: Prefer pythonw (no console window); fall back to python (console) if missing.
:: Each candidate is verified to actually import llamacpp_loader before use.
set "PYW="
for %%P in (pythonw python) do (
    if not defined PYW (
        for /f "delims=" %%I in ('where %%P 2^>nul') do (
            if not defined PYW (
                "%%I" -c "import llamacpp_loader" >nul 2>nul
                if not errorlevel 1 set "PYW=%%I"
            )
        )
    )
)

if not defined PYW (
    if not exist "%APPDATA%\llamacpp-loader" mkdir "%APPDATA%\llamacpp-loader" >nul 2>nul
    >>"%APPDATA%\llamacpp-loader\app.log" echo %DATE% %TIME% ERROR: no Python with llamacpp_loader found on PATH
    exit /b 1
)

:: Launch hidden via PowerShell: the console window flashes and closes by
:: itself, only the GUI stays. NOTE: keep this file ASCII-only - non-ASCII
:: chars (UTF-8) get mangled by cmd's GBK parsing and break the script.
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath '%PYW%' -ArgumentList '-m','llamacpp_loader.main' -WorkingDirectory '%CD%'" >nul 2>nul
if errorlevel 1 (
    :: PowerShell blocked -> visible console fallback (window stays, GUI opens).
    "%PYW%" -u -m llamacpp_loader.main
)
exit /b
