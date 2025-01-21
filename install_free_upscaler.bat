@echo off
title [Hallett-AI Free and Pro Upscaler Installer]

echo.
echo ############################################################
echo  Regional Prompt Upscaler - Universal Installer
echo  for Automatic1111 / Forge by hallett-ai.com
echo ############################################################
echo.

:: 1) Detect default paths for Automatic1111/Forge installations
set "DEFAULT_PATH="
if exist "C:\Stable Diffusion WebUI Forge\webui-user.bat" (
    set "DEFAULT_PATH=C:\Stable Diffusion WebUI Forge"
) else (
    if exist "C:\stable-diffusion-webui\webui-user.bat" (
        set "DEFAULT_PATH=C:\stable-diffusion-webui"
    )
)

:: 2) Ask the user to confirm or enter the path
echo If you already know your WebUI or Forge root folder, enter it now.
if defined DEFAULT_PATH (
    echo Detected likely path: %DEFAULT_PATH%
) else (
    echo No default path detected; please type your path.
)
echo (Example:  C:\Stable Diffusion WebUI Forge   or   C:\Stable Diffusion WebUI)
echo.

:askPath
set /p "USER_PATH=Enter path (or leave blank to use detected path above): "

if "%USER_PATH%"=="" (
    if not defined DEFAULT_PATH (
        echo No default path found -- you must type a path.
        goto askPath
    )
    set "TARGET_PATH=%DEFAULT_PATH%"
) else (
    set "TARGET_PATH=%USER_PATH%"
)

if not exist "%TARGET_PATH%\webui-user.bat" (
    echo.
    echo *** Could not find webui-user.bat in "%TARGET_PATH%". ***
    echo This does not appear to be a valid Automatic1111/Forge folder.
    echo Please try again or press Ctrl+C to cancel.
    echo.
    goto askPath
)

echo.
echo Installing to: %TARGET_PATH%
echo.

:: 3) Create required extension and scripts directories if they don't exist
echo Creating required extension folders...
mkdir "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett" >nul 2>&1
mkdir "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\scripts" >nul 2>&1

:: 4) Clear the existing contents of the scripts directory
echo Clearing existing files in the scripts directory...
if exist "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\scripts\" (
    del /Q "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\scripts\*" >nul 2>&1
) else (
    echo Scripts directory not found. Creating it now...
    mkdir "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\scripts"
)

:: 5) Copy all local files from the "scripts" folder to the target directory
echo Copying new script files...
if exist "scripts\" (
    for %%F in (scripts\*) do (
        copy /Y "%%F" "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\scripts\" >nul
    )
) else (
    echo *** Local "scripts" folder not found. Ensure this installer is in the correct directory. ***
    pause
    exit /b 1
)

:: 6) Copy required files for the Free extension
echo Copying required files for the Free extension...
copy /Y "__init__.py" "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\__init__.py" >nul
copy /Y "requirements.txt" "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt" >nul

:: Verify all critical files were copied successfully
if not exist "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\__init__.py" (
    echo.
    echo *** Failed to copy __init__.py. ***
    echo Please check the installer files and try again.
    pause
    exit /b 1
)
if not exist "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt" (
    echo.
    echo *** Failed to copy requirements.txt. ***
    echo Please check the installer files and try again.
    pause
    exit /b 1
)

:: 7) Install dependencies
echo.
echo Installing Python dependencies from requirements.txt...
if exist "%TARGET_PATH%\venv\Scripts\python.exe" (
    "%TARGET_PATH%\venv\Scripts\python.exe" -m pip install --upgrade pip
    "%TARGET_PATH%\venv\Scripts\python.exe" -m pip install -r "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt"
) else (
    echo Could not find python in venv. Attempting system-wide python...
    python -m pip install --upgrade pip
    python -m pip install -r "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt"
)

:: 8) Completion message
echo.
echo ############################################################
echo Installation complete!
echo 1) Restart your Automatic1111/Forge WebUI.
echo 2) The free version is now installed:
echo    - "regional-prompt-upscaler-hallett" (core extension)
echo    containing the latest scripts.
echo.
echo If you receive the Pro version:
echo   Place the PRO .py file into:
echo   %TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\scripts\
echo   Then restart the WebUI.
echo ############################################################
echo.
pause
exit /b 0
