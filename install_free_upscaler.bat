@echo off
title [Hallett-AI Free and Pro Upscaler Installer]

:: Use a pleasing color scheme: black background, green text
color 0A

echo ############################################################
echo #               Hallett-AI Upscaler Installer             #
echo #       (Regional-Prompt-Upscaler-Detailer v1.x)          #
echo ############################################################
echo.

:: 1) Detect default paths for Automatic1111/Forge
set "DEFAULT_PATH="
if exist "C:\Stable Diffusion WebUI Forge\webui-user.bat" (
    set "DEFAULT_PATH=C:\Stable Diffusion WebUI Forge"
) else if exist "C:\stable-diffusion-webui\webui-user.bat" (
    set "DEFAULT_PATH=C:\stable-diffusion-webui"
)

echo Searching for Automatic1111 or Forge installation...
if defined DEFAULT_PATH (
    echo Detected default path: %DEFAULT_PATH%
) else (
    echo No default path detected. You will be prompted to enter the folder path manually.
)
echo.

:: 2) Prompt user to confirm or enter the path
:askPath
set /p "USER_PATH=Enter your WebUI/Forge root path (or press Enter to use detected path): "

if "%USER_PATH%"=="" (
    if not defined DEFAULT_PATH (
        echo.
        echo [ERROR] No default path found -- you must type a valid path containing webui-user.bat
        echo.
        goto askPath
    )
    set "TARGET_PATH=%DEFAULT_PATH%"
) else (
    set "TARGET_PATH=%USER_PATH%"
)

if not exist "%TARGET_PATH%\webui-user.bat" (
    echo.
    echo [ERROR] Could not find webui-user.bat in:
    echo        "%TARGET_PATH%"
    echo This does not appear to be a valid Automatic1111/Forge folder.
    echo Please try again or press Ctrl+C to cancel.
    echo.
    goto askPath
)

echo.
echo Installing to: %TARGET_PATH%
echo.

:: 3) Create target extension folder if missing
echo Creating extension folder (if needed)...
mkdir "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett" >nul 2>&1

:: 4) Copy EVERYTHING from the current folder to the extension folder
::    excluding certain hidden/system folders and the installer .bat itself
echo Copying all local files (scripts, YAML, etc.) to the extension folder...
set "CURRENT_DIR=%~dp0"
pushd "%CURRENT_DIR%"
robocopy . "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett" /E ^
    /XD .git .github venv ^
    /XF install_free_upscaler.bat

if %ERRORLEVEL% LSS 8 (
    echo [OK] Files copied successfully.
) else (
    echo [WARNING] Robocopy encountered a non-trivial error. ErrorLevel=%ERRORLEVEL%.
    echo See above for details; continuing anyway...
)

popd

:: 5) Verify a few critical files exist
if not exist "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\__init__.py" (
    echo.
    echo [ERROR] __init__.py not found after copy.
    echo Make sure it exists in this folder and rerun the installer.
    pause
    exit /b 1
)
if not exist "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt" (
    echo.
    echo [ERROR] requirements.txt not found after copy.
    echo Make sure it exists in this folder and rerun the installer.
    pause
    exit /b 1
)

:: 6) Install dependencies
echo.
echo Installing Python dependencies from requirements.txt...
if exist "%TARGET_PATH%\venv\Scripts\python.exe" (
    echo Using local venv Python at: %TARGET_PATH%\venv\Scripts\python.exe
    "%TARGET_PATH%\venv\Scripts\python.exe" -m pip install --upgrade pip
    "%TARGET_PATH%\venv\Scripts\python.exe" -m pip install -r "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt"
) else (
    echo [INFO] Could not find local venv Python. Attempting system-wide Python...
    python -m pip install --upgrade pip
    python -m pip install -r "%TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\requirements.txt"
)

echo.
echo ############################################################
echo #        Installation complete!                           #
echo #   1) Restart your Automatic1111/Forge WebUI.            #
echo #   2) The upscaler is now installed at:                  #
echo #      %TARGET_PATH%\extensions\regional-prompt-upscaler-hallett\
echo #
echo # If you have the PRO version:                            #
echo #   Place the PRO script .py in the "scripts\" folder.    #
echo #   Then restart the WebUI again.                         #
echo ############################################################
echo.
pause
exit /b 0
