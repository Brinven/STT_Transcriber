@echo off
setlocal

set "VENV_DIR=%~dp0venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

:: Check if venv exists
if exist "%PYTHON%" goto :run

echo ============================================
echo  Creating virtual environment...
echo ============================================
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create venv. Is Python installed?
    pause
    exit /b 1
)

echo ============================================
echo  Installing PyTorch with CUDA 12.8...
echo ============================================
"%PIP%" install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)

echo ============================================
echo  Installing dependencies...
echo ============================================
"%PIP%" install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo ============================================
echo  Installing transformers from source...
echo ============================================
"%PIP%" install "git+https://github.com/huggingface/transformers.git"
if errorlevel 1 (
    echo WARNING: transformers source install failed. MedASR may not work.
)

echo ============================================
echo  Patching speechbrain for torchaudio 2.9...
echo ============================================
"%PIP%" install --force-reinstall --no-deps "git+https://github.com/speechbrain/speechbrain.git@develop"
if errorlevel 1 (
    echo WARNING: speechbrain patch failed. Speaker ID may not work.
)

echo ============================================
echo  Removing torchcodec (broken on Windows)...
echo ============================================
"%PIP%" uninstall torchcodec -y >nul 2>&1

echo ============================================
echo  Setup complete!
echo ============================================

:run
echo Starting STT Transcriber...
"%PYTHON%" "%~dp0main.py"
