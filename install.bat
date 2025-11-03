@echo off
REM Windows installation script for AI SecOps

echo ======================================================================
echo AI SecOps Red Team Kit - Installation
echo ======================================================================
echo.

echo [1/4] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python 3 is required but not installed.
    echo     Please install Python 3.8 or higher.
    exit /b 1
)
python --version
echo.

echo [2/4] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo [+] Virtual environment created
) else (
    echo [+] Virtual environment already exists
)
echo.

echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo [+] Virtual environment activated
echo.

echo [4/4] Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
echo.

echo [+] Installing AI SecOps Red Team Kit...
pip install -e .

echo.
echo ======================================================================
echo Installation Complete!
echo ======================================================================
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate.bat
echo.
echo For more information, see README.md
echo ======================================================================
pause

