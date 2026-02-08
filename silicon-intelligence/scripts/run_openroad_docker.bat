@echo off
echo Starting OpenROAD Container...
echo Mounting current directory: %CD%

:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

:: Run OpenROAD
:: -it: Interactive terminal
:: -v: Mount current directory to /design inside container
:: -w: Set working directory
docker run -it --rm -v "%CD%":/design -w /design openroad/openroad

if %errorlevel% neq 0 (
    echo Error: Failed to start OpenROAD container.
    echo Try running 'docker pull openroad/openroad' first.
    pause
)
