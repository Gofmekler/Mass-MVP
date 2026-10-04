@echo off
chcp 65001 >nul
rem Сборка MassLab.exe на Windows. Нужен Python 3.10+.
cd /d "%~dp0"
python -m pip install -r requirements-dev.txt || goto :error
python -m pytest -q || goto :error
python -m PyInstaller --noconfirm --clean --windowed --name MassLab --exclude-module tkinter main.py || goto :error
echo.
echo Готово: папка dist\MassLab — скопируйте её целиком на флешку и запускайте MassLab.exe
pause
exit /b 0
:error
echo Ошибка сборки
pause
exit /b 1
