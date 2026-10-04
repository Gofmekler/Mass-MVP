@echo off
chcp 65001 >nul
rem Сборка MassLab.exe. Нужен Python 3.8 (последняя версия для Windows 7).
rem Получившаяся программа работает на Windows 7, 8, 10 и 11.
cd /d "%~dp0"
py -3.8 -m pip install -r requirements-win7.txt || goto :error
py -3.8 -m pytest -q || goto :error
py -3.8 -m PyInstaller --noconfirm --clean --windowed --name MassLab --exclude-module tkinter --exclude-module PySide6 main.py || goto :error
echo.
echo Готово: папка dist\MassLab — скопируйте её целиком на флешку и запускайте MassLab.exe
pause
exit /b 0
:error
echo Ошибка сборки
pause
exit /b 1
