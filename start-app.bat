@echo off

echo Activating virtual environment...
call venv\Scripts\activate
echo Done!

echo.

echo Launching app:
call python app.py