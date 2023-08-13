@echo off

echo Updating...
call git pull
call venv\Scripts\activate
call pip install -r requirements.txt --upgrade
echo Done!
pause