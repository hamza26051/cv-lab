@echo off
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python lab08.py all --seed 42 --n-train 150 --n-val 30 --n-test 30 --epochs 12
pause
