@echo off
rem Run the daily ETF forward test and append its output to paper\forward_test.log.
rem Used by the Windows scheduled task "ETF Forward Test" (weekdays after the US close).
cd /d "%~dp0.."
if not exist paper mkdir paper
echo ===== %DATE% %TIME% ===== >> paper\forward_test.log
".venv\Scripts\python.exe" scripts\forward_test.py >> paper\forward_test.log 2>&1
echo exit code %ERRORLEVEL% >> paper\forward_test.log
