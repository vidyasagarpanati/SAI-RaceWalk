@echo off
REM Convenience wrapper so you do not have to activate the venv.
setlocal
set REPO=%~dp0
"%REPO%.venv\Scripts\python.exe" -m racewalk %*
endlocal
