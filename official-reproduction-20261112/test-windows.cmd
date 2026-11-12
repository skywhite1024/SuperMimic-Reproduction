@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
"C:\Users\260010076\anaconda3\envs\supermimic-official\python.exe" -m pytest upstream/supermimic/tests -q --tb=short --junitxml=official-reproduction-20261112/windows-tests.xml
