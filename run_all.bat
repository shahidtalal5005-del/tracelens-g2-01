@echo off
REM One command for Windows: tests + benchmark + phase 1 + phase 2 + viewer
cd /d "%~dp0"
python -m pytest || echo (pytest not installed - skipping tests: pip install pytest)
python -m tracelens all --out results
echo.
echo Open results\viewer.html in your browser.
