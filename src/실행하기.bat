@echo off
chcp 65001 > nul
cls
color 0F

cd /d "%~dp0"

echo ========================================================
echo  🌌 [딥러닝 은하 분류 탐구실] 실행을 준비합니다.
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 goto NOPYTHON
goto HASPYTHON

:NOPYTHON
echo [경고] 파이썬(Python)이 설치되어 있지 않거나 환경 변수(PATH)에 등록되지 않았습니다!
echo.
echo [해결 방법]
echo 1. 인터넷 창에 python.org 에 접속하여 파이썬 최신 버전을 다운로드하세요.
echo 2. 설치 프로그램을 실행할 때 첫 화면 맨 밑에 있는
echo    "Add python.exe to PATH" 체크박스를 반드시 누르고 설치하세요!
echo 3. 설치가 완료되면 이 창을 닫고 다시 실행해 주세요.
echo.
pause
exit /b

:HASPYTHON
if not exist "main.py" goto NOFILE
goto RUNAPP

:NOFILE
echo [오류] 메인 실행 파일인 'main.py'를 찾을 수 없습니다.
echo 압축을 풀지 않고 실행하셨거나, 파일이 누락되었습니다.
echo 'main.py', 'ui.py', 'utils.py', 'contents.py' 4개의 파일이
echo 모두 같은 폴더에 있는지 확인해주세요.
echo.
pause
exit /b

:RUNAPP
echo  >> 필수 라이브러리 패키지를 점검하고 설치합니다...
echo  >> 최초 실행 시 1~3분 정도 소요될 수 있습니다. 창을 닫지 마세요!
echo  --------------------------------------------------------

python -m pip install --upgrade pip --quiet
python -m pip install streamlit numpy pandas scikit-learn tensorflow matplotlib seaborn pillow tf-keras

echo  --------------------------------------------------------
echo  >> 점검 완료! 프로그램 화면 UI를 띄웁니다.
echo  >> 잠시 후 인터넷 브라우저가 자동으로 열립니다.
echo.

python -m streamlit run main.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================================
    echo  [오류 발생] 프로그램을 실행하는 중 문제가 발생했습니다.
    echo ========================================================
    echo  위 화면에 출력된 에러 메시지를 확인해 주세요.
    echo.
    pause
) else (
    pause
)
