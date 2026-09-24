[딥러닝 은하 분류 탐구실 - Mac 실행 안내]

1. 이 폴더에 기존 프로그램의 galaxy_rne 폴더 전체를 복사하세요.
   최종 구조 예시:
   - 실행하기_Mac_안정판.command
   - main.py
   - ui.py
   - utils.py
   - contents.py
   - guides.py
   - galaxy_rne/
       - 0/
       - 1/
       - 2/
       - 3/
       - 4/

2. 실행하기_Mac_안정판.command 를 더블클릭합니다.
3. macOS가 차단하면 파일을 우클릭 > 열기 > 열기로 실행합니다.
4. '권한이 없습니다'가 나오면 터미널에서 아래처럼 한 번만 실행하세요.
   chmod +x 실행하기_Mac_안정판.command
5. 최초 실행은 Python 패키지 설치를 위해 인터넷 연결이 필요합니다.
6. 권장 Python: 3.12

* Apple Silicon(M1/M2/M3/M4/M5 등): 현재 TensorFlow 패키지를 설치합니다.
* Intel Mac: 호환성을 위해 TensorFlow 2.16.2를 설치합니다.
