# 🌌 딥러닝 기반 은하 분류 AI 실습키트

실제 은하 이미지를 이용해 **데이터 수집 → 전처리·증강 → CNN 학습 → 성능 평가 → 오분류 해석**까지 경험하는 Streamlit 기반 AI·천문학 교육 콘텐츠입니다.

> **권장 대상:** AI를 처음 배우는 학생·예비교사·교사  
> **사용 방식:** 완성 배포본은 실행 파일을 클릭하는 방식 / 소스 실행은 Python 사용 경험이 있는 경우 권장  
> **분류 대상:** 타원 · 렌즈 · 정상 나선 · 막대 나선 · 불규칙 은하

---

## 🚀 가장 쉬운 실행 방법

### Windows
완성 배포 ZIP이 GitHub Releases에 공개된 경우:

1. `galaxy-ai-windows.zip`을 다운로드합니다.
2. ZIP을 **반드시 압축 해제**합니다.
3. 압축을 푼 폴더에서 `실행하기.bat`를 더블클릭합니다.
4. 최초 실행 시 필요한 Python 패키지를 자동 점검·설치합니다.
5. 잠시 후 웹브라우저에서 Streamlit 실습 화면이 열립니다.

### macOS
완성 배포 ZIP이 GitHub Releases에 공개된 경우:

1. `galaxy-ai-macos.zip`을 다운로드합니다.
2. 압축을 해제합니다.
3. `실행하기_Mac_안정판.command`를 실행합니다.
4. macOS가 차단하면 파일을 **우클릭 → 열기 → 열기**로 실행합니다.
5. 권한 오류가 있으면 터미널에서 한 번만:
   ```bash
   chmod +x 실행하기_Mac_안정판.command
   ```
6. 최초 실행 시 필요한 환경을 준비한 뒤 브라우저가 열립니다.

> 권장 Python: **3.12**  
> Apple Silicon과 Intel Mac의 TensorFlow 호환성을 실행기에서 구분하도록 구성한 배포본을 사용합니다.

---

## 📌 현재 GitHub 공개 상태

이 저장소는 **최신 은하분류 AI 실습키트의 공식 저장소**입니다.

현재는 소스와 실행 안내를 정리하는 단계이며, 전체 은하 이미지 데이터(`galaxy_rne`)와 Windows/macOS 완성 배포 ZIP은 **GitHub Releases**에서 제공하는 구조로 설계했습니다.

따라서 처음 방문한 사용자는:

- Releases에 완성 ZIP이 있으면 → **ZIP 다운로드 후 실행기 클릭**
- Releases가 아직 비어 있으면 → 배포본 공개 전 상태
- 소스 코드를 직접 실행하려면 → 아래 개발자용 실행 방법 참고

---

## 🧠 실습에서 무엇을 하나요?

### 1. 문제 정의
- 은하 형태 분류의 기준 이해
- 사람의 분류와 AI 분류의 차이 생각하기

### 2. 데이터 수집
- 5개 은하 유형의 이미지 선택
- 유형별 데이터 수 확인

### 3. 데이터 처리·탐색
- 이미지 크기 표준화
- 컬러/흑백 처리
- 회전·대칭 등 데이터 증강
- 처리 결과 미리보기

### 4. 데이터 분석 및 표현
- CNN 구조 설정
- 필터 수, 풀링 크기, Dense 레이어 설정
- 학습률, 배치 크기, 에포크 수 조절
- 실시간 Loss/Accuracy 확인
- 혼동행렬과 평가 지표 확인

### 5. 일반화
- 어떤 은하가 왜 오분류되는지 해석
- 모델의 한계와 데이터의 영향 토의
- 데이터 윤리와 AI 결과 해석 성찰

---

## 📁 주요 소스

| 파일 | 역할 |
|---|---|
| `src/main.py` | Streamlit 앱 시작 파일 |
| `src/ui.py` | 단계별 화면과 학습 인터페이스 |
| `src/utils.py` | 이미지 처리·CNN·평가 함수 |
| `src/contents.py` | 학습 목표·질문·탐구 내용 |
| `src/guides.py` | 튜토리얼·CNN 개념·평가 지표 설명 |
| `src/실행하기.bat` | Windows 실행기 |
| `src/실행하기_Mac_안정판.command` | macOS 실행기 |
| `galaxy_rne/0~4` | 은하 이미지 데이터 |

---

## 💻 소스에서 직접 실행하기

완성 배포본의 `galaxy_rne` 폴더를 `src/` 안에 둔 뒤:

```text
src/
├─ main.py
├─ ui.py
├─ utils.py
├─ contents.py
├─ guides.py
└─ galaxy_rne/
   ├─ 0/
   ├─ 1/
   ├─ 2/
   ├─ 3/
   └─ 4/
```

프로젝트 루트에서:

```bash
pip install -r requirements.txt
cd src
streamlit run main.py
```

---

## 🔢 은하 유형 폴더 번호

| 폴더 | 은하 유형 |
|---|---|
| 0 | Elliptical Galaxy |
| 1 | Lens Galaxy |
| 2 | Spiral Galaxy |
| 3 | Barred Spiral Galaxy |
| 4 | Irregular Galaxy |

---

## ❓ 자주 발생하는 문제

**Q. 프로그램 화면은 열리는데 이미지가 없습니다.**  
A. `src/galaxy_rne` 폴더가 있는지 확인하세요. 이미지 데이터가 없으면 2단계 이후 실습을 진행할 수 없습니다.

**Q. Windows에서 bat 파일이 실행되지 않습니다.**  
A. ZIP 안에서 바로 실행하지 말고 먼저 전체 압축을 해제한 뒤 실행하세요. Python이 설치되어 있지 않다면 Python 설치 시 **Add Python to PATH**를 체크하세요.

**Q. Mac에서 “확인되지 않은 개발자” 경고가 나옵니다.**  
A. Finder에서 command 파일을 우클릭한 뒤 **열기**를 선택하세요.

**Q. TensorFlow 설치가 오래 걸립니다.**  
A. 최초 실행에서는 정상일 수 있습니다. 인터넷 연결을 유지하고 터미널 창을 닫지 마세요.

---

## 📄 관련 연구

**조훈 · 손정주, 「딥러닝 기반 은하 분류 교육 콘텐츠 개발」**

프로그래밍 의존도를 낮추면서 CNN과 데이터 리터러시를 함께 경험하도록 설계한 교육 콘텐츠입니다.

---

## 🌐 통합 연구·교육 플랫폼

**AI · 데이터 사이언스로 탐구하는 천문학**  
https://GodTANKS.github.io/astronomy-data-science/

기존 연구 아카이브:  
https://sites.google.com/view/astronomydatascience/

---

## 기존 저장소와의 구분

기존 `Galaxy`, `AI_galaxy`, `Galaxy_Classification_Deepleaning` 저장소는 이번 최신 실습키트의 공식 저장소로 사용하지 않습니다.  
**이 저장소가 최신 교육용 배포판의 공식 저장소입니다.**
