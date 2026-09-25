# 🌌 딥러닝 기반 은하 분류 AI 실습키트

실제 은하 이미지를 이용해 **데이터 수집 → 전처리·증강 → CNN 학습 → 성능 평가 → 오분류 해석**까지 경험하는 Streamlit 기반 AI·천문학 교육 콘텐츠입니다.

> **권장 대상:** AI를 처음 배우는 학생·예비교사·교사  
> **사용 방식:** 완성 배포본은 실행 파일을 클릭하는 방식 / 소스 실행은 Python 사용 경험이 있는 경우 권장  
> **분류 대상:** 타원 · 렌즈 · 정상 나선 · 막대 나선 · 불규칙 은하

---

## 🌐 웹에서 바로 실행하는 Streamlit 버전

Streamlit 웹 버전은 **새로 단순화한 별도 앱이 아니라, 논문 제출 최종 실습키트의 화면·단계·학습 내용·CNN 설정을 그대로 실행**하도록 구성합니다.

웹 배포에서 달라지는 부분은 **이미지 파일을 읽는 방식뿐**입니다.

- 원본 Windows/macOS 실습키트: `galaxy_rne/0~4`의 로컬 이미지 사용
- Streamlit Cloud: 동일한 548장 이미지의 파일명을 유지한 채 기존 `GodTANKS/AI_galaxy` 저장소에서 필요한 이미지를 원격으로 불러옴
- 문제 정의, 데이터 수집, 수집 결과 요약, 데이터 처리·탐색, 처리 결과 요약, CNN 분석·표현, 일반화 단계는 최종 실습키트 구조를 유지
- `contents.py`의 학습목표·탐구질문·모범답안과 `guides.py`의 튜토리얼·CNN 개념·성능지표·실험 안내도 최종판을 사용
- 데이터 증강, 데이터 분할, CNN 레이어·필터·풀링·Dense·활성화함수·드롭아웃·학습률·에포크·콜백 설정과 혼동행렬/오분류 분석도 최종판 흐름을 유지

## 👥 다중 사용자 실습 안내

웹 버전은 여러 사용자의 **세션·수집 데이터·전처리 상태·CNN 학습 결과 파일이 서로 섞이지 않도록 분리**되어 있습니다. 따라서 여러 사람이 접속한다고 해서 학생별 데이터가 서로 덮어쓰는 구조는 아닙니다.

다만 웹에서 CNN을 학습할 때는 모든 사용자가 **같은 Streamlit 서버의 CPU·메모리 자원**을 공유합니다. 동시에 많은 사용자가 모델 학습을 실행하면 속도가 느려지거나 서버가 불안정해질 수 있습니다.

- **개인·소규모 실습:** 아래 Streamlit 웹 버전을 바로 이용
- **10명 이상의 수업·워크숍:** Windows/macOS 배포본을 내려받아 각 PC에서 개별 실행 권장

**[🌐 Streamlit 웹 실습 바로 시작](https://galaxy-ai-education.streamlit.app/)**

> `10명`은 Streamlit의 공식 접속 제한이 아니라, CNN 학습이 포함된 본 콘텐츠의 안정적인 수업 운영을 위한 권장 기준입니다.

---

## 📥 배포본·소스코드 다운로드

- **[🪟 Windows 실습키트 다운로드](https://github.com/GodTANKS/Galaxy-Classification-AI-Education/releases/latest/download/galaxy-ai-windows.zip)**
- **[🍎 macOS 실습키트 다운로드](https://github.com/GodTANKS/Galaxy-Classification-AI-Education/releases/latest/download/galaxy-ai-macos.zip)**
- **[📦 소스코드 저장소 전체 ZIP](https://github.com/GodTANKS/Galaxy-Classification-AI-Education/archive/refs/heads/main.zip)**
- **[🏷️ 최신 GitHub Release 보기](https://github.com/GodTANKS/Galaxy-Classification-AI-Education/releases/latest)**

---

## 🚀 가장 쉬운 실행 방법

### Windows
정식 배포본을 사용하는 경우:

1. `galaxy-ai-windows.zip`을 다운로드합니다.
2. ZIP을 **반드시 압축 해제**합니다.
3. 압축을 푼 폴더에서 `실행하기.bat`를 더블클릭합니다.
4. 최초 실행 시 필요한 Python 패키지를 자동 점검·설치합니다.
5. 잠시 후 웹브라우저에서 Streamlit 실습 화면이 열립니다.

### macOS
정식 배포본을 사용하는 경우:

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

## 📌 현재 공개 방식

이 저장소는 **최신 은하분류 AI 실습키트의 공식 저장소**입니다.

- **웹 실습:** Streamlit에서 설치 없이 바로 실행
- **단체수업·로컬 실습:** GitHub Releases의 Windows/macOS 완성 배포본 사용
- **개발·수정:** 저장소의 `src/` 소스코드 직접 사용
- **전체 은하 이미지:** 완성 배포본에 포함

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

논문 PDF는 코드 저장소에 중복 보관하지 않고 **통합 논문 모음**에서 관리합니다.

**[📚 통합 논문 모음에서 보기](https://GodTANKS.github.io/astronomy-data-science/papers/)**


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

---

## 📘 교육·학습 목적 이용 조건

이 저장소에서 **공개된 코드·노트북·교육 자료**는 원저자·원본 저장소·관련 논문 출처를 명시하는 조건으로 **교육·학습 및 비상업적 연구 목적의 복제·수정·재배포가 가능합니다.**

**상업적 판매·유료 서비스·출처 삭제·타인의 독창적 연구 결과인 것처럼 사용하는 행위는 허용하지 않습니다.**

자세한 조건: [EDUCATIONAL_USE_NOTICE.md](EDUCATIONAL_USE_NOTICE.md)

