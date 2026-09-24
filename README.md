# Galaxy-Classification-AI-Education

## 딥러닝 기반 은하 분류 AI 실습키트

실제 은하 이미지를 활용해 데이터 수집, 전처리·증강, CNN 학습, 정확도·혼동행렬·오분류 분석까지 경험하도록 개발한 Streamlit 기반 교육용 AI 실습 프로그램입니다.

### 관련 연구
**조훈 · 손정주, 「딥러닝 기반 은하 분류 교육 콘텐츠 개발」**

## 분류 대상

- Elliptical Galaxy
- Lens Galaxy
- Spiral Galaxy
- Barred Spiral Galaxy
- Irregular Galaxy

## 탐구 흐름

**문제 정의 → 데이터 수집 → 데이터 처리·탐색 → 데이터 분석 및 표현 → 일반화**

## 주요 소스

- `src/main.py`
- `src/ui.py`
- `src/utils.py`
- `src/contents.py`
- `src/guides.py`
- `src/실행하기.bat`
- `src/실행하기_Mac_안정판.command`

## 실행형 배포본

전체 은하 이미지 데이터가 포함된 Windows/macOS 배포 ZIP은 용량이 크기 때문에 저장소 본문이 아니라 **GitHub Releases**에 두는 구조를 사용합니다.

권장 Release asset 이름:

- `galaxy-ai-windows.zip`
- `galaxy-ai-macos.zip`

### Windows
압축 해제 후 `실행하기.bat` 실행

### macOS
압축 해제 후 `실행하기_Mac_안정판.command` 실행

## 소스에서 실행

전체 배포본의 `galaxy_rne/` 폴더를 `src/` 안에 둔 뒤:

```bash
pip install -r requirements.txt
cd src
streamlit run main.py
```

## 연구·교육 플랫폼

기존 공식 연구자료실:  
https://sites.google.com/view/astronomydatascience/

통합 GitHub Pages:  
https://GodTANKS.github.io/astronomy-data-science/

## 기존 프로젝트와의 구분

기존 `Galaxy`, `AI_galaxy`, `Galaxy_Classification_Deepleaning` 저장소는 이번 최신 실습키트의 공식 저장소로 사용하지 않습니다. 이 저장소가 최신 교육용 배포판의 공식 저장소입니다.
