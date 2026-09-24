import streamlit as st
import ui

def main():
    # 1. 기본 설정 및 CSS 디자인 적용 (디자인 유지)
    st.set_page_config(
        page_title="딥러닝 기반 은하 분류 탐구",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Noto Sans KR', sans-serif; }
        .main-title { font-size: 3rem; font-weight: 700; color: #4B0082; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px #cccccc; }
        .step-header { font-size: 1.8rem; font-weight: 600; color: #2E86C1; border-bottom: 2px solid #2E86C1; padding-bottom: 10px; margin-bottom: 20px; }
        .goal-box { background-color: #f0f2f6; border-left: 5px solid #2E86C1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🌌 딥러닝을 활용한 은하 분류 분석</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: gray; margin-bottom: 30px; font-size: 0.9rem;">
    1.문제 정의 &nbsp; ➤ &nbsp; 2.데이터 수집 &nbsp; ➤ &nbsp; 3.전처리 &nbsp; ➤ &nbsp; 4.탐색 &nbsp; ➤ &nbsp; 5.분석 및 표현 &nbsp; ➤ &nbsp; 6.일반화
    </div>
    """, unsafe_allow_html=True)

    BASE_FOLDER = "galaxy_rne"

    # [수정됨] 다시 원본처럼 영문 이름으로만 표기되도록 복구
    FOLDER_NAMES = {
        "0": "Elliptical Galaxy",
        "1": "Lens Galaxy",
        "2": "Spiral Galaxy",
        "3": "Barred Spiral Galaxy",
        "4": "Irregular Galaxy",
    }

    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = None
    if 'collected_data' not in st.session_state:
        st.session_state.collected_data = {key: [] for key in FOLDER_NAMES.keys()}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {key: [] for key in FOLDER_NAMES.keys()}
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = {}

    st.sidebar.header("🚀 탐구 단계 선택")
    step = st.sidebar.radio(
        "단계를 선택하세요",
        ["1단계: 문제 정의", "2단계: 데이터 수집", "2단계: 데이터 수집 결과 요약",
         "3-4단계: 데이터 처리 및 탐색", "3-4단계: 데이터 처리 및 탐색 결과 요약",
         "5단계: 데이터 분석 및 표현", "6단계: 일반화"],
        index=0
    )
    st.sidebar.markdown("---")

    if step == "1단계: 문제 정의": ui.step_1()
    elif step == "2단계: 데이터 수집": ui.step_2(BASE_FOLDER, FOLDER_NAMES)
    elif step == "2단계: 데이터 수집 결과 요약": ui.step_3(FOLDER_NAMES)
    elif step == "3-4단계: 데이터 처리 및 탐색": ui.step_4(BASE_FOLDER, FOLDER_NAMES)
    elif step == "3-4단계: 데이터 처리 및 탐색 결과 요약": ui.step_4_5_summary(FOLDER_NAMES)
    elif step == "5단계: 데이터 분석 및 표현": ui.step_5(FOLDER_NAMES)
    elif step == "6단계: 일반화": ui.step_6()

if __name__ == "__main__":
    main()