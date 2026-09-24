import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re
import time
import platform
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import contents
import utils
import guides

# 동일 조건 재현을 위한 결정적 연산 설정
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

IMAGE_SIZE = 50
# ==============================================================================
# [중요] Matplotlib 한글 깨짐(네모 폰트) 방지 글로벌 설정
# ==============================================================================
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 공통 UI 헬퍼 함수
# ==============================================================================
def render_header(step_dict):
    st.markdown(f'<div class="step-header">{step_dict["title"]}</div>', unsafe_allow_html=True)

    # 괄호() 안의 글자를 자동으로 볼드체 처리
    formatted_goals = []
    for g in step_dict["goals"]:
        formatted_g = re.sub(r'(\([^)]+\))', r'<strong>\1</strong>', g, count=1)
        formatted_goals.append(formatted_g)

    goals_html = "<br>".join(formatted_goals)
    st.markdown(f"""
    <div class="goal-box">
    <strong>🎯 학습 목표</strong><br>
    {goals_html}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 탐구 질문 및 힌트 열기"):
        for i, q in enumerate(step_dict["questions"]):
            st.write(f"**{q['q']}**")
            if st.checkbox(f"힌트 보기 (질문{i + 1})", key=f"{step_dict['title']}_hint_{i}"):
                st.info(f"{q['hint']}")
            st.write("")

        if st.checkbox("🔑 모범 답안 보기", key=f"{step_dict['title']}_ans"):
            st.markdown("---")
            for q in step_dict["questions"]:
                st.success(f"**[{q['q'].split('.')[0]}]**\n{q['ans']}")
    st.markdown("---")


# ==============================================================================
# 각 단계별 UI 함수
# ==============================================================================

def step_1():
    render_header(contents.STEP_1)
    st.info(contents.NOTICE_TEXT)

    st.markdown("### 📝 문제 정의 튜토리얼")
    st.info(guides.STEP_1_TUTORIAL)


def step_2(base_folder, folder_names):
    render_header(contents.STEP_2)

    st.sidebar.markdown("### 📂 은하 유형 선택")
    if st.sidebar.button("🗑️ 수집된 데이터 전체 초기화", type="primary"):
        st.session_state.collected_data = {key: [] for key in folder_names.keys()}
        st.session_state.processed_data = {key: [] for key in folder_names.keys()}
        st.session_state.processing_log = {}
        st.rerun()

    for folder_key, folder_label in folder_names.items():
        if st.sidebar.button(folder_label):
            st.session_state.selected_folder = folder_key

    st.markdown("### 🛠️ 데이터 수집 튜토리얼")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(guides.STEP_2_TUTORIAL)

    with col2:
        if st.session_state.selected_folder is not None:
            selected_folder = st.session_state.selected_folder
            folder_label = folder_names[selected_folder]
            st.markdown(f"#### 📡 **{folder_label}** 데이터 수집 중...")

            images = utils.get_local_image_files(base_folder, str(selected_folder))

            if images:
                image_range = st.text_area("이미지 번호 입력", height=100, placeholder="예: 이미지1-이미지10, 이미지15")
                if st.button(f"📥 {folder_label} 이미지 수집", type="primary"):
                    if image_range:
                        try:
                            selected_images = []
                            for part in image_range.split(","):
                                part = part.strip()
                                if "-" in part:
                                    start, end = map(int, part.replace("이미지", "").split("-"))
                                    selected_images.extend(range(start, end + 1))
                                else:
                                    selected_images.append(int(part.replace("이미지", "")))

                            count = 0
                            for index in selected_images:
                                if index <= len(images):
                                    img_name = images[index - 1]
                                    if img_name not in st.session_state.collected_data[selected_folder]:
                                        st.session_state.collected_data[selected_folder].append(img_name)
                                        count += 1
                            if count > 0:
                                st.toast(f"✅ 총 {count}장의 이미지가 성공적으로 수집되었습니다!", icon="🎉")
                                st.success(f"총 {count}장의 이미지가 리스트에 추가되었습니다.")
                            else:
                                st.warning("이미 수집된 이미지이거나 잘못된 번호입니다.")
                        except ValueError:
                            st.error("입력 형식이 잘못되었습니다. 예: 이미지1-이미지10, 이미지15")

                st.markdown(f"#### 🖼️ {folder_label} 폴더의 전체 이미지")
                img_cols = st.columns(5)

                for i, image_name in enumerate(images):
                    with img_cols[i % 5]:
                        try:
                            img = utils.open_source_image(base_folder, selected_folder, image_name)
                            st.image(img, caption=image_name, use_container_width=True)
                        except Exception:
                            pass
            else:
                st.warning(f"⚠️ {folder_label} 폴더에 이미지가 없습니다.")
        else:
            st.info("👈 **좌측 사이드바에서 수집할 은하 유형을 먼저 선택해 주세요.**")


def step_3(folder_names):
    st.markdown('<div class="step-header">📊 2-1단계: 데이터 수집 결과 요약</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ 전체 초기화", type="primary", key="reset_step3"):
            st.session_state.collected_data = {key: [] for key in folder_names.keys()}
            st.session_state.processed_data = {key: [] for key in folder_names.keys()}
            st.session_state.processing_log = {}
            st.rerun()

    st.write("### 📦 수집된 은하 데이터 현황")
    for folder_key, label in folder_names.items():
        st.subheader(f"📂 {label} ({folder_key})")
        count = len(st.session_state.collected_data[folder_key])
        if count > 0:
            st.write(f"- **수집된 데이터 수**: {count}개")
        else:
            st.write("- **수집된 데이터 수**: 0개 (수집되지 않음)")
        st.divider()
    st.info("데이터가 충분히 수집되었나요? 그렇다면 **3-4단계: 데이터 처리 및 탐색**으로 이동하세요.")


def step_4(base_folder, folder_names):
    render_header(contents.STEP_3_4)

    st.markdown("### 🛠️ 데이터 처리 및 탐색 튜토리얼")
    st.info(guides.STEP_4_TUTORIAL)
    st.markdown("---")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("### 🛠️ 전처리 옵션 설정")
        if any(len(imgs) > 0 for imgs in st.session_state.collected_data.values()):
            if st.button("🔄 전처리 초기화", type="primary", key="reset_proc_main"):
                st.session_state.processed_data = {key: [] for key in folder_names.keys()}
                st.session_state.processing_log = {}
                st.rerun()

            selected_folder = st.selectbox("처리할 은하 선택",
                                           options=[k for k, v in st.session_state.collected_data.items() if v],
                                           format_func=lambda x: folder_names[x])

            st.markdown("#### 1. 기본 처리")
            resize_option = st.checkbox(
                f"이미지 크기 표준화 "
                f"({utils.IMAGE_SIZE}x{utils.IMAGE_SIZE}) "
                f"(미선택시 오류발생)",
                value=True
            )
            is_color = st.checkbox("칼라로 이미지 처리 (미선택시 흑백)", value=False)

            st.markdown("#### 2. 데이터 증강 (Augmentation)")
            full_opts = ["모두 선택", "45도 회전", "90도 회전", "135도 회전", "180도 회전", "225도 회전", "270도 회전", "315도 회전", "좌우 대칭",
                         "좌우 대칭 및 45도 회전", "좌우 대칭 및 90도 회전", "좌우 대칭 및 135도 회전", "좌우 대칭 및 180도 회전", "좌우 대칭 및 225도 회전",
                         "좌우 대칭 및 270도 회전", "좌우 대칭 및 315도 회전"]
            transformations = st.multiselect("추가할 이미지 변환 선택", full_opts, default=[])
            if "모두 선택" in transformations: transformations = full_opts[1:]

            if st.button("데이터 처리 시작", type="primary"):
                with st.spinner("이미지 처리 중..."):
                    img_names = st.session_state.collected_data[selected_folder]
                    # [UI 개선] 4단계에서는 학생들이 결과를 눈으로 볼 수 있게 '증강이 적용된' 전체 데이터를 저장합니다.
                    processed = utils.process_images(selected_folder, base_folder, img_names, resize_option,
                                                     transformations, is_color)
                    st.session_state.processed_data[selected_folder] = processed
                    st.session_state.processing_log[selected_folder] = {"resize": resize_option, "is_color": is_color,
                                                                        "augmentations": transformations}
                st.success(f"데이터 처리 결과 {len(processed)}개의 학습용 이미지가 준비되었습니다.")
        else:
            st.error("수집된 데이터가 없습니다. 2단계에서 데이터를 먼저 수집하세요.")

    with col_r:
        st.markdown("### 🖼️ 처리 결과 미리보기")
        if any(len(imgs) > 0 for imgs in st.session_state.processed_data.values()):
            if 'selected_folder' in locals() and st.session_state.processed_data[selected_folder]:
                p_imgs = st.session_state.processed_data[selected_folder]
                st.write(f"**{folder_names[selected_folder]}** (총 {len(p_imgs)}장)")
                grid = st.columns(5)
                for i, img_data in enumerate(p_imgs[:20]):
                    with grid[i % 5]:
                        st.image(img_data, use_container_width=True, clamp=True)
                if len(p_imgs) > 20: st.caption("... (생략)")
            else:
                st.info("왼쪽에서 옵션을 선택하고 '처리 시작'을 눌러주세요.")
        else:
            st.info("데이터 처리가 아직 수행되지 않았습니다.")


def step_4_5_summary(folder_names):
    st.markdown('<div class="step-header">📋 3-4단계: 데이터 처리 결과 요약</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 전처리 초기화", key="reset_in_summary", type="primary"):
            st.session_state.processed_data = {key: [] for key in folder_names.keys()}
            st.session_state.processing_log = {}
            st.rerun()

    st.markdown("### 📊 은하 유형별 데이터 준비 현황")
    tabs = st.tabs(list(folder_names.values()))
    for i, (f_key, label) in enumerate(folder_names.items()):
        with tabs[i]:
            count = len(st.session_state.processed_data.get(f_key, []))
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("준비된 데이터 수", f"{count}개")
            with c2:
                if f_key in st.session_state.processing_log:
                    log = st.session_state.processing_log[f_key]
                    st.markdown("##### 🛠️ 적용된 처리 옵션")
                    size_text = (
                        f"{utils.IMAGE_SIZE}x{utils.IMAGE_SIZE} 표준화"
                        if log["resize"]
                        else "원본 크기"
                    )

                    st.write(f"✅ **크기**: {size_text}")
                    st.write(f"✅ **색상**: {'RGB (컬러)' if log['is_color'] else 'Grayscale (흑백)'}")
                    if log['augmentations']:
                        st.info(f"🔄 **적용된 증강 기법**:\n\n{', '.join(log['augmentations'])}")
                    else:
                        st.caption("🚫 적용된 증강 기법 없음")
                else:
                    st.warning("아직 데이터 처리가 수행되지 않았습니다.")
    st.markdown("---")
    st.info("모든 데이터가 준비되었다면, **5단계: 데이터 분석 및 표현**으로 이동하여 AI 모델을 훈련시키세요!")


# ==============================================================================
# 실시간 그래프 (축 라벨 추가) 및 표 출력 콜백
# ==============================================================================
class RealTimePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.chart_loc = st.empty()
        self.table_loc = st.empty()  # 실시간 표 위치
        self.df = pd.DataFrame(columns=["Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(min(progress, 1.0))

        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            lr = 0.0

        new_row = {"Epoch": epoch + 1, "LR": lr, "Train Loss": logs.get('loss'), "Train Acc": logs.get('accuracy'),
                   "Val Loss": logs.get('val_loss'), "Val Acc": logs.get('val_accuracy')}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        with self.chart_loc.container():
            c1, c2 = st.columns(2)
            fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
            ax_loss.plot(self.df["Epoch"], self.df["Train Loss"], label="Train", color='blue')
            ax_loss.plot(self.df["Epoch"], self.df["Val Loss"], label="Val", color='red', linestyle='--')
            ax_loss.set_title("Loss Graph")
            ax_loss.set_xlabel("Epoch (학습 횟수)")
            ax_loss.set_ylabel("Loss (손실값)")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            c1.pyplot(fig_loss)

            fig_acc, ax_acc = plt.subplots(figsize=(5, 3))
            ax_acc.plot(self.df["Epoch"], self.df["Train Acc"], label="Train", color='green')
            ax_acc.plot(self.df["Epoch"], self.df["Val Acc"], label="Val", color='orange', linestyle='--')
            ax_acc.set_title("Accuracy Graph")
            ax_acc.set_xlabel("Epoch (학습 횟수)")
            ax_acc.set_ylabel("Accuracy (정확도)")
            ax_acc.set_ylim(0.0, 1.0)
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
            c2.pyplot(fig_acc)

        # 실시간 표 업데이트
        self.table_loc.dataframe(
            self.df.style.format({
                "LR": lambda x: np.format_float_positional(
                    np.float32(x),
                    unique=True,
                    trim='-'
                ),
                "Train Loss": "{:.4f}",
                "Train Acc": "{:.4f}",
                "Val Loss": "{:.4f}",
                "Val Acc": "{:.4f}"
            }),
            use_container_width=True
        )
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} 완료")


def step_5(folder_names):
    render_header(contents.STEP_5)

    # guides.py에 저장된 상세한 설명문 불러오기 (HTML 태그 포함 부분만 unsafe_allow_html 유지)
    with st.expander("📚 CNN 주요 개념 및 하이퍼파라미터 설명 보기"):
        st.markdown(guides.STEP_5_CNN_GUIDE, unsafe_allow_html=True)

    with st.expander("📊 성능 지표 (혼동행렬, 정확도 등) 설명 보기"):
        st.markdown(guides.STEP_5_METRICS_GUIDE, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ 데이터 분석 및 표현 튜토리얼")
    st.info(guides.STEP_5_TUTORIAL)
    st.markdown("---")

    total_images = sum([len(imgs) for imgs in st.session_state.processed_data.values()])
    if total_images == 0:
        st.error("❌ 학습할 데이터가 없습니다. 3-4단계에서 데이터를 먼저 처리해주세요.")
        return

    with st.expander("📝 실험 결과 기록지 펼치기", expanded=False):
        st.markdown("본인의 탐구 목표에 맞는 실험표를 골라 하이퍼파라미터 조건을 확인하고, 모델 훈련 결과를 직접 표에 입력해 보세요.")
        tab_e1, tab_e2, tab_a1, tab_a2 = st.tabs(["[필수1] 데이터 증강", "[필수2] 학습률", "[심화1] 파라미터 수", "[심화2] 풀링 크기"])

        with tab_e1:
            st.info(guides.EXP_GUIDE_1)
            st.dataframe(contents.exp1_cond_df, hide_index=True, use_container_width=True)
            edited_df_e1 = st.data_editor(contents.exp1_result_df, key="editor_e1", hide_index=True,
                                          use_container_width=True)
            csv_e1 = edited_df_e1.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 [필수1] 결과표 다운로드", data=csv_e1, file_name="실험1_데이터증강_결과.csv",
                               mime="text/csv", key="dl_e1")

        with tab_e2:
            st.info(guides.EXP_GUIDE_2)
            st.dataframe(contents.exp2_cond_df, hide_index=True, use_container_width=True)
            edited_df_e2 = st.data_editor(contents.exp2_result_df, key="editor_e2", hide_index=True,
                                          use_container_width=True)
            csv_e2 = edited_df_e2.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 [필수2] 결과표 다운로드", data=csv_e2, file_name="실험2_학습률_결과.csv",
                               mime="text/csv", key="dl_e2")

        with tab_a1:
            st.info(guides.EXP_GUIDE_A1)
            st.dataframe(contents.adv1_cond_df, hide_index=True, use_container_width=True)
            edited_df_a1 = st.data_editor(contents.adv1_result_df, key="editor_a1", hide_index=True,
                                          use_container_width=True)
            csv_a1 = edited_df_a1.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 [심화1] 결과표 다운로드", data=csv_a1, file_name="심화1_파라미터수_결과.csv",
                               mime="text/csv", key="dl_a1")

        with tab_a2:
            st.info(guides.EXP_GUIDE_A2)
            st.dataframe(contents.adv2_cond_df, hide_index=True, use_container_width=True)
            edited_df_a2 = st.data_editor(contents.adv2_result_df, key="editor_a2", hide_index=True,
                                          use_container_width=True)
            csv_a2 = edited_df_a2.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 [심화2] 결과표 다운로드", data=csv_a2, file_name="심화2_풀링크기_결과.csv",
                               mime="text/csv", key="dl_a2")

        st.info("💡 위의 표를 보면서, 좌측 사이드바(🎛️ 하이퍼파라미터 설정)에서 값을 동일하게 맞춰 훈련을 진행하세요.")

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 하이퍼파라미터 설정")

    # =========================================================
    # 1. 모델 구성
    # =========================================================
    st.sidebar.subheader("🛠️ 1. 모델 구성")

    # --- [전반부] 특징 추출기(CNN) ---
    st.sidebar.markdown("#### **[전반부] 특징 추출기(CNN)**")
    conv_layers = st.sidebar.slider(
        "전반부 합성곱 레이어 개수 (Convolutional Layers)", 1, 5, 3,
        help="이미지를 정독하는 횟수입니다. 레이어가 많을수록 은하의 복잡한 특징을 깊게 이해할 수 있지만, 너무 많으면 특정 데이터만 암기하는 과적합 위험이 있습니다."
    )

    filters, pool_sizes, acts = [], [], []
    filter_options = [8, 16, 32, 64, 128, 256, 512]
    pool_options = [2, 3, 4]

    for i in range(conv_layers):
        filters.append(st.sidebar.selectbox(
            f"전반부 필터(채널) 개수 (Filters, 예: 32는 이미지 채널 32개 의미) (레이어 {i + 1})",
            filter_options, index=2 if i == 0 else 3, key=f"f{i}",
            help="이미지에서 특징을 찾아내는 '안경'의 개수입니다. 필터가 많을수록 나선팔, 핵, 막대 등 다양한 시각적 정보를 동시에 학습할 수 있습니다."
        ))
        pool_sizes.append(st.sidebar.selectbox(
            f"전반부 최대 풀링 크기 (Max Pooling Size, 예: 2는 2x2 픽셀 의미) (레이어 {i + 1})",
            pool_options, index=0, key=f"p{i}",
            help="이미지 내용을 요약해 핵심만 뽑아내는 과정의 크기입니다. 불필요한 정보를 줄여 학습 효율을 높이고 모델을 더 똑똑하게 만듭니다."
        ))
        acts.append(st.sidebar.selectbox(
            f"전반부 활성화 함수 (Activation Function) (레이어 {i + 1})",
            ["relu", "sigmoid", "tanh", "softmax"], index=0, key=f"a{i}",
            help="추출된 정보를 다음 단계로 넘길지 말지 결정하는 판단 기준입니다. 은하 분류에는 주로 'relu'가 효과적입니다."
        ))

    # --- [후반부] 분류기 ---
    st.sidebar.markdown("#### **[후반부] 분류기**")
    dense_layers = st.sidebar.slider(
        "마지막을 제외한 후반부 완전 연결(Dense) 레이어 개수", 1, 3, 2,
        help="추출된 시각적 특징들을 모아 최종 판단을 내리는 '생각하는 사람들(독자)'의 층수입니다."
    )

    d_units, d_acts = [], []
    for i in range(dense_layers):
        d_units.append(st.sidebar.selectbox(
            f"후반부 완전 연결(Dense) 레이어 {i + 1} 크기",
            filter_options, index=3, key=f"du{i}",
            help="각 판단 층에서 정보를 처리하는 노드(뉴런)의 수입니다. 많을수록 더 복잡한 관계를 파악할 수 있습니다."
        ))
        d_acts.append(st.sidebar.selectbox(
            f"후반부 완전 연결(Dense) 레이어 {i + 1} 활성화 함수",
            ["relu", "sigmoid", "tanh", "softmax"], index=0, key=f"da{i}"
        ))

    last_act = st.sidebar.selectbox(
        "마지막 완전 연결(Dense) 레이어의 활성화 함수 (Activation Function for Last Dense)",
        ["softmax", "relu", "sigmoid", "tanh"], index=0,
        help="최종적으로 5가지 은하 유형 중 어느 것일지 확률을 계산하는 방식입니다. 다중 분류에는 주로 'softmax'를 사용합니다."
    )

    # =========================================================
    # 2. 학습 설정
    # =========================================================
    st.sidebar.subheader("🚀 2. 학습 설정")
    batch_size = st.sidebar.slider(
        "배치 크기 (Batch Size)", 8, 128, 16,
        help="AI에게 한 번에 던져주는 문제지의 양입니다. 너무 많으면 학습이 빠르지만 메모리를 많이 쓰고, 적으면 꼼꼼하지만 시간이 오래 걸립니다."
    )
    epochs = st.sidebar.slider(
        "에포크 수 (Epochs)", 1, 100, 100,
        help="전체 데이터를 몇 번 반복해서 복습할지 결정합니다. 너무 적으면 공부가 부족하고, 너무 많으면 과적합이 발생할 수 있습니다."
    )
    learning_rate = st.sidebar.selectbox(
        "학습률 (Learning Rate)", [0.1, 0.01, 0.001, 0.0001, 0.00001], index=2,
        help="정답을 향해 나아가는 보폭입니다. 보폭이 너무 크면 정답을 지나치고, 너무 작으면 배우는 데 너무 오래 걸립니다."
    )

    model_seed = 2024

    st.sidebar.markdown("**🚫 드롭아웃 (과적합 방지)**")
    dropout_layers = st.sidebar.slider(
        "드롭아웃(Dropout) 레이어 개수", 0, 3, 0,
        help="학습 중 일부 뉴런을 의도적으로 꺼서 모델이 데이터를 통째로 외우는 것(과적합)을 방지하는 감독 장치입니다."
    )
    dropout_rate = st.sidebar.slider(
        "드롭아웃(Dropout) 비율", 0.0, 1.0, 0.0, step=0.05,
        help="훈련 시 무작위로 쉴(꺼질) 뉴런의 비율을 정합니다."
    )

    st.sidebar.markdown("**💡 콜백 (Callback)**")
    st.sidebar.info(
        "모델이 학습하는 과정을 실시간으로 모니터링하며, 성능이 더 이상 개선되지 않을 때 **자동으로 개입하여 학습을 제어**하는 기능입니다.\n\n"
        "불필요한 반복 학습을 멈춰 과적합을 방지하거나, 정답에 가까워졌을 때 학습 보폭을 줄여 모델의 최종 성능을 끌어올립니다."
    )

    st.sidebar.markdown("**1️⃣ 조기 종료 (Early Stopping)**")
    st.sidebar.caption("성능 개선이 멈추면 학습을 조기에 종료하여 과적합과 시간 낭비를 막습니다.")
    es_pat = st.sidebar.slider(
        "기다리는 횟수 (Patience)", 1, 30, 15,
        help="검증 정확도가 오르지 않더라도, 학습을 강제 종료하기 전까지 지켜보는 에포크(Epoch) 횟수입니다."
             "이를 통해 모델이 훈련 데이터에만 과도하게 맞춰지는 과적합(Overfitting)을 방지합니다."
    )

    st.sidebar.markdown("**2️⃣ 학습률 감소 (Reduce LR)**")
    st.sidebar.caption("학습이 정체될 때 보폭(학습률)을 줄여 더 세밀하게 최적값을 탐색합니다.")
    rl_pat = st.sidebar.slider(
        "감소 전 대기 횟수", 1, 10, 3,
        help="검증 손실값(Loss)이 줄어들지 않고 정체될 때, 학습률을 삭감하기 전까지 기다려주는 에포크 횟수입니다."
    )
    reduce_lr_factor = st.sidebar.select_slider(
        "학습률 감소 비율 (factor)", options=[0.1, 0.3, 0.5, 0.7, 0.9], value=0.1,
        help="학습 정체 시 기존 학습률을 어느 정도로 줄일지 결정하는 비율입니다."
             "예를 들어 0.3이면, 기존 학습 보폭의 30% 수준으로 좁혀서 최적의 가중치를 더 정밀하게 찾아냅니다."
    )
    reduce_lr_min = st.sidebar.selectbox(
        "최저 학습률 제한 (min_lr)", options=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15],
        index=11,
        help="학습률이 계속 감소하더라도 더 이상 작아지지 않도록 막아주는 하한선입니다."
             "학습이 아예 멈춰버리는 것을 방지합니다."
    )

    st.markdown(f"""
    📌 **본 모델에는 최적 성능을 위해 다음과 같은 콜백(Call back)이 설정되었습니다.**
    - **조기 종료**: 검증 정확도가 **{es_pat} 에포크** 동안 개선되지 않으면 학습을 자동 중단합니다.
    - **학습률 감소**: 검증 손실값이 **{rl_pat} 에포크** 동안 개선되지 않으면 **학습률을 기존의 {int(reduce_lr_factor * 100)}%로 감소**시켜 더 세밀하게 정답을 탐색합니다.
    """)

    st.subheader("1. 데이터셋 분할")
    c1, c2 = st.columns(2)
    with c1:
        t_pct = st.slider("훈련 데이터 비율 (%)", 60, 100, 80)
    with c2:
        v_pct = st.slider("검증 데이터 비율 (%)", 0, 40, 10)
    test_pct = 100 - (t_pct + v_pct)

    if test_pct < 0:
        st.error("🚨 비율 합계가 100%를 넘었습니다! 슬라이더를 조절해주세요.")
        return

    st.progress(t_pct / 100)
    st.caption(f"🔵 훈련({t_pct}%) | 🟠 검증({v_pct}%) | 🔴 테스트({test_pct}%)")

    # =========================================================
    # [핵심] 데이터 누수 방지 및 층화 추출 (수정됨)
    # =========================================================
    images, labels = [], []

    # 5단계 분할 시, 증강된 processed_data가 아닌 원본 collected_data를 기반으로 순수 원본 이미지만 불러옴
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder = "galaxy_rne"  # main.py와 동일하게 고정

    for folder_key, img_names in st.session_state.collected_data.items():
        if not img_names:
            continue

        # 해당 폴더의 4단계 처리 로그(크기/색상) 가져오기
        log = st.session_state.processing_log.get(folder_key, {"resize": True, "is_color": False})

        # '순수 원본(증강 제외)' 상태로 이미지만 다시 로드
        pure_base_imgs = utils.process_images(folder_key, base_folder, img_names, log["resize"], [], log["is_color"])

        images.extend(pure_base_imgs)
        labels.extend([int(folder_key)] * len(pure_base_imgs))

    is_color = True if len(images[0].shape) == 3 else False
    if is_color:
        X = np.array(images, dtype=np.float32) / 255.0
    else:
        X = np.array(images, dtype=np.float32).reshape(
            -1,
            utils.IMAGE_SIZE,
            utils.IMAGE_SIZE,
            1
        ) / 255.0
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=5)

    v_t_pct = 100 - t_pct
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=v_t_pct / 100, random_state=42, stratify=y)

    if len(X_temp) > 0 and test_pct > 0:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_pct / (v_pct + test_pct),
                                                        random_state=42, stratify=y_temp)
    else:
        X_val, y_val = X_temp, y_temp
        X_test, y_test = [], []

    with st.spinner("훈련 데이터에 이미지 증강을 적용하여 양을 늘리고 있습니다..."):
        X_train, y_train = utils.augment_train_data(X_train, y_train, st.session_state.processing_log)

    c1, c2, c3 = st.columns(3)
    c1.metric("훈련 데이터 (증강 후)", f"{len(X_train)}개")
    c2.metric("검증 데이터 (순수 원본)", f"{len(X_val)}개")
    c3.metric("테스트 데이터 (순수 원본)", f"{len(X_test)}개")

    # =========================================================
    # [실험 재현 설정] 모델 초기 가중치 및 학습 난수 기준값 적용
    # =========================================================
    tf.keras.backend.clear_session()

    SEED = int(model_seed)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # st.caption(
    #     f"🔒 데이터 분할 Seed: 42 (고정) | 🔁 모델 학습 Seed: {SEED}"
    # )

    model = utils.build_cnn_model(is_color, conv_layers, filters, pool_sizes, acts, dense_layers, d_units, d_acts,
                                  dropout_layers, dropout_rate, last_act)

    st.subheader("2. 모델 요약")
    df_info, txt_info = utils.get_layer_details_korean(model)
    with st.expander("모델 레이어 정보 펼치기"):
        st.dataframe(df_info, use_container_width=True)
        st.markdown(txt_info)
    st.markdown("---")

    if st.button("🔥 모델 훈련 시작 (Start Training)", type="primary"):
        st.write("### 3. 실시간 학습 현황")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.0
            ),
            metrics=["accuracy"]
        )

        callbacks = [RealTimePlotCallback(epochs)] + utils.get_callbacks(es_pat, rl_pat, reduce_lr_factor,
                                                                         reduce_lr_min)

        start_time = time.time()
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=0
        )
        end_time = time.time()

        st.success("✅ 훈련 완료!")
        st.write(f"**⏱️ 훈련 소요 시간:** {end_time - start_time:.2f}초")

        st.subheader("4. 최종 성능 평가")
        t1, t2, t3 = st.tabs(["훈련 데이터", "검증 데이터", "테스트 데이터 (최종)"])
        with t1:
            utils.display_evaluation_report(model, X_train, y_train, folder_names, "훈련 데이터")
        with t2:
            utils.display_evaluation_report(model, X_val, y_val, folder_names, "검증 데이터")
        with t3:
            if len(X_test) > 0:
                utils.display_evaluation_report(model, X_test, y_test, folder_names, "테스트 데이터")

                st.markdown("**📊 테스트 결과 요약 (Bar Chart)**")
                test_results = model.evaluate(X_test, y_test, verbose=0)
                metrics = ["Loss", "Accuracy"]
                values = test_results[:2]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(metrics, values, color=["#1f77b4", "#ff7f0e"])
                ax.set_ylim([0, 1.0])
                ax.set_ylabel("Value")
                ax.set_title("Test Results Summary")
                for i, v in enumerate(values):
                    ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10)
                st.pyplot(fig)

            else:
                st.warning("테스트 데이터가 없습니다.")


def step_6():
    render_header(contents.STEP_6)
    st.markdown("### 🗣️ 일반화 튜토리얼")
    st.info(guides.STEP_6_TUTORIAL)