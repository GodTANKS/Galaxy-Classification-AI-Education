import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="은하 분류 AI 웹 실습",
    page_icon="🌌",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "remote_manifest.json"
REMOTE_BASE = "https://raw.githubusercontent.com/GodTANKS/AI_galaxy/main"

CLASSES = {
    "0": "타원 은하 (Elliptical)",
    "1": "렌즈 은하 (Lens)",
    "2": "정상 나선 은하 (Spiral)",
    "3": "막대 나선 은하 (Barred Spiral)",
    "4": "불규칙 은하 (Irregular)",
}
IMAGE_SIZE = 50

st.markdown("""
<style>
.block-container{max-width:1200px;padding-top:2rem}
.big-title{font-size:2.5rem;font-weight:800;margin-bottom:.2rem}
.sub{color:#6b7280;margin-bottom:1.5rem}
.stepbox{padding:1rem;border:1px solid #dbe4f0;border-radius:14px;margin:.5rem 0}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)["classes"]


@st.cache_data(show_spinner=False)
def fetch_image(class_key, filename):
    url = f"{REMOTE_BASE}/{class_key}_{filename}"
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content


def open_image(class_key, filename):
    return Image.open(io.BytesIO(fetch_image(class_key, filename))).convert("RGB")


def parse_range(text, max_n):
    result = []
    for token in text.split(","):
        token = token.strip().replace("이미지", "")
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(token))
    return sorted({i for i in result if 1 <= i <= max_n})


def preprocess_one(img, color, augmentations):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    img = img.convert("RGB" if color else "L")
    out = [np.array(img)]
    for aug in augmentations:
        if aug == "좌우 대칭":
            out.append(np.array(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
        elif aug == "90도 회전":
            out.append(np.array(img.rotate(90)))
        elif aug == "180도 회전":
            out.append(np.array(img.rotate(180)))
        elif aug == "270도 회전":
            out.append(np.array(img.rotate(270)))
    return out


manifest = load_manifest()

if "selected" not in st.session_state:
    st.session_state.selected = {k: [] for k in CLASSES}
if "processed" not in st.session_state:
    st.session_state.processed = None
if "result" not in st.session_state:
    st.session_state.result = None

st.markdown('<div class="big-title">🌌 딥러닝 기반 은하 분류 AI 웹 실습</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">실제 은하 이미지 수집 → 전처리·증강 → CNN 학습 → 혼동행렬 해석을 브라우저에서 직접 수행합니다.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🚀 실습 단계")
    step = st.radio(
        "단계 선택",
        ["1. 문제 정의", "2. 데이터 수집", "3. 데이터 처리", "4. CNN 학습", "5. 결과 해석"],
    )
    st.caption("전체 교육용 배포판의 핵심 흐름을 웹에서 쉽게 실행하도록 구성한 간편판입니다.")

if step == "1. 문제 정의":
    st.header("1단계 · 문제 정의")
    st.info(
        "은하의 형태는 타원, 렌즈, 정상 나선, 막대 나선, 불규칙 등으로 분류할 수 있습니다. "
        "사람의 눈으로 애매한 이미지를 반복해서 분류하는 대신, CNN이 이미지의 패턴을 학습하도록 해 봅니다."
    )
    st.markdown("""
    ### 탐구 질문
    - 은하 종류별로 어떤 시각적 특징이 다른가?
    - 데이터 수와 균형은 AI 분류 성능에 어떤 영향을 줄까?
    - 회전·대칭 증강은 성능에 도움이 될까?
    - 어떤 은하끼리 오분류가 많이 발생할까?

    ### 실습 흐름
    **문제 정의 → 이미지 수집 → 전처리·증강 → 학습/검증 분할 → CNN 학습 → 혼동행렬 해석**
    """)

elif step == "2. 데이터 수집":
    st.header("2단계 · 데이터 수집")
    st.caption("웹 간편판은 현재 공식 실습키트와 동일한 548장 이미지 세트를 기존 데이터 저장소에서 필요한 만큼만 불러옵니다.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⚡ 각 은하 유형에서 20장씩 자동 수집", type="primary", use_container_width=True):
            st.session_state.selected = {k: files[:20] for k, files in manifest.items()}
            st.session_state.processed = None
            st.session_state.result = None
            st.success("총 100장의 원본 이미지를 수집했습니다.")

    with c2:
        if st.button("🗑️ 수집 데이터 초기화", use_container_width=True):
            st.session_state.selected = {k: [] for k in CLASSES}
            st.session_state.processed = None
            st.session_state.result = None
            st.rerun()

    class_key = st.selectbox("은하 유형 선택", list(CLASSES), format_func=lambda k: CLASSES[k])
    files = manifest[class_key]
    st.write(f"전체 이미지: **{len(files)}장**")

    range_text = st.text_input("수집할 이미지 번호", value="1-20", help="예: 1-20, 25, 30-35")
    if st.button("선택 범위 수집"):
        idxs = parse_range(range_text, len(files))
        selected = st.session_state.selected[class_key]
        for i in idxs:
            name = files[i - 1]
            if name not in selected:
                selected.append(name)
        st.session_state.processed = None
        st.session_state.result = None
        st.success(f"{CLASSES[class_key]}: 현재 {len(selected)}장 수집됨")

    st.subheader("미리보기")
    preview_start = st.number_input("미리보기 시작 번호", 1, max(1, len(files)), 1, step=10)
    preview = files[int(preview_start)-1:int(preview_start)-1+10]
    cols = st.columns(5)
    for i, name in enumerate(preview):
        with cols[i % 5]:
            try:
                st.image(open_image(class_key, name), caption=f"{int(preview_start)+i}. {name}", use_container_width=True)
            except Exception as e:
                st.warning(f"로드 실패: {name}")

    st.divider()
    st.subheader("현재 수집 현황")
    for k, label in CLASSES.items():
        st.write(f"- {label}: **{len(st.session_state.selected[k])}장**")

elif step == "3. 데이터 처리":
    st.header("3단계 · 데이터 처리 및 증강")

    color = st.checkbox("RGB 컬러로 학습", value=False)
    augmentations = st.multiselect(
        "데이터 증강",
        ["좌우 대칭", "90도 회전", "180도 회전", "270도 회전"],
        default=["좌우 대칭"],
    )

    total_original = sum(len(v) for v in st.session_state.selected.values())
    st.write(f"수집된 원본: **{total_original}장**")

    if st.button("⚙️ 전처리 실행", type="primary"):
        if total_original == 0:
            st.error("먼저 2단계에서 이미지를 수집하세요.")
        else:
            X, y = [], []
            prog = st.progress(0)
            jobs = [(k, n) for k, names in st.session_state.selected.items() for n in names]
            for j, (k, name) in enumerate(jobs):
                img = open_image(k, name)
                variants = preprocess_one(img, color, augmentations)
                X.extend(variants)
                y.extend([int(k)] * len(variants))
                prog.progress((j + 1) / len(jobs))

            X = np.asarray(X, dtype=np.float32) / 255.0
            if not color:
                X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
            st.session_state.processed = {
                "X": X,
                "y": np.asarray(y, dtype=np.int64),
                "color": color,
                "augmentations": augmentations,
            }
            st.session_state.result = None
            st.success(f"완료: **{len(X)}개** 학습용 이미지 생성")

    if st.session_state.processed is not None:
        p = st.session_state.processed
        st.write(f"현재 처리 데이터: **{len(p['X'])}개**")
        st.write("이미지 크기:", p["X"].shape[1:])

elif step == "4. CNN 학습":
    st.header("4단계 · CNN 학습")
    if st.session_state.processed is None:
        st.warning("먼저 3단계에서 전처리를 실행하세요.")
        st.stop()

    p = st.session_state.processed
    X, y = p["X"], p["y"]

    counts = {i: int(np.sum(y == i)) for i in range(5)}
    st.write("클래스별 처리 데이터:", {CLASSES[str(k)]: v for k, v in counts.items()})

    if any(v < 3 for v in counts.values()):
        st.error("모든 은하 유형에서 최소 3개 이상의 처리 이미지가 필요합니다. 2단계에서 각 유형의 이미지를 더 수집하세요.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.slider("Epochs", 3, 20, 5)
    with c2:
        learning_rate = st.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001)
    with c3:
        batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)

    if st.button("🔥 CNN 학습 시작", type="primary"):
        import tensorflow as tf

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        input_shape = X_train.shape[1:]
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(5, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        with st.spinner("CNN을 학습하고 있습니다..."):
            hist = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        cm = confusion_matrix(y_test, pred, labels=[0,1,2,3,4])

        st.session_state.result = {
            "history": hist.history,
            "accuracy": float(acc),
            "loss": float(loss),
            "cm": cm,
            "y_test": y_test,
            "pred": pred,
        }
        st.success(f"학습 완료 · 테스트 정확도 **{acc:.3f}**")

    if st.session_state.result is not None:
        r = st.session_state.result
        st.metric("테스트 정확도", f"{r['accuracy']:.3f}")
        st.metric("테스트 손실값", f"{r['loss']:.3f}")

elif step == "5. 결과 해석":
    st.header("5단계 · 결과 해석")
    r = st.session_state.result
    if r is None:
        st.warning("먼저 4단계에서 CNN을 학습하세요.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("학습 곡선")
        fig, ax = plt.subplots()
        ax.plot(r["history"]["accuracy"], label="train accuracy")
        ax.plot(r["history"]["val_accuracy"], label="validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

    with c2:
        st.subheader("혼동행렬")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(r["cm"], cmap="Blues")
        labels = [CLASSES[str(i)].split(" ")[0] for i in range(5)]
        ax.set_xticks(range(5), labels, rotation=45, ha="right")
        ax.set_yticks(range(5), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(r["cm"][i, j]), ha="center", va="center")
        st.pyplot(fig)

    st.success(f"테스트 정확도: {r['accuracy']:.3f}")
    st.markdown("""
    ### 생각해 보기
    - 어떤 두 은하 유형 사이에서 오분류가 가장 많았나요?
    - 데이터 증강을 바꾸면 결과가 어떻게 달라지나요?
    - 각 유형의 데이터 수를 늘리면 성능은 어떻게 변하나요?
    - 높은 정확도가 항상 좋은 과학적 모델을 의미할까요?
    """)

st.divider()
st.caption(
    "연구·교육용 웹 간편판 · 전체 교육용 배포판과 논문은 "
    "https://GodTANKS.github.io/astronomy-data-science/ 에서 확인할 수 있습니다."
)
