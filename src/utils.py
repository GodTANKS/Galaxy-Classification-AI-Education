import os
import platform
import io
import json
import requests
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

IMAGE_SIZE = 50

# ==============================================================================
# [중요] Matplotlib 한글 깨짐 방지 글로벌 설정 (utils.py 내부 그래프용)
# ==============================================================================
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 데이터 수집 및 전처리 엔진
# ==============================================================================

REMOTE_IMAGE_BASE = "https://raw.githubusercontent.com/GodTANKS/AI_galaxy/main"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_MANIFEST_PATH = PROJECT_ROOT / "data" / "remote_manifest.json"

@st.cache_data(show_spinner=False)
def _load_remote_manifest():
    try:
        with open(REMOTE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["classes"]
    except Exception:
        return {str(i): [] for i in range(5)}

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_remote_image_bytes(folder_name, image_name):
    url = f"{REMOTE_IMAGE_BASE}/{folder_name}_{image_name}"
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return response.content

def get_local_image_files(base_folder, folder_name):
    """로컬 완성판은 galaxy_rne 폴더, 웹판은 동일 548장 manifest를 사용"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, base_folder, str(folder_name))

    if os.path.exists(folder_path):
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            files.sort()
            return files
        except Exception:
            return []

    return list(_load_remote_manifest().get(str(folder_name), []))

def open_source_image(base_folder, folder_name, image_name):
    """로컬 완성판과 Streamlit 웹판에서 동일한 원본 이미지를 여는 함수"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, base_folder, str(folder_name), image_name)

    if os.path.exists(image_path):
        with Image.open(image_path) as source_img:
            return source_img.copy()

    raw = _fetch_remote_image_bytes(str(folder_name), image_name)
    with Image.open(io.BytesIO(raw)) as source_img:
        return source_img.copy()


def process_images(
    folder_key,
    base_folder,
    image_names,
    resize,
    transformations,
    is_color=True
):
    """원본 이미지 전처리 및 선택한 증강 적용"""
    processed_images = []
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for image_name in image_names:
        image_path = os.path.join(
            current_dir,
            base_folder,
            str(folder_key),
            image_name
        )

        try:
            # 로컬 완성판은 galaxy_rne에서, 웹판은 동일 548장 원격 데이터에서 불러옴
            img = open_source_image(base_folder, folder_key, image_name)

            # 모든 이미지의 색상 채널 통일
            img = img.convert("RGB") if is_color else img.convert("L")

            # 이미지 크기 표준화
            if resize:
                img = img.resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.Resampling.LANCZOS
                )

            # 중요: 증강 여부와 관계없이 원본 이미지를 반드시 추가
            processed_images.append(np.array(img))

            # 선택된 증강 이미지 추가
            for transformation in transformations:

                if transformation == "45도 회전":
                    new_img = img.rotate(
                        45,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "90도 회전":
                    new_img = img.rotate(
                        90,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "135도 회전":
                    new_img = img.rotate(
                        135,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "180도 회전":
                    new_img = img.rotate(
                        180,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "225도 회전":
                    new_img = img.rotate(
                        225,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "270도 회전":
                    new_img = img.rotate(
                        270,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "315도 회전":
                    new_img = img.rotate(
                        315,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                elif transformation == "좌우 대칭":
                    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                elif transformation.startswith("좌우 대칭 및"):
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

                    angle_map = {
                        "좌우 대칭 및 45도 회전": 45,
                        "좌우 대칭 및 90도 회전": 90,
                        "좌우 대칭 및 135도 회전": 135,
                        "좌우 대칭 및 180도 회전": 180,
                        "좌우 대칭 및 225도 회전": 225,
                        "좌우 대칭 및 270도 회전": 270,
                        "좌우 대칭 및 315도 회전": 315,
                    }

                    angle = angle_map.get(transformation)

                    if angle is None:
                        continue

                    new_img = flipped.rotate(
                        angle,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=(0, 0, 0) if is_color else 0
                    )

                else:
                    continue

                processed_images.append(np.array(new_img))

        except Exception as e:
            print(f"이미지 처리 실패: {image_path} / {e}")
            continue

    return processed_images

def augment_train_data(X_train, y_train, processing_log):
    """[데이터 누수 방지용] 분할된 훈련 데이터(Train Set)에만 증강을 적용하는 함수"""
    aug_X = []
    aug_y = []

    for img_array, label_array in zip(X_train, y_train):
        # 1. 원본 데이터는 무조건 훈련셋에 포함
        aug_X.append(img_array)
        aug_y.append(label_array)

        # 2. 정답 라벨(one-hot)에서 폴더 키(클래스 인덱스)를 추출하여 사용자가 설정한 증강 옵션 찾기
        folder_key = str(np.argmax(label_array))
        if folder_key not in processing_log:
            continue

        transformations = processing_log[folder_key].get('augmentations', [])
        if not transformations:
            continue

        # 3. 이미지 회전/대칭을 위해 float32(0~1) 배열을 잠시 PIL 이미지(0~255)로 복원
        img_uint8 = np.uint8(img_array * 255.0)
        if img_uint8.shape[-1] == 1:
            img = Image.fromarray(img_uint8.squeeze(), mode='L')
        else:
            img = Image.fromarray(img_uint8, mode='RGB')

        # 4. 사용자가 선택했던 증강 기법을 훈련 데이터에만 개별 적용
        for t in transformations:
            try:
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT) if "좌우 대칭" in t else None
                base_img = flipped if "좌우 대칭" in t else img

                if t == "좌우 대칭":
                    new_img = flipped
                elif "45도 회전" in t:
                    new_img = base_img.rotate(45)
                elif "90도 회전" in t:
                    new_img = base_img.rotate(90)
                elif "135도 회전" in t:
                    new_img = base_img.rotate(135)
                elif "180도 회전" in t:
                    new_img = base_img.rotate(180)
                elif "225도 회전" in t:
                    new_img = base_img.rotate(225)
                elif "270도 회전" in t:
                    new_img = base_img.rotate(270)
                elif "315도 회전" in t:
                    new_img = base_img.rotate(315)
                else:
                    continue

                # 5. 변환된 이미지를 다시 모델 학습용 float32 배열로 정규화
                new_array = np.array(new_img, dtype=np.float32) / 255.0
                if len(new_array.shape) == 2:  # 흑백 이미지인 경우 채널(1) 복구
                    new_array = np.expand_dims(new_array, axis=-1)

                aug_X.append(new_array)
                aug_y.append(label_array)
            except:
                pass

    return np.array(aug_X), np.array(aug_y)


# ==============================================================================
# 2. 딥러닝 모델(CNN) 구축 엔진
# ==============================================================================

def build_cnn_model(is_color, conv_layers, filters, pool_sizes, activations,
                    dense_layers, dense_units, dense_acts,
                    dropout_layers, dropout_rate, last_activation):
    """사용자가 입력한 하이퍼파라미터를 바탕으로 CNN 모델을 동적으로 생성하는 함수"""
    model = tf.keras.Sequential()
    input_channels = 3 if is_color else 1
    model.add(tf.keras.layers.InputLayer(input_shape=(50, 50, input_channels)))

    for i in range(conv_layers):
        model.add(tf.keras.layers.Conv2D(
            filters[i], (3, 3),
            padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(0.03)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activations[i]))
        model.add(tf.keras.layers.MaxPooling2D(pool_sizes[i], padding='same'))

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    for i in range(dense_layers):
        model.add(tf.keras.layers.Dense(
            dense_units[i],
            use_bias=False,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(0.03)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(dense_acts[i]))

        if i < dropout_layers:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        5,
        kernel_regularizer=tf.keras.regularizers.l2(0.03),
        activation=last_activation
    ))

    return model

def get_layer_details_korean(model):
    """생성된 모델의 구조와 파라미터 수를 한글 표로 정리하여 반환"""
    data = []
    for layer in model.layers:
        try:
            output_shape = str(layer.output.shape)
            params = layer.count_params()
        except:
            output_shape = "N/A"
            params = 0
        data.append([layer.name, layer.__class__.__name__, output_shape, f"{params:,}"])

    df = pd.DataFrame(data, columns=["레이어 이름", "종류", "출력 형태", "파라미터 수"])

    total = model.count_params()
    trainable = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable = total - trainable

    info_text = f"""
    - **총 파라미터:** {total:,}개
    - **학습 가능:** {trainable:,}개
    - **학습 불가능:** {non_trainable:,}개
    """
    return df, info_text


def get_callbacks(patience, reduce_patience, reduce_factor, min_lr):
    """[복구 완료] 조기 종료, 최고 모델 저장, 엑셀 로그 기록 등 원본 콜백 완전 복구"""
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True,
                                                  verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=reduce_patience,
                                                     min_lr=min_lr, verbose=1)

    # 모델 저장 및 CSV 기록 기능 복구
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5", monitor="val_accuracy",
                                                          save_best_only=True, verbose=0)
    csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv", append=True)

    return [early_stop, model_checkpoint, reduce_lr, csv_logger]


# ==============================================================================
# 3. 결과 평가 및 시각화 도구
# ==============================================================================

def print_classification_report_as_table(y_true, y_pred, folder_names):
    """분류 성능 상세 보고서를 표 형태로 출력"""
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [folder_names.get(str(l), f"Class {l}") for l in unique_labels]

    report = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, output_dict=True,
                                   zero_division=0)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df.style.format("{:.2f}"))


def plot_confusion_matrix(y_true, y_pred, folder_names):
    """혼동 행렬(Confusion Matrix) 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))
    tick_labels = [folder_names.get(str(l), str(l)) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.ylabel('Actual (실제 은하)')
    plt.xlabel('Predicted (AI 예측)')
    st.pyplot(fig)


def visualize_misclassified_images(X, y_true, y_pred, title, folder_names, num_images=30):
    """[복구 완료] 앱 멈춤 방지를 위해 오분류 이미지를 최대 num_images(30장)만 랜덤으로 뽑아서 출력"""
    mis_indices = np.where(y_true != y_pred)[0]

    if len(mis_indices) == 0:
        st.success(f"🎉 {title}: 모든 이미지를 정확하게 분류했습니다!")
        return

    # 브라우저 과부하 방지를 위한 랜덤 샘플링 복구
    display_count = min(num_images, len(mis_indices))
    st.markdown(f"#### 🔍 {title}: 오분류 사례 분석 (총 {len(mis_indices)}개 중 {display_count}개 무작위 표출)")

    selected_indices = np.random.choice(mis_indices, display_count, replace=False)

    cols = st.columns(5)
    for i, idx in enumerate(selected_indices):
        true_label = folder_names.get(str(y_true[idx]), str(y_true[idx]))
        pred_label = folder_names.get(str(y_pred[idx]), str(y_pred[idx]))

        with cols[i % 5]:
            img_data = X[idx].squeeze()
            caption_text = f"[{i + 1}]\n정답:{true_label}\n예측:{pred_label}"

            if img_data.ndim == 2:
                st.image(img_data, clamp=True, caption=caption_text, output_format='PNG')
            else:
                st.image(img_data, clamp=True, caption=caption_text)


def display_evaluation_report(model, X, y, folder_names, title):
    """훈련, 검증, 테스트 데이터의 최종 평가 리포트를 화면에 종합하여 출력"""
    if len(X) == 0:
        st.warning("데이터가 없습니다.")
        return

    y_pred_prob = model.predict(X, verbose=0)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y.argmax(axis=1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. 혼동 행렬 (Confusion Matrix)**")
        plot_confusion_matrix(y_true, y_pred, folder_names)
    with c2:
        st.markdown("**2. 성능 지표 요약**")
        res = model.evaluate(X, y, verbose=0)
        st.metric("정확도 (Accuracy)", f"{res[1]:.2%}")
        st.metric("손실값 (Loss)", f"{res[0]:.4f}")

    st.markdown("**3. 상세 분류 보고서**")
    print_classification_report_as_table(y_true, y_pred, folder_names)

    st.divider()

    visualize_misclassified_images(X, y_true, y_pred, f"{title} 오분류", folder_names)