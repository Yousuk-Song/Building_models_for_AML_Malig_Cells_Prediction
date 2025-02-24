import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score, recall_score, precision_score
from google.colab import drive

# ✅ Google Drive 마운트
drive.mount('/content/drive/')

# ✅ 데이터 경로 설정
BASE_PATH = "/content/drive/MyDrive/hepscope/train_val_sample_test/deg.all.fc0.5/"
RESULTS_PATH = "/content/drive/MyDrive/hepscope/results/MLP_ONLY/"
MERGED_RESULTS_PATH = os.path.join(RESULTS_PATH, "merged_mlp_results.csv")
os.makedirs(RESULTS_PATH, exist_ok=True)

# ✅ 파일 로드 함수
def load_csv(file_path):
    try:
        return pd.read_csv(file_path).drop(columns=['Unnamed: 0'], errors='ignore')
    except Exception as e:
        print(f"❌ 파일 로드 실패 ({file_path}): {e}")
        return None

# ✅ 데이터 정규화 및 결합 함수
def prepare_data(normal_df, malig_df, scaler=None):
    if normal_df is None or malig_df is None:
        return None, None, None
    normal_df['label'] = 0
    malig_df['label'] = 1
    data = pd.concat([normal_df, malig_df]).reset_index(drop=True)
    y = data.pop('label').values
    if scaler:
        x = scaler.transform(data)
    else:
        scaler = RobustScaler()
        x = scaler.fit_transform(data)
    return x, y, scaler

# ✅ 개별 train_case 학습 및 평가 함수 (MLP만 사용)
def process_train_case(train_case):
    case_path = os.path.join(BASE_PATH, train_case)
    print(f"\n🔹 Processing {train_case} ...")

    # 🔹 데이터 로드
    train_normal = load_csv(os.path.join(case_path, 'train_Normal.csv'))
    train_malig = load_csv(os.path.join(case_path, 'train_Malig.csv'))
    val_normal = load_csv(os.path.join(case_path, 'val_internal_Normal.csv'))
    val_malig = load_csv(os.path.join(case_path, 'val_internal_Malig.csv'))

    if train_normal is None or train_malig is None or val_normal is None or val_malig is None:
        print(f"❌ {train_case} 데이터 부족으로 학습 스킵")
        return None

    # 🔹 데이터 전처리
    train_x, train_y, scaler = prepare_data(train_normal, train_malig)
    val_x, val_y, _ = prepare_data(val_normal, val_malig, scaler)

    # ✅ 결과 저장 폴더 생성
    model_save_dir = os.path.join(RESULTS_PATH, train_case)
    os.makedirs(model_save_dir, exist_ok=True)

    # ✅ MLP 모델 학습
    mlp_model = Sequential([
        Dense(256, activation='relu', input_shape=(train_x.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    mlp_model.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # ✅ 모델 저장
    mlp_model.save(os.path.join(model_save_dir, f"{train_case}_mlp.h5"))
    print(f"✅ 모델 저장 완료: {model_save_dir}")

    results = []

    # ✅ Internal Validation 평가
    val_mlp_preds = mlp_model.predict(val_x).flatten()
    val_pred_binary = (val_mlp_preds > 0.5).astype(int)

    f1 = f1_score(val_y, val_pred_binary)
    recall = recall_score(val_y, val_pred_binary)
    precision = precision_score(val_y, val_pred_binary)
    accuracy = accuracy_score(val_y, val_pred_binary)
    train_acc = accuracy_score(train_y, (mlp_model.predict(train_x).flatten() > 0.5).astype(int))
    val_acc = accuracy
    generalization_gap = train_acc - val_acc
    loss_ratio = log_loss(val_y, val_mlp_preds) / log_loss(train_y, mlp_model.predict(train_x).flatten())

    results.append([train_case, "internal", f1, recall, precision, accuracy, train_acc, val_acc, generalization_gap, loss_ratio])

    # 🔹 External Validation 평가 (val_로 시작하는 폴더 자동 탐색)
    for subfolder in os.listdir(case_path):
        subfolder_path = os.path.join(case_path, subfolder)

        if os.path.isdir(subfolder_path) and subfolder.startswith("val_"):
            ext_normal = load_csv(os.path.join(subfolder_path, "val_external_Normal.csv"))
            ext_malig = load_csv(os.path.join(subfolder_path, "val_external_Malig.csv"))

            if ext_normal is not None and ext_malig is not None:
                ext_x, ext_y, _ = prepare_data(ext_normal, ext_malig, scaler)

                ext_mlp_preds = mlp_model.predict(ext_x).flatten()
                ext_pred_binary = (ext_mlp_preds > 0.5).astype(int)

                results.append([
                    train_case, subfolder, f1_score(ext_y, ext_pred_binary),
                    recall_score(ext_y, ext_pred_binary), precision_score(ext_y, ext_pred_binary),
                    accuracy_score(ext_y, ext_pred_binary), train_acc, val_acc, generalization_gap, loss_ratio
                ])

    results_df = pd.DataFrame(results, columns=[
        "Train Case", "Dataset", "F1 Score", "Recall", "Precision", "Accuracy",
        "Train Acc", "Val Acc", "Generalization Gap", "Loss Ratio"
    ])
    results_df.to_csv(os.path.join(model_save_dir, "MLP_Results.csv"), index=False)
    return results_df

# ✅ 학습 실행 및 결과 병합
all_results = []
train_cases = sorted([d for d in os.listdir(BASE_PATH) if d.startswith("train_case_")])

for case in train_cases:
    result_df = process_train_case(case)
    if result_df is not None:
        all_results.append(result_df)

if all_results:
    merged_df = pd.concat(all_results, ignore_index=True)
    merged_df.to_csv(MERGED_RESULTS_PATH, index=False)
    print(f"\n✅ 모든 학습 결과 병합 완료: {MERGED_RESULTS_PATH}")
else:
    print("\n❌ 병합할 학습 결과 없음")
