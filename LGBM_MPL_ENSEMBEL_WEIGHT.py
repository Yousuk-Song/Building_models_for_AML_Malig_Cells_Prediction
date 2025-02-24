import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score, recall_score, precision_score
from google.colab import drive
import joblib  # LightGBM 모델 저장을 위한 라이브러리

# ✅ Google Drive 마운트
drive.mount('/content/drive/')

# ✅ 데이터 경로 설정
BASE_PATH = "/content/drive/MyDrive/hepscope/train_val_sample_test/deg.all.fc0.5/"
RESULTS_PATH = "/content/drive/MyDrive/hepscope/results/LGBM_MPL_ENSEMBEL_WEIGHT/"
MERGED_RESULTS_PATH = os.path.join(RESULTS_PATH, "merged_lbgm_mlp_ensemble_weight_results_external.csv")
os.makedirs(RESULTS_PATH, exist_ok=True)  # 결과 폴더 생성

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

# ✅ 개별 train_case 학습 및 평가 함수
def process_train_case(train_case):
    case_path = os.path.join(BASE_PATH, train_case)
    result_case_path = os.path.join(RESULTS_PATH, train_case)  # train_case별 결과 폴더 생성
    os.makedirs(result_case_path, exist_ok=True)

    print(f"\n🔹 Processing {train_case} ...")

    # 🔹 데이터 로드
    train_normal = load_csv(os.path.join(case_path, 'train_Normal.csv'))
    train_malig = load_csv(os.path.join(case_path, 'train_Malig.csv'))

    if train_normal is None or train_malig is None:
        print(f"❌ {train_case} 데이터 부족으로 학습 스킵")
        return None

    # 🔹 데이터 전처리
    train_x, train_y, scaler = prepare_data(train_normal, train_malig)

    results = []

    # ✅ External Validation을 모든 val_ 디렉토리에 대해 수행
    val_dirs = [d for d in os.listdir(case_path) if d.startswith("val_")]
    for val_dir in val_dirs:
        val_path = os.path.join(case_path, val_dir)
        val_normal = load_csv(os.path.join(val_path, 'val_external_Normal.csv'))
        val_malig = load_csv(os.path.join(val_path, 'val_external_Malig.csv'))

        if val_normal is None or val_malig is None:
            print(f"❌ {train_case} - {val_dir} 데이터 부족으로 스킵")
            continue

        val_x, val_y, _ = prepare_data(val_normal, val_malig, scaler)

        # ✅ LGBM 학습
        lgbm_model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.1,
            random_state=42
        )
        lgbm_model.fit(train_x, train_y)

        # ✅ MLP 학습
        mlp_model = Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(train_x.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        mlp_model.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

        # ✅ 모델 저장
        lgbm_model_path = os.path.join(result_case_path, f"{train_case}_LGBM.txt")
        mlp_model_path = os.path.join(result_case_path, f"{train_case}_MLP.h5")
        joblib.dump(lgbm_model, lgbm_model_path)  # LightGBM 모델 저장
        mlp_model.save(mlp_model_path)  # MLP 모델 저장
        print(f"✅ 모델 저장 완료: {lgbm_model_path}, {mlp_model_path}")

        # ✅ External Validation 평가
        val_lgbm_preds = lgbm_model.predict_proba(val_x)[:, 1]
        val_mlp_preds = mlp_model.predict(val_x).flatten()

        # ✅ Train & Validation Loss 계산
        train_lgbm_loss = log_loss(train_y, lgbm_model.predict_proba(train_x)[:, 1])
        train_mlp_loss = log_loss(train_y, mlp_model.predict(train_x).flatten())
        val_lgbm_loss = log_loss(val_y, val_lgbm_preds)
        val_mlp_loss = log_loss(val_y, val_mlp_preds)

        # ✅ Log Loss 기반 가중치 계산
        epsilon = 1e-6
        weight_lgbm = 1 / (val_lgbm_loss + epsilon)
        weight_mlp = 1 / (val_mlp_loss + epsilon)

        # ✅ 가중 평균 예측
        weight_sum = weight_lgbm + weight_mlp
        weight_lgbm /= weight_sum
        weight_mlp /= weight_sum

        ensemble_preds = (weight_lgbm * val_lgbm_preds + weight_mlp * val_mlp_preds)
        ensemble_pred_binary = (ensemble_preds > 0.5).astype(int)

        f1 = f1_score(val_y, ensemble_pred_binary)
        recall = recall_score(val_y, ensemble_pred_binary)
        precision = precision_score(val_y, ensemble_pred_binary)
        accuracy = accuracy_score(val_y, ensemble_pred_binary)

        # ✅ Generalization Gap & Loss Ratio
        generalization_gap_lgbm = accuracy_score(train_y, (lgbm_model.predict_proba(train_x)[:, 1] > 0.5).astype(int)) - accuracy
        generalization_gap_mlp = accuracy_score(train_y, (mlp_model.predict(train_x).flatten() > 0.5).astype(int)) - accuracy
        loss_ratio_lgbm = val_lgbm_loss / (train_lgbm_loss + epsilon)
        loss_ratio_mlp = val_mlp_loss / (train_mlp_loss + epsilon)

        results.append([train_case, val_dir, f1, recall, precision, accuracy, weight_lgbm, weight_mlp, generalization_gap_lgbm, generalization_gap_mlp, loss_ratio_lgbm, loss_ratio_mlp])
        
    # ✅ train_case별 External Validation 결과 저장
    results_df = pd.DataFrame(results, columns=[
        "Train Case", "Dataset", "F1 Score", "Recall", "Precision", "Accuracy",
        "LGBM Weight", "MLP Weight",
        "LGBM Generalization Gap", "MLP Generalization Gap",
        "LGBM Loss Ratio", "MLP Loss Ratio"
    ])
    results_df.to_csv(os.path.join(result_case_path, f"{train_case}_LGBM_MLP_Ensemble_WEIGHT_Results_external.csv"), index=False)
    return results_df

# ✅ 실행 및 결과 병합
all_results = [process_train_case(case) for case in sorted(os.listdir(BASE_PATH)) if case.startswith("train_case_")]
if all_results:
    pd.concat(all_results).to_csv(MERGED_RESULTS_PATH, index=False)
    print(f"\n✅ 결과 병합 완료: {MERGED_RESULTS_PATH}")
