### LightGBM 방식

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score, recall_score, precision_score

# ✅ 데이터 경로 설정
BASE_PATH = "/content/drive/MyDrive/hepscope/train_val_sample_test/deg.all.fc0.5/"
RESULTS_PATH = "/content/drive/MyDrive/hepscope/results/LGMB/"
MERGED_RESULTS_PATH = os.path.join(RESULTS_PATH, "merged_lbgm_results.csv")
os.makedirs(RESULTS_PATH, exist_ok=True)

# ✅ 파일 로드 함수
def load_csv(file_path):
    try:
        return pd.read_csv(file_path).drop(columns=['Unnamed: 0'], errors='ignore')
    except Exception as e:
        print(f"❌ 파일 로드 실패 ({file_path}): {e}")
        return None

# ✅ 데이터 중복 확인 함수
def check_data_leakage(train_df, val_df, ext_val_df):
    train_index = set(train_df.index)
    val_index = set(val_df.index) if val_df is not None else set()
    ext_index = set(ext_val_df.index) if ext_val_df is not None else set()

    if train_index & val_index or train_index & ext_index or val_index & ext_index:
        print("❌ 데이터가 완전히 분리되지 않았습니다! 실행을 중단합니다.")
        exit(1)

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
    print(f"\n🔹 Processing {train_case} ...")

    # 🔹 데이터 로드 (Train)
    train_normal = load_csv(os.path.join(case_path, 'train_Normal.csv'))
    train_malig = load_csv(os.path.join(case_path, 'train_Malig.csv'))
    if train_normal is None or train_malig is None:
        return []

    # 🔹 데이터 전처리 (Train)
    train_x, train_y, scaler = prepare_data(train_normal, train_malig)

    # ✅ LightGBM 모델 생성 및 학습
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        random_state=42
    )
    model.fit(train_x, train_y)

    train_pred = (model.predict_proba(train_x)[:, 1] > 0.5).astype(int)
    train_acc = accuracy_score(train_y, train_pred)
    train_loss = log_loss(train_y, model.predict_proba(train_x)[:, 1])

    # ✅ 모델 저장
    model_save_dir = os.path.join(RESULTS_PATH, train_case)
    os.makedirs(model_save_dir, exist_ok=True)
    save_model_path = os.path.join(model_save_dir, f"{train_case}_model.txt")
    model.booster_.save_model(save_model_path)
    print(f"✅ 모델 저장 완료: {save_model_path}")

    results = []

    # 🔹 Validation (Internal)
    val_normal = load_csv(os.path.join(case_path, 'val_internal_Normal.csv'))
    val_malig = load_csv(os.path.join(case_path, 'val_internal_Malig.csv'))
    val_x, val_y, _ = prepare_data(val_normal, val_malig, scaler) if val_normal is not None and val_malig is not None else (None, None, None)

    if val_x is not None:
        val_pred = (model.predict_proba(val_x)[:, 1] > 0.5).astype(int)
        val_acc = accuracy_score(val_y, val_pred)
        val_loss = log_loss(val_y, model.predict_proba(val_x)[:, 1])
    else:
        val_acc, val_loss = None, None

    # 🔹 Generalization Gap 및 Loss Ratio 계산
    gen_gap = abs(train_acc - val_acc) if val_acc is not None else None
    loss_ratio = train_loss / val_loss if val_loss is not None and val_loss != 0 else None

    # ✅ 데이터 중복 검증 수행
    check_data_leakage(train_normal, val_normal, None)

    # 🔹 External Validation (External 검증)
    external_path = os.path.join(case_path, "external_validation")
    ext_normal_path = os.path.join(external_path, "val_external_Normal.csv")
    ext_malig_path = os.path.join(external_path, "val_external_Malig.csv")

    if os.path.exists(ext_normal_path) and os.path.exists(ext_malig_path):
        ext_normal = load_csv(ext_normal_path)
        ext_malig = load_csv(ext_malig_path)
        ext_x, ext_y, _ = prepare_data(ext_normal, ext_malig, scaler)

        pred = (model.predict_proba(ext_x)[:, 1] > 0.5).astype(int)

        ext_f1 = f1_score(ext_y, pred)
        ext_recall = recall_score(ext_y, pred)
        ext_precision = precision_score(ext_y, pred)
        ext_accuracy = accuracy_score(ext_y, pred)

        print(f"🔹 External Validation 결과: F1: {ext_f1:.4f}, Recall: {ext_recall:.4f}, Precision: {ext_precision:.4f}, Accuracy: {ext_accuracy:.4f}")
    else:
        ext_f1, ext_recall, ext_precision, ext_accuracy = None, None, None, None

    # ✅ 결과 저장
    results.append([
        train_case, f1_score(train_y, train_pred), recall_score(train_y, train_pred),
        precision_score(train_y, train_pred), train_acc, val_acc, gen_gap, loss_ratio,
        ext_f1, ext_recall, ext_precision, ext_accuracy
    ])

    return results

# ✅ 학습 실행 및 결과 병합
all_results = []
train_cases = sorted([d for d in os.listdir(BASE_PATH) if d.startswith("train_case_")])

for case in train_cases:
    result_df = process_train_case(case)
    if result_df:
        all_results.extend(result_df)

if all_results:
    merged_df = pd.DataFrame(all_results, columns=["Train Case", "F1 Score", "Recall", "Precision", "Train Acc", "Val Acc", "Gen Gap", "Loss Ratio", "Ext F1", "Ext Recall", "Ext Precision", "Ext Acc"])
    merged_df.to_csv(MERGED_RESULTS_PATH, index=False)
    print(f"\n✅ 모든 학습 결과 병합 완료: {MERGED_RESULTS_PATH}")
else:
    print("\n❌ 병합할 학습 결과 없음")
