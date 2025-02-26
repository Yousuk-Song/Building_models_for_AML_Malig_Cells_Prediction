
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def visualize_and_save_confusion_matrix(y_true, y_pred, prediction_file):
    """
    Seaborn 없이 Matplotlib으로 혼동 행렬을 시각화하여 저장.
    """
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 저장할 파일 이름 변환
    output_file = prediction_file.replace(".csv", ".confusion_matrix.png")

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    # x, y 축 설정
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    # 축 레이블 설정
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)

    # 각 셀에 텍스트 추가
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="black")

    # 색상바 추가
    fig.colorbar(im)

    # 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_file}")


def print_confusion_matrix(y_true, y_pred):
    """
    혼동 행렬을 텍스트로 출력합니다.
    """
    # 🚨 labels=[0,1] 추가하여 항상 0과 1을 포함하도록 설정
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 혼동 행렬 결과 텍스트 출력
    print("\nConfusion Matrix (Text Output):")
    print(f"True Negative  (TN): {cm[0, 0]}")
    print(f"False Positive (FP): {cm[0, 1]}")
    print(f"False Negative (FN): {cm[1, 0]}")
    print(f"True Positive  (TP): {cm[1, 1]}\n")

    print("Full Confusion Matrix:")

#def visualize_and_save_confusion_matrix(y_true, y_pred, prediction_file):
#    """
#    혼동 행렬을 시각화하여 저장합니다.
#    모든 숫자가 검은색(black)으로 표시되도록 설정합니다.
#    """
#    # 혼동 행렬 계산
#    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

#    # 저장할 파일 이름 변환
#    output_file = prediction_file.replace(".csv", ".confusion_matrix.png")

    # 시각화
#    plt.figure(figsize=(6, 5))
#    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'], annot_kws={"color": "black"})

    # 🚨 모든 숫자를 강제로 검정(black)으로 설정
#    for text in ax.texts:
#        text.set_color("black")

#    plt.xlabel("Predicted Label")
#    plt.ylabel("True Label")
#    plt.title("Confusion Matrix")

    # 저장
#    plt.savefig(output_file, dpi=300, bbox_inches='tight')
#    plt.close()

 #   print(f"Confusion matrix saved to {output_file}")

def evaluate_and_visualize(ground_truth_file, prediction_file):
    """
    Ground truth 파일과 prediction 파일을 비교하여 성능을 평가하고,
    혼동 행렬을 텍스트로 출력한 후 시각화하여 저장합니다.
    """
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', header=None, names=['cell_id', 'label'], dtype={'cell_id': str, 'label': int})

    # Load predictions
    predictions = pd.read_csv(prediction_file, delim_whitespace=True)

    # Merge on cell_id
    merged = ground_truth.merge(predictions, on='cell_id', how='inner')

    # 실제 라벨과 예측값
    y_true = merged['label']
    y_pred = merged['Ensemble_Prediction']

    # 🚨 `y_true`와 `y_pred`에 0과 1이 모두 있는지 확인
    print("Unique values in y_true:", set(y_true))
    print("Unique values in y_pred:", set(y_pred))

    # 🔥 혼동 행렬 텍스트 출력
    print_confusion_matrix(y_true, y_pred)

    # 🔥 혼동 행렬 시각화 및 저장 (숫자는 검정색으로 표시)
    visualize_and_save_confusion_matrix(y_true, y_pred, prediction_file)

# 실행 예시
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <prediction_file>")
        sys.exit(1)

    ground_truth_file = "/data/workbench/scRSEQ_AML/exdata/mlp_prediction/YS_predict/aml.test.validation.label.TP.tsv"
    prediction_file = sys.argv[1]

    evaluate_and_visualize(ground_truth_file, prediction_file)
