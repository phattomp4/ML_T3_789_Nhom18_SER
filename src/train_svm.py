import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

CSV_PATH = "data/ravdess_features_1d.csv"

if not os.path.exists(CSV_PATH):
    print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy chạy file build_dataset_1d.py trước!")
    exit()

print("Đang tải dữ liệu từ file CSV...")
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=['label']).values
y = df['label'].values

print(f"Tổng số mẫu dữ liệu: {X.shape[0]}")
print(f"Số lượng đặc trưng đầu vào: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Số lượng mẫu tập Train: {X_train.shape[0]}")
print(f"Số lượng mẫu tập Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nĐang huấn luyện mô hình SVM Baseline...")

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("Huấn luyện hoàn tất!")

y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n======================================")
print(f"ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {accuracy * 100:.2f}%")
print(f"======================================")

print("\nBáo cáo chi tiết các phân lớp cảm xúc (Classification Report):")
print(classification_report(y_test, y_pred))

print("Ma trận nhầm lẫn (Confusion Matrix):")
labels = sorted(list(set(y)))
cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

import joblib
joblib.dump(svm_model, 'models/svm_emotion_model.pkl')
joblib.dump(scaler, 'models/scaler_1d.pkl')
