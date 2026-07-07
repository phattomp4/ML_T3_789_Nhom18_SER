import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- 1. ĐỌC DỮ LIỆU ĐÃ TRÍCH XUẤT ---
CSV_PATH = "data/ravdess_features_1d.csv"

if not os.path.exists(CSV_PATH):
    print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy chạy file build_dataset_1d.py trước!")
    exit()

print("Đang tải dữ liệu từ file CSV...")
df = pd.read_csv(CSV_PATH)

# Tách đặc trưng (X) và nhãn (y)
# X lấy từ cột feature_1 đến feature_80
X = df.drop(columns=['label']).values
y = df['label'].values

print(f"Tổng số mẫu dữ liệu: {X.shape[0]}")
print(f"Số lượng đặc trưng đầu vào: {X.shape[1]}")

# --- 2. CHIA TẬP TRAIN / TEST ---
# test_size=0.2 tương đương 20% dữ liệu dùng để test
# random_state=42 giúp cố định kết quả chia giữa các lần chạy để tiện so sánh
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Số lượng mẫu tập Train: {X_train.shape[0]}")
print(f"Số lượng mẫu tập Test: {X_test.shape[0]}")

# --- 3. CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. HUẤN LUYỆN MÔ HÌNH SVM ---
print("\nĐang huấn luyện mô hình SVM Baseline...")
# Khởi tạo SVM với các tham số mặc định ban đầu (Kernel RBF là lựa chọn mạnh mẽ nhất)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("Huấn luyện hoàn tất!")

# --- 5. ĐÁNH GIÁ MÔ HÌNH ---
# Dự đoán trên tập dữ liệu Test chưa từng được học
y_pred = svm_model.predict(X_test_scaled)

# Tính độ chính xác tổng thể
accuracy = accuracy_score(y_test, y_pred)
print(f"\n======================================")
print(f"ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {accuracy * 100:.2f}%")
print(f"======================================")

# Hiển thị bảng báo cáo chi tiết (Precision - độ chính xác, Recall - độ bao phủ, F1-Score cho từng cảm xúc)
print("\nBáo cáo chi tiết các phân lớp cảm xúc (Classification Report):")
print(classification_report(y_test, y_pred))

# Hiển thị Ma trận nhầm lẫn (Confusion Matrix)
print("Ma trận nhầm lẫn (Confusion Matrix):")
labels = sorted(list(set(y)))
cm = confusion_matrix(y_test, y_pred, labels=labels)

# In ma trận dạng bảng dễ nhìn
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# Gợi ý: Bạn có thể dùng thư viện joblib để lưu mô hình lại phục vụ làm ứng dụng sau này
import joblib
joblib.dump(svm_model, 'models/svm_emotion_model.pkl')
joblib.dump(scaler, 'models/scaler_1d.pkl')