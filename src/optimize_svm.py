import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# --- 1. ĐỌC DỮ LIỆU ĐÃ TRÍCH XUẤT ---
CSV_PATH = "data/ravdess_features_1d.csv"
if not os.path.exists(CSV_PATH):
    print(f"Lỗi: Không tìm thấy file {CSV_PATH}.")
    exit()

df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['label']).values
y = df['label'].values

# Chia tập Train/Test đồng nhất với file train_svm.py trước đó
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM (PARAM GRID) ---
# Thử nghiệm các giá trị C và Gamma khác nhau để tìm điểm tối ưu
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf'] # Tập trung tối ưu cho kernel mạnh nhất là RBF
}

print("=== BẮT ĐẦU QUÁ TRÌNH GRID SEARCH CV ===")
print(f"Đang cấu hình quét {len(param_grid['C']) * len(param_grid['gamma'])} tổ hợp tham số khác nhau...")

# --- 3. KHỞI TẠO VÀ CHẠY GRID SEARCH ---
# cv=5: Chia tập Train thành 5 phần để kiểm định chéo (5-fold Cross Validation)
# n_jobs=-1: Sử dụng tối đa tất cả các lõi CPU của máy để chạy song song cho nhanh
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)

# --- 4. XUẤT KẾT QUẢ TỐI ƯU ---
print("\n=== QUÁ TRÌNH TỐI ƯU HOÀN TẤT ===")
print(f"Bộ siêu tham số tốt nhất tìm được: {grid_search.best_params_}")
print(f"Độ chính xác tốt nhất trên tập Validation (Cross-Val Score): {grid_search.best_score_ * 100:.2f}%")

# Lấy mô hình xuất sắc nhất sau khi tối ưu
best_svm_model = grid_search.best_estimator_

# --- 5. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP ---
y_pred = best_svm_model.predict(X_test_scaled)
optimized_accuracy = accuracy_score(y_test, y_pred)

print(f"\n======================================")
print(f"ĐỘ CHÍNH XÁC SAU KHI TỐI ƯU (TEST ACCURACY): {optimized_accuracy * 100:.2f}%")
print(f"======================================")

print("\nBáo cáo chi tiết sau khi tối ưu:")
print(classification_report(y_test, y_pred))

# --- 6. LƯU MÔ HÌNH XUẤT SẮC NHẤT ---
os.makedirs('models', exist_ok=True)
joblib.dump(best_svm_model, 'models/best_svm_emotion_model.pkl')
joblib.dump(scaler, 'models/scaler_1d.pkl')
print("Đã lưu mô hình tối ưu và bộ chuẩn hóa vào thư mục 'models/'")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Vẽ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y))) # Lấy danh sách tên cảm xúc

plt.figure(figsize=(10, 8))
# Dùng thư viện seaborn để vẽ ma trận màu xanh dương (Blues)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Ma trận nhầm lẫn - Mô hình SVM (108 chiều)', fontsize=14)
plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)

# Lưu ảnh ra file png
plt.tight_layout()
plt.savefig('confusion_matrix_svm.png', dpi=300)
print("Đã lưu ảnh ma trận nhầm lẫn thành file 'confusion_matrix_svm.png'")