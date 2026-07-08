import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
CSV_PATH = "data/ravdess_features_1d.csv"
if not os.path.exists(CSV_PATH):
    print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy kiểm tra lại!")
    exit()

df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['label']).values
y_string = df['label'].values # Nhãn gốc dạng chuỗi ('happy', 'sad'...)

# --- MÃ HÓA NHÃN TỪ CHỮ SANG SỐ (Sửa lỗi ufunc isnan) ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_string) 
# Lúc này nhãn đã chuyển thành số nguyên từ 0 đến 7

print("Bảng ánh xạ nhãn cảm xúc:")
for index, class_label in enumerate(label_encoder.classes_):
    print(f"  Mã số {index} -> Cảm xúc: {class_label}")

# Phân chia tập dữ liệu Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. THIẾT LẬP VÀ HUẤN LUYỆN MLP ---
print("\nĐang khởi tạo và huấn luyện mạng MLP...")
print("Cấu trúc mạng: 3 lớp ẩn (256, 128, 64 neurons)")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',                 
    solver='adam',                     
    alpha=0.001,                       
    batch_size=32,
    learning_rate='adaptive',
    max_iter=500,                      
    early_stopping=True,               
    random_state=42,
    verbose=True                       
)

# Tiến hành huấn luyện
mlp_model.fit(X_train_scaled, y_train)

# --- 3. ĐÁNH GIÁ MÔ HÌNH ---
y_pred = mlp_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n======================================")
print(f"ĐỘ CHÍNH XÁC CỦA MLP (TEST ACCURACY): {accuracy * 100:.2f}%")
print(f"======================================")

# Sử dụng target_names để hiển thị lại tên chữ gốc trên báo cáo kết quả
print("\nBáo cáo chi tiết MLP:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- 4. LƯU MÔ HÌNH VÀ BỘ MÃ HÓA ---
os.makedirs('models', exist_ok=True)
joblib.dump(mlp_model, 'models/mlp_emotion_model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("Đã lưu mô hình MLP và bộ mã hóa nhãn thành công vào thư mục 'models/'")