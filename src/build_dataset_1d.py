import os
import pandas as pd
import numpy as np
# Gọi hàm trích xuất 80 đặc trưng đã cấu hình ở bước trước
from extract_features_1d import process_audio_1d

# Đường dẫn chính xác tới thư mục RAVDESS trên máy của bạn
# Đảm bảo cấu trúc bên trong gồm các thư mục con: Actor_01, Actor_02...
RAVDESS_PATH = "data/RAVDESS"

# Từ điển ánh xạ mã số (phần tử thứ 3 trong tên file) sang tên cảm xúc
EMOTION_DICT = {
    '01': 'neutral', 
    '02': 'calm', 
    '03': 'happy', 
    '04': 'sad',
    '05': 'angry', 
    '06': 'fearful', 
    '07': 'disgust', 
    '08': 'surprised'
}

def build_ravdess_dataset():
    features_list = []
    labels_list = []
    file_count = 0
    
    # Kiểm tra thư mục gốc có tồn tại không
    if not os.path.exists(RAVDESS_PATH):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {RAVDESS_PATH}. Hãy kiểm tra lại đường dẫn!")
        return None

    print("Bắt đầu quét qua các thư mục Actor...")
    
    # Duyệt qua các thư mục Actor_01, Actor_02...
    for actor_dir in sorted(os.listdir(RAVDESS_PATH)):
        actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        
        # Chỉ xử lý nếu đó là thư mục
        if os.path.isdir(actor_path):
            print(f"  Đang xử lý thư mục: {actor_dir}")
            
            # Duyệt qua từng file âm thanh bên trong thư mục Actor
            for file_name in os.listdir(actor_path):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(actor_path, file_name)
                    
                    try:
                        # 1. BÓC TÁCH NHÃN CẢM XÚC
                        # Tên file mẫu: 03-01-05-01-01-01-01.wav
                        parts = file_name.split('-')
                        emotion_code = parts[2] # Lấy số '05'
                        emotion_label = EMOTION_DICT[emotion_code]
                        
                        # 2. TRÍCH XUẤT 80 ĐẶC TRƯNG (MEAN + STD)
                        features = process_audio_1d(file_path)
                        
                        if features is not None:
                            features_list.append(features)
                            labels_list.append(emotion_label)
                            file_count += 1
                            
                    except IndexError:
                        print(f"Cảnh báo: File {file_name} sai định dạng đặt tên của RAVDESS.")
                    except KeyError:
                        print(f"Cảnh báo: Mã cảm xúc không hợp lệ trong file {file_name}.")

    print(f"\n-> Đã xử lý thành công tổng cộng: {file_count} file âm thanh.")
    
    # 3. TẠO DATAFRAME VÀ ĐÓNG GÓI DỮ LIỆU
    df = pd.DataFrame(features_list)
    
    # Tự động đặt tên cột theo số lượng chiều thực tế (1 đến 108)
    num_features = df.shape[1]
    df.columns = [f'feature_{i}' for i in range(1, num_features + 1)]
    
    # Thêm cột nhãn vào cuối bảng
    df['label'] = labels_list
    
    return df

if __name__ == "__main__":
    print("=== TIẾN TRÌNH XỬ LÝ DỮ LIỆU BẢNG 1D ===")
    dataset_df = build_ravdess_dataset()
    
    if dataset_df is not None:
        # Xuất dữ liệu ra file CSV đặt tại thư mục data/
        output_csv = "data/ravdess_features_1d.csv"
        dataset_df.to_csv(output_csv, index=False)
        
        print(f"=== HOÀN TẤT ===")
        print(f"Dữ liệu đã được lưu thành công vào: {output_csv}")
        print(f"Kích thước bảng dữ liệu (Dòng, Cột): {dataset_df.shape}")
        print("\nHển thị 5 dòng dữ liệu đầu tiên mẫu:")
        print(dataset_df.head())