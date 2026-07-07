import librosa
import numpy as np

# --- THIẾT LẬP CÁC THÔNG SỐ TOÀN CỤC ---
TARGET_SR = 22050     # Tần số lấy mẫu chuẩn của librosa
MAX_DURATION = 3.0    # Đặt độ dài chuẩn là 3 giây (bạn có thể điều chỉnh sau khi EDA)
FIXED_LENGTH = int(TARGET_SR * MAX_DURATION) # Tổng số lượng điểm dữ liệu cho 3 giây

def process_audio_1d(file_path):
    """
    Đọc file wav, chuẩn hóa độ dài về 3s, và trích xuất vector MFCC 1D.
    """
    try:
        # 1. Load âm thanh
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # 2. Chuẩn hóa độ dài (Padding / Truncating)
        if len(y) > FIXED_LENGTH:
            # Nếu câu nói dài hơn 3s -> Cắt bỏ phần đuôi (Truncating)
            y = y[:FIXED_LENGTH]
        else:
            # Nếu câu nói ngắn hơn 3s -> Chèn thêm khoảng lặng (số 0) vào cuối (Padding)
            padding_length = FIXED_LENGTH - len(y)
            y = np.pad(y, (0, padding_length), mode='constant')
            
        # 3. Trích xuất đặc trưng MFCC
        # n_mfcc=40 là con số tối ưu thường dùng trong Speech Emotion Recognition
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 3. Trích xuất đặc trưng MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 4. Tính Mean và Std, sau đó ghép lại
        # Tính giá trị trung bình (kích thước 40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Tính độ lệch chuẩn (kích thước 40)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Ghép 2 mảng 1D này lại với nhau theo chiều ngang (Horizontal Stack)
        # Kết quả sẽ là một vector 1D có kích thước 80
        mfccs_combined = np.hstack([mfccs_mean, mfccs_std])
        
        return mfccs_combined
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

# --- CHẠY THỬ NGHIỆM TRÊN 1 FILE ---
if __name__ == "__main__":
    # Thay đường dẫn này bằng 1 file âm thanh thực tế trên máy bạn
    test_file = "data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav" 
    
    features = process_audio_1d(test_file)
    
    if features is not None:
        print(f"Trích xuất thành công! Kích thước vector 1D: {features.shape}")
        print(f"Một vài giá trị đầu tiên:\n{features[:5]}")