import librosa
import numpy as np

TARGET_SR = 22050     
MAX_DURATION = 3.0    
FIXED_LENGTH = int(TARGET_SR * MAX_DURATION) 

def process_audio_1d(file_path):
    """
    Trích xuất tổ hợp đặc trưng 108 chiều (MFCC + ZCR + RMS + Chroma) kèm Mean và Std.
    """
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # 1. Chuẩn hóa độ dài (Padding / Truncating)
        if len(y) > FIXED_LENGTH:
            y = y[:FIXED_LENGTH]
        else:
            padding_length = FIXED_LENGTH - len(y)
            y = np.pad(y, (0, padding_length), mode='constant')
            
        # 2. Trích xuất các đặc trưng âm học
        # 2.1 MFCCs (40 dải tần số)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 2.2 Zero Crossing Rate (ZCR - Đo mức độ gắt, bạo phát âm thanh)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr.T, axis=0)
        zcr_std = np.std(zcr.T, axis=0)
        
        # 2.3 Root Mean Square Energy (RMS - Đo độ to/nhỏ thực tế của giọng nói)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms.T, axis=0)
        rms_std = np.std(rms.T, axis=0)
        
        # 2.4 Chroma STFT (12 dải phổ cao độ nhạc lý - bắt sóng lên xuống giọng)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # 3. Ghép tất cả thành một vector phẳng 1D duy nhất
        # Tổng số chiều: (40+40) + (1+1) + (1+1) + (12+12) = 108 chiều
        features_combined = np.hstack([
            mfccs_mean, mfccs_std,
            zcr_mean, zcr_std,
            rms_mean, rms_std,
            chroma_mean, chroma_std
        ])
        
        return features_combined

    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Test nhanh cấu hình đặc trưng mới
    test_file = "data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav" 
    features = process_audio_1d(test_file)
    if features is not None:
        print(f"Nâng cấp thành công! Vector đặc trưng mới có kích thước: {features.shape} chiều.")