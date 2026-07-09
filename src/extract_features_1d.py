import librosa
import numpy as np

TARGET_SR = 22050
MAX_DURATION = 3.0
FIXED_LENGTH = int(TARGET_SR * MAX_DURATION)

def process_audio_1d(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        
        if len(y) > FIXED_LENGTH:
            y = y[:FIXED_LENGTH]
        else:
            padding_length = FIXED_LENGTH - len(y)
            y = np.pad(y, (0, padding_length), mode='constant')
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr.T, axis=0)
        zcr_std = np.std(zcr.T, axis=0)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms.T, axis=0)
        rms_std = np.std(rms.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
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
    test_file = "data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
    features = process_audio_1d(test_file)
    if features is not None:
        print(f"Trích xuất thành công! Kích thước vector 1D: {features.shape}")
        print(f"Một vài giá trị đầu tiên:\n{features[:5]}")
