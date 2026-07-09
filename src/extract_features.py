import os
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "../data/RAVDESS"
IMAGE_PATH = "../data/images"

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
EMOTION_DICT = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def create_spectrogram(wav_path, save_path):
    y, sr = librosa.load(wav_path, sr=22050)
    y, index = librosa.effects.trim(y, top_db=20)
    fixed_length = 22050 * 3
    if len(y) > fixed_length:
        y = y[:fixed_length]
    else:
        y = np.pad(y, (0, fixed_length - len(y)))
        
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    plt.figure(figsize=(3, 3))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel_spect_db, cmap='viridis')
    
    plt.savefig(save_path, bbox_inches=None, pad_inches=0)
    plt.close()

def process_and_split_data():
    print("BƯỚC 1: Bắt đầu gọt khoảng lặng và chuyển WAV sang ảnh PNG...")
    
    for emo in EMOTIONS:
        os.makedirs(os.path.join(IMAGE_PATH, emo), exist_ok=True)
        
    all_images = []
    all_labels = []

    for actor_dir in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if os.path.isdir(actor_path):
            for file_name in os.listdir(actor_path):
                if file_name.endswith(".wav"):
                    wav_path = os.path.join(actor_path, file_name)
                    
                    emo_code = file_name.split('-')[2]
                    emo_label = EMOTION_DICT[emo_code]
                    img_name = file_name.replace(".wav", ".png")
                    save_path = os.path.join(IMAGE_PATH, emo_label, img_name)
                    
                    if not os.path.exists(save_path):
                        create_spectrogram(wav_path, save_path)
                    
                    all_images.append(save_path)
                    all_labels.append(emo_label)

    print(f"-> Đã tạo xong {len(all_images)} ảnh Spectrogram!")
    
    print("BƯỚC 2: Chia tập Train (70%) / Val (15%) / Test (15%)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_images, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )
    
    print(f"-> Số lượng: Train ({len(X_train)}), Val ({len(X_val)}), Test ({len(X_test)})")
    
    splits = {
        '../data/train': (X_train, y_train),
        '../data/val': (X_val, y_val),
        '../data/test': (X_test, y_test)
    }
    
    print("BƯỚC 3: Đang phân bổ ảnh vào 3 thư mục chuẩn...")
    for split_dir, (x_data, y_data) in splits.items():
        for img_path, label in zip(x_data, y_data):
            dest_dir = os.path.join(split_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
            if not os.path.exists(dest_path):
                shutil.copy(img_path, dest_path)
                
    print("HOÀN TẤT GIAI ĐOẠN 2! Dữ liệu đã sẵn sàng để train ResNet.")

if __name__ == "__main__":
    process_and_split_data()
