import librosa
import soundfile
import os,glob,pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        result = np.array([])
        sample_rate = sound_file.samplerate
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.vstack((result, mfccs)) if result.size else mfccs
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            if chroma.size < mfccs.size:  # 调整chroma的维度
                chroma = np.pad(chroma, (0, mfccs.size - chroma.size))
            else:
                chroma = chroma[:mfccs.size]
            result = np.vstack((result, chroma)) if result.size else chroma
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            if mel.size < mfccs.size:  # 调整mel的维度
                mel = np.pad(mel, (0, mfccs.size - mel.size))
            else:
                mel = mel[:mfccs.size]
            result = np.vstack((result, mel)) if result.size else mel
        return result

# 建立数字到情感的映射字典
emotions ={
    "01":"neutral",
    "02":"calm",
    "03":"happy",
    "04":"sad",
    "05":"angry",
    "06":"fearful",
    "07":"disgust",
    "08":"surprised",
}
observed_emotions = ["neutral",'calm','happy','sad','angry',"fearful"]

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("D:\study\大三上课程资料\数据采集与融合\Audio_Song_Actors_01-24\*\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train, x_test, y_train, y_test = load_data(0.25)
print(x_train.shape, len(y_train))
print(f"feature:{x_train.shape[1]}")
model = MLPClassifier(alpha=0.02,
                      batch_size=256,
                      activation='relu',
                      solver='adam',
                      epsilon=1e-08,
                      hidden_layer_sizes=(300,),
                      learning_rate='adaptive',
                      max_iter=300)
train_model = model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accracy = accuracy_score(y_true=y_test,y_pred=y_pred)
print("Accuracy:{:.2f}%".format(accracy*100))
