import numpy as np
from fastapi import FastAPI, UploadFile, File
import joblib
import librosa
import soundfile
import os
import shutil
from models import VGG  # 确保这与您的项目结构相匹配
from PIL import Image
import torch
from torchvision import transforms
from moviepy.editor import *

app = FastAPI()

model_audio = joblib.load('SRE/SEP.h5')


@app.post("/text", tags=["这是文本情感分析测试接口"])
async def text(text_word: str):
    import erniebot

    erniebot.api_type = 'aistudio'
    erniebot.access_token = '1494077bf8be8c621b617186bed49f507ba25829'

    text_word = text_word

    response = erniebot.ChatCompletion.create(
        model='ernie-bot-turbo',
        messages=[{
            'role': 'user',
            'content': text_word + "这句话是什么情感，不需要解析，只要在以下的选项中回答我，开心，平淡，悲伤，只返回如下格式：XX"
        }])
    return response.get_result()


@app.post("/audio", tags=["这是音频情感分析测试接口"])
async def audio(audio_file: UploadFile = File(...)):
    with soundfile.SoundFile(audio_file.file) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        # 提取音频特征
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        chroma_stft = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        mel_spect = librosa.feature.melspectrogram(y=X, sr=sample_rate)

        # 假设我们只取每种特征的前N个特征的均值
        N = 60  # 根据模型训练情况调整这个值
        mfccs_mean = np.mean(mfccs[:N, :], axis=1)
        chroma_stft_mean = np.mean(chroma_stft[:N, :], axis=1)
        mel_spect_mean = np.mean(mel_spect[:N, :], axis=1)

        # 检查特征维度
        print(
            f"mfccs_shape: {mfccs_mean.shape}, chroma_shape: {chroma_stft_mean.shape}, mel_shape: {mel_spect_mean.shape}")

        # 组合特征
        result = np.hstack((mfccs_mean, chroma_stft_mean, mel_spect_mean))

        # 将结果数组变形为2D，适合模型预测使用
        result = result.reshape(1, -1)

        # 预测情感
        y_pred = model_audio.predict(result)

        # 清理保存的临时音频文件
        # os.remove(input_file_path)
        # os.remove(output_file_path)

        # 返回预测结果
        return {"prediction": y_pred[0]}


@app.post("/predict-emotion", tags=["这是图片情感分析测试接口"])
async def predict_emotion(file: UploadFile = File(...)):
    os.makedirs('file', exist_ok=True)

    file_name, file_extension = os.path.splitext(file.filename)
    # 构建文件保存路径
    file_path = f'file/{file.filename}{file_extension}'

    # 使用异步方式保存上传的文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # 使用shutil模块来保存上传的文件

    model = VGG('VGG19')
    checkpoint = torch.load(r'CK+_VGG19\1\Best_Test_model.pth')

    model.eval()

    # 图像预处理
    image_path = file_path
    image = Image.open(image_path).convert('RGB')  # 转换为 RGB
    transform = transforms.Compose([
        transforms.Resize((44, 44)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)

    # 推断
    with torch.no_grad():
        output = model(input_tensor)

    # 获取预测结果
    _, predicted = torch.max(output.data, 1)

    # 类别标签
    classes = ['生气', '厌恶', '害怕', '开心', '伤心', '惊讶', '蔑视']

    # 打印预测结果
    return {"预测类别:": classes[predicted.item()]}


@app.post("/video", tags=["这是视频情感分析测试接口"])
async def video(file: UploadFile = File(...)):
    try:
        os.makedirs('video', exist_ok=True)

        file_name, file_extension = os.path.splitext(file.filename)
        # 构建文件保存路径
        input_file_path = f'video/{file_name}{file_extension}'
        output_file_path = f'video/{file_name}.wav'
        # 使用异步方式保存上传的文件
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  # 使用 shutil 模块来保存上传的文件

        # 将视频转换为音频文件
        video = VideoFileClip(input_file_path)
        video.audio.write_audiofile(output_file_path)

        # 从音频文件提取特征
        with soundfile.SoundFile(output_file_path) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            # 提取音频特征
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            chroma_stft = librosa.feature.chroma_stft(y=X, sr=sample_rate)
            mel_spect = librosa.feature.melspectrogram(y=X, sr=sample_rate)

            # 假设我们只取每种特征的前N个特征的均值
            N = 60  # 根据模型训练情况调整这个值
            mfccs_mean = np.mean(mfccs[:N, :], axis=1)
            chroma_stft_mean = np.mean(chroma_stft[:N, :], axis=1)
            mel_spect_mean = np.mean(mel_spect[:N, :], axis=1)

            # 检查特征维度
            print(
                f"mfccs_shape: {mfccs_mean.shape}, chroma_shape: {chroma_stft_mean.shape}, mel_shape: {mel_spect_mean.shape}")

            # 组合特征
            result = np.hstack((mfccs_mean, chroma_stft_mean, mel_spect_mean))

            # 将结果数组变形为2D，适合模型预测使用
            result = result.reshape(1, -1)

            # 预测情感
            y_pred = model_audio.predict(result)

            # 清理保存的临时音频文件
            # os.remove(input_file_path)
            # os.remove(output_file_path)

            # 返回预测结果
            return {"prediction": y_pred[0]}
    except Exception as e:
        # 清理可能产生的任何临时文件
        # if os.path.exists(input_file_path):
        #     os.remove(input_file_path)
        # if os.path.exists(output_file_path):
        #     os.remove(output_file_path)
        # 如果发生异常，返回错误信息
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    #
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
    #             ssl_keyfile="chenziyang.top.key",ssl_certfile="chenziyang.top.pem")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
