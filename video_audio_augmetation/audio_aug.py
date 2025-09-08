import torch
import torchaudio
import soundfile as sf
import os

SEGAN_MODEL_PATH = "/home/wangyuan/SEGAN_ckpt"
PRETRAINED_WEIGHTS_PATH = os.path.join(SEGAN_MODEL_PATH, "segan_generator.pth")
INPUT_WAV_PATH = "path/to/your/input.wav"
OUTPUT_WAV_PATH = "path/to/your/output_enhanced.wav"

import sys
sys.path.append(SEGAN_MODEL_PATH)
from segan_model import SEGANGenerator  # 这里的SEGANGenerator是假设的模型类名

def enhance_audio_with_segan(input_path: str, output_path: str, model_path: str):
    try:
        model = SEGANGenerator()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # 切换到评估模式
        print("SEGAN模型加载成功。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    try:
        waveform, sample_rate = torchaudio.load(input_path)
        print(f"音频文件 '{os.path.basename(input_path)}' 加载成功。采样率: {sample_rate} Hz")
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # 转为单声道
        input_tensor = waveform
        
    except Exception as e:
        print(f"加载或处理音频文件时出错: {e}")
        return

    print("voice enhance...")
    with torch.no_grad(): 
        enhanced_waveform = model(input_tensor)

    enhanced_waveform = enhanced_waveform.squeeze(0).numpy()
    try:
        sf.write(output_path, enhanced_waveform, sample_rate)
        print(f"音频增强完成。已保存到 '{os.path.basename(output_path)}'")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_WAV_PATH):
        print(f"错误: 输入文件 '{INPUT_WAV_PATH}' 不存在。")
    elif not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"错误: 模型权重文件 '{PRETRAINED_WEIGHTS_PATH}' 不存在。")
    else:
        enhance_audio_with_segan(INPUT_WAV_PATH, OUTPUT_WAV_PATH, PRETRAINED_WEIGHTS_PATH)