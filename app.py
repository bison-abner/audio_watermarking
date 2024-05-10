import pdb
import time
import streamlit as st
import os
from utils import wm_add_v2, file_reader, model_util, wm_decode_v2, bin_util
from models import my_model_v7_recover
import torch
import uuid
import datetime
import numpy as np
import soundfile

# 功能：给音频添加水印
def add_watermark(audio_path, watermark_text):
    assert len(watermark_text) == 5

    start_bit, msg_bit, watermark = wm_add_v2.create_parcel_message(len_start_bit, 32, watermark_text)

    data, sr, audio_length_second = file_reader.read_as_single_channel_16k(audio_path, 16000)

    _, signal_wmd, time_cost = wm_add_v2.add_watermark(watermark, data, 16000, 0.1, device, model)

    tmp_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + str(uuid.uuid4()) + ".wav"
    tmp_file_path = 'C:/temp/' + tmp_file_name
    soundfile.write(tmp_file_path, signal_wmd, sr)
    return tmp_file_path

# 功能：从音频中解码水印
def decode_watermark(audio_path):
    data, sr, audio_length_second = file_reader.read_as_single_channel_16k(audio_path, 16000)
    data = data[0:5 * sr]
    start_bit = wm_add_v2.fix_pattern[0:len_start_bit]
    support_count, mean_result, results = wm_decode_v2.extract_watermark_v2(
        data,
        start_bit,
        0.1,
        16000,
        0.3,
        model,
        device, "best")

    if mean_result is None:
        return "无水印"

    payload = mean_result[len_start_bit:]
    return bin_util.binArray2HexStr(payload)

# 主应用程序
def main():
    if "def_value" not in st.session_state:
        st.session_state.def_value = bin_util.binArray2HexStr(np.random.choice([0, 1], size=32 - len_start_bit))

    st.title("深度学习音频水印")
    st.write("选择您要执行的操作:")

    action = st.selectbox("选择操作", ["添加水印", "解码水印"])

    if action == "添加水印":
        audio_file = st.file_uploader("上传音频文件 (WAV格式)", type=["wav"], accept_multiple_files=False)
        if audio_file:
            tmp_input_audio_file = os.path.join("C:/temp/", audio_file.name)
            with open(tmp_input_audio_file, "wb") as f:
                f.write(audio_file.getbuffer())
            st.audio(tmp_input_audio_file, format="audio/wav")

            watermark_text = st.text_input("输入水印文本（5个英文字母）", value=st.session_state.def_value)

            add_watermark_button = st.button("添加水印")
            if add_watermark_button:
                if audio_file and watermark_text:
                    with st.spinner("正在添加水印..."):
                        t1 = time.time()

                        watermarked_audio = add_watermark(tmp_input_audio_file, watermark_text)
                        encode_time_cost = time.time() - t1

                        st.write("水印音频：")
                        st.audio(watermarked_audio, format="audio/wav")
                        st.write("耗时：%d秒" % encode_time_cost)

    elif action == "解码水印":
        audio_file = st.file_uploader("上传音频文件 (WAV/MP3格式)", type=["wav", "mp3"], accept_multiple_files=False)
        if audio_file:
            if st.button("解码水印"):
                tmp_file_for_decode_path = os.path.join("C:/temp/", audio_file.name)
                with open(tmp_file_for_decode_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                with st.spinner("正在解码..."):
                    t1 = time.time()
                    decoded_watermark = decode_watermark(tmp_file_for_decode_path)
                    decode_cost = time.time() - t1

                print("解码水印", decoded_watermark)
                st.write("解码水印：", decoded_watermark)
                st.write("耗时：%d秒" % (decode_cost))

def load_model(resume_path):
    n_fft = 1000
    hop_length = 400

    model = my_model_v7_recover.Model(16000, 32, n_fft, hop_length,
                                      use_recover_layer=False, num_layers=8).to(device)
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    state_dict = model_util.map_state_dict(checkpoint['model'])
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

if __name__ == "__main__":
    len_start_bit = 12

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = "C:\\Users\\abner\\PycharmProjects\\audio_watermarking\\step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.pkl"

    model = load_model(model_path)
    main()
