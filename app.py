import base64
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
from blind_watermark import bw_notes, WaterMark
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 关闭启动信息
bw_notes.close()
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

# 函数：嵌入水印
def embed_watermark(input_path, output_path, watermark_text='felix is the best!'):
    bwm1 = WaterMark(password_img=1, password_wm=1)
    bwm1.read_img(input_path)
    bwm1.read_wm(watermark_text, mode='str')
    bwm1.embed(output_path)
    len_wm = len(bwm1.wm_bit)
    return len_wm  # 返回水印长度

# 函数：提取水印
def extract_watermark(input_path, len_wm):
    bwm1 = WaterMark(password_img=1, password_wm=1)
    wm_extract = bwm1.extract(input_path, wm_shape=len_wm, mode='str')
    return wm_extract



# 主应用程序
def audio_watermark_module():
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


def image_watermark_module():
    st.title("深度学习图片水印")
    st.write("选择您要执行的操作:")

    # 选择操作：添加水印 or 解码水印
    action = st.selectbox("选择操作", ["添加水印", "解码水印"])

    # 检查并创建输出文件夹
    output_folder = 'C:/temp/output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if action == "添加水印":
        uploaded_img = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
        if uploaded_img is not None:
            st.image(uploaded_img, caption='上传的图片', use_column_width=True)
            watermark_text = st.text_input("请输入水印文本", "felix is the best!")
            output_path = os.path.join(output_folder, uploaded_img.name.split('.')[0] + '_watermarked.png')
            if st.button("嵌入水印"):
                image = Image.open(uploaded_img)
                image.save(output_path)
                len_wm = embed_watermark(output_path, output_path, watermark_text)
                st.success(f"成功嵌入水印！水印长度：{len_wm}")
                st.markdown("🔔 **请记住水印长度，解码时将需要它！**")
                st.image(Image.open(output_path), caption='水印嵌入后的图片', use_column_width=True)
                st.markdown(get_binary_file_downloader_html(output_path, '下载水印图片'), unsafe_allow_html=True)

    elif action == "解码水印":
        uploaded_watermarked_img = st.file_uploader("上传带水印的图片", type=['jpg', 'png', 'jpeg'])
        len_wm = st.number_input("请输入水印长度（嵌入时记下的长度）", min_value=1, step=1)
        if uploaded_watermarked_img is not None and len_wm is not None:
            st.image(uploaded_watermarked_img, caption='上传的带水印图片', use_column_width=True)
            if st.button("提取水印"):
                uploaded_watermarked_img_path = os.path.join(output_folder, uploaded_watermarked_img.name)
                with open(uploaded_watermarked_img_path, 'wb') as f:
                    f.write(uploaded_watermarked_img.read())
                wm_extract = extract_watermark(uploaded_watermarked_img_path, len_wm)
                st.success("成功提取水印！")
                st.text("提取的水印文本:")
                st.write(wm_extract)

# 二进制文件下载器
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href


# 主函数
def main():
    st.title("富文本确权保护系统")

    # 创建侧边栏目录
    selected_module = st.sidebar.radio("模块选择", ["音频确权模块", "图片确权模块"])

    if selected_module == "音频确权模块":
        audio_watermark_module()

    elif selected_module == "图片确权模块":
        image_watermark_module()


if __name__ == "__main__":
    len_start_bit = 12

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = "C:\\Users\\abner\\PycharmProjects\\audio_watermarking\\step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.pkl"

    model = load_model(model_path)
    main()
