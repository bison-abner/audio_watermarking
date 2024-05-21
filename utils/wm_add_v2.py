from utils import silent_util
import torch
import numpy as np
from utils import bin_util
# 固定模式水印
fix_pattern = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
               1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
               1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0,
               0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]


def create_parcel_message(len_start_bit, num_bit, wm_text, verbose=False):
    # 创建包裹消息函数，生成起始bit、信息内容和封装信息
    # 1. 起始bit
    # start_bit = np.array([0] * len_start_bit)
    start_bit = fix_pattern[0:len_start_bit]
    error_prob = 2 ** len_start_bit / 10000

    if verbose:
        print("起始bit长度:%d,错误率:%.1f万" % (len(start_bit), error_prob))

    # 3.信息内容
    length_msg = num_bit - len(start_bit)
    if wm_text:
        msg_arr = bin_util.hexStr2BinArray(wm_text)
    else:
        msg_arr = np.random.choice([0, 1], size=length_msg)

    # 4.封装信息
    watermark = np.concatenate([start_bit, msg_arr])  # 将起始bit和信息内容连接起来
    assert len(watermark) == num_bit  # 断言确保水印长度正确
    return start_bit, msg_arr, watermark


import time


def add_watermark(bir_array, data, num_point, shift_range, device, model, silence_check=False):
    t1 = time.time()
    # 1.获得区块大小
    chunk_size = num_point + int(num_point * shift_range)

    output_chunks = []
    idx_trunck = -1
    for i in range(0, len(data), chunk_size):
        idx_trunck += 1
        current_chunk = data[i:i + chunk_size].copy()
        # 最后一块，长度不足
        if len(current_chunk) < chunk_size:
            output_chunks.append(current_chunk)
            break

        # 处理区块: [水印区|间隔区]
        current_chunk_cover_area = current_chunk[0:num_point]
        current_chunk_shift_area = current_chunk[num_point:]
        current_chunk_cover_area_wmd = encode_trunck_with_silence_check(silence_check,
                                                                        idx_trunck,
                                                                        current_chunk_cover_area, bir_array,
                                                                        device, model)
        output = np.concatenate([current_chunk_cover_area_wmd, current_chunk_shift_area])
        assert output.shape == current_chunk.shape
        output_chunks.append(output)

    assert len(output_chunks) > 0
    reconstructed_array = np.concatenate(output_chunks)  # 将处理后的区块连接成完整的数组
    time_cost = time.time() - t1
    return data, reconstructed_array, time_cost


def encode_trunck_with_silence_check(silence_check, trunck_idx, trunck, wm, device, model):
    # 编码区块函数（带静音检测）
    # 1. 判断是否是静音,通过判断子段是否静音来处理
    if silence_check and silent_util.is_silent(trunck):  # 如果启用了静音检测并且当前区块是静音的，则直接返回当前区块
        print("跳过静音区块:", trunck_idx)
        return trunck

    # 2.加入水印
    trnck_wmd = encode_trunck(trunck, wm, device, model) # 将水印编码到区块中
    return trnck_wmd


def encode_trunck(trunck, wm, device, model):
    with torch.no_grad():
        signal = torch.FloatTensor(trunck).to(device)[None]  # 将区块转换为张 量
        message = torch.FloatTensor(np.array(wm)).to(device)[None]
        signal_wmd_tensor = model.encode(signal, message)
        signal_wmd = signal_wmd_tensor.detach().cpu().numpy().squeeze()  # 将编码后的张量转换为NumPy数组
        return signal_wmd
