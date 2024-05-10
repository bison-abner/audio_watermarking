import pdb

import torch
import numpy as np
from utils import bin_util


def decode_trunck(trunck, model, device):
    with torch.no_grad():
        signal = torch.FloatTensor(trunck).to(device).unsqueeze(0)
        message = (model.decode(signal) >= 0.5).int()
        message = message.detach().cpu().numpy().squeeze()
    return message


def is_start_bit_match(start_bit, decoded_start_bit, start_bit_ber_threshold):
    assert decoded_start_bit.shape == start_bit.shape
    ber = 1 - np.mean(start_bit == decoded_start_bit)
    return ber < start_bit_ber_threshold


def extract_watermark(data, start_bit, shift_range, num_point, start_bit_ber_threshold, model, device,
                      verbose=False):
    # pdb.set_trace()
    shift_range_points = int(shift_range * num_point)
    i = 0  # 当前的指针位置
    results = []
    while True:
        start = i
        end = start + num_point
        trunck = data[start:end]
        if len(trunck) < num_point:
            break

        bit_array = decode_trunck(trunck, model, device)
        decoded_start_bit = bit_array[0:len(start_bit)]
        if not is_start_bit_match(start_bit, decoded_start_bit, start_bit_ber_threshold):
            i = i + shift_range_points
            continue
        # 寻找到了起始位置
        if verbose:
            msg_bit = bit_array[len(start_bit):]
            msg_str = bin_util.binArray2HexStr(msg_bit)
            print(i, "解码信息:", msg_str)
        results.append(bit_array)
        i = i + num_point + shift_range_points

    support_count = len(results)
    if support_count == 0:
        mean_result = None
        first_result = None
        exist_prob = None
    else:
        mean_result = (np.array(results).mean(axis=0) >= 0.5).astype(int)
        exist_prob = (mean_result[0:len(start_bit)] == start_bit).mean()
        first_result = results[0]

    return support_count, exist_prob, mean_result, first_result


def extract_watermark_v2(data, start_bit, shift_range, num_point,
                         start_bit_ber_threshold, model, device,
                         merge_type,
                         shift_range_p=0.5, ):
    shift_range_points = int(shift_range * num_point * shift_range_p)
    i = 0  # 当前的指针位置
    results = []
    while True:
        start = i
        end = start + num_point
        trunck = data[start:end]
        if len(trunck) < num_point:
            break

        bit_array = decode_trunck(trunck, model, device)
        decoded_start_bit = bit_array[0:len(start_bit)]

        ber_start_bit = 1 - np.mean(start_bit == decoded_start_bit)
        if ber_start_bit > start_bit_ber_threshold:
            i = i + shift_range_points
            continue
        # 寻找到了起始位置
        results.append({
            "sim": 1 - ber_start_bit,
            "msg": bit_array,
        })
        # 这里很重要，如果threshold设置的太大，那么就会跳过一些可能的点
        # i = i + num_point + shift_range_points
        i = i + shift_range_points

    support_count = len(results)
    if support_count == 0:
        mean_result = None
    else:
        # 1.加权得到最终结果
        if merge_type == "weighted":
            raise Exception("")
        elif merge_type == "best":
            # 相似度从大到小排序
            best_val = sorted(results, key=lambda x: x["sim"], reverse=True)[0]
            if np.isclose(1.0, best_val["sim"]):
                # 那么对所有为1.0的进行求平均
                results_1 = [i["msg"] for i in results if np.isclose(i["sim"], 1.0)]
                mean_result = (np.array(results_1).mean(axis=0) >= 0.5).astype(int)
            else:
                mean_result = best_val["msg"]

        else:
            raise Exception("")
            # assert merge_type == "mean"
            # mean_result = (np.array([i[-1] for i in results]).mean(axis=0) >= 0.5).astype(int)

    return support_count, mean_result, results
