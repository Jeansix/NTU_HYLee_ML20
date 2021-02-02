import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle
import os


def network_slimming(old_model, new_model):
    # 1. get state dictionary
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    selected_idx = []
    for i in range(8):
        # 抓取第i层cnn的权重系数
        importance = params[f'cnn.{i}.1.weight']
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        # 较大的放在前面
        ranking = torch.argsort(importance, descending=True)
        selected_idx.append(ranking[:new_dim])
    now_processed = 1
    for (name1, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # cnn层才移植参数
        # 如果是fc层，或者该参数只有一个数字(如batchnorm的tracenum)，就直接复制
        if name1.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # 当处理到pointwise的weight，将now_processed+1，表示该层的移植已经完成
            if name1.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1
            # pointwise的weight会被上一层pruning和下一层pruning共同影响，需要特判
            if name1.endwith('3.weight'):
                # 最后一层cnn，输出的neuron不用prune
                if len(selected_idx) == now_processed:
                    new_params[name1] = p1[:, selected_idx[now_processed - 1]]
                # 否则，按照上层和下层选择的index进行移植
                else:
                    new_params[name1] = p1[selected_idx[now_processed]][:, selected_idx[now_processed - 1]]
            else:
                new_params[name1] = p1[selected_idx[now_processed]]
        else:
            new_params[name1] = p1
    new_model.load_state_dict(new_params)
    return new_model


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的cross entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 将logits的log_softmax对目标概率(teacher的logits/T后softmax)做KL Divergence
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                    F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss


def encode16(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)  # 将param压缩成16bit
        else:
            custom_dict[name] = param
    pickle.dump(custom_dict, open(fname, 'wb'))  # 存入fname


def decode16(fname):
    params = pickle.load(open(fname, 'rb'))  # 从fname获取params
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param
    return custom_dict


def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param
    pickle.dump(custom_dict, open(fname, 'wb'))  # 存入fname


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))  # 从fname获取params
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)
        custom_dict[name] = param
    return custom_dict


if __name__ == "__main__":
    params = torch.load('student_custom_small.bin')
    encode8(params, '8_bit_model.pkl')
    print(f"8-bit cost: {os.stat('8_bit_model.pkl').st_size} bytes.")
