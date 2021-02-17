import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def count_parameters(model, only_trainable=False):
    """
    计算参数数量
    :param model: 训练的模型
    :param only_trainable: 当前模式是否是train
    :return:
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def same_seed(seed):
    """
    固定训练的随机种子，以便reproduce
    :param seed:当前的种子
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_acc(label, pred):
    """
    Computes categorization accuracy of our task.
    :param label: ground truth labels (9000,)
    :param pred: predicted labels
    :return:acc:0~1 accuracy
    """
    correct = np.sum(label == pred)
    acc = correct / label.shape[0]
    return max(acc, 1 - acc)  # binary unsupervised clustering


def plot_scatter(feat, label, savefig=None):
    """
    Plot scatter image
    :param feat:点集
    :param label:真实标签
    :param savefig:图片存放目录
    :return:
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c=label)
    plt.legend(loc="best")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return
