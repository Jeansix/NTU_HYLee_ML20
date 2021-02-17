import torch
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from preprocess import *
from data import *
from model import *
from torch.utils.data import DataLoader


def inference(X, model, batch_size=256):
    """
    得到encoder的结果
    :param X: raw image
    :param model: 训练好的模型
    :param batch_size: batch大小
    :return: latents:中间结果，还需要进行进一步预测
    """
    X = preprocess(X)
    dataset = ImageDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()  # detach()切断了反向传播
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis=0)
    print('Latents Shape:', latents.shape)
    return latents


def invert(pred):
    return np.abs(1 - pred)


def predict(latents):
    """
    进行cluster后分类
    :param latents: 经过inference之后得到的结果
    :return:pred:
    :return:x_embedded:
    """
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded


def save_prediction(pred, out_csv="prediction.csv"):
    """
    保存预测结果
    :param pred: 预测结果
    :param out_csv: 保存路径
    :return:
    """
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
        print(f'Save prediction to {out_csv}.')


def test(trainX, save=True):
    # load model
    model = AE().cuda()
    model.load_state_dict(torch.load('./checkpoints/last_checkpoint.pth'))
    model.eval()
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)
    if save is True:
        save_prediction(pred, 'prediction.csv')
        save_prediction(invert(pred), 'prediction_invert.csv')
    return pred, X_embedded
