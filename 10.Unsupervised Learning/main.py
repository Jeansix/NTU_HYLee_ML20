from data import *
from preprocess import *
from utils import *
from model import *
from train import *
from test import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import glob

if __name__ == "__main__":
    """
    trainX = np.load('trainX.npy')
    trainX_preprocessed = preprocess(trainX)
    img_dataset = ImageDataset(trainX_preprocessed)

    same_seed(0)
    model = AE().cuda()  # 加载到gpu
    criterion = nn.MSELoss()  # define loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    model.train()  # 设置模型模式为train
    n_epoch = 100
    img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)
    """
    # train(img_dataloader, model, optimizer, criterion, n_epoch)
    # test(trainX)

    valX = np.load('valX.npy')
    valY = np.load('valY.npy')
    #  Problem 1:将val data 的降维 (embedding) 与对应的Label画出来
    """
    pred_from_latent, emb_from_latent = test(valX, False)
    acc_latent = cal_acc(valY, pred_from_latent)
    print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_latent, valY, savefig='p1_baseline.png')
    """

    # problem2
    """
    trainX = np.load('trainX.npy')
    trainX_preprocessed = preprocess(trainX)
    # 0.指定图片
    indexes = [1, 2, 3, 6, 7, 9]
    # 1.画出原图
    plt.figure(figsize=(10, 4))
    imgs = trainX[indexes,]
    for i, img in enumerate(imgs):
        plt.subplot(2, 6, i + 1, xticks=[], yticks=[])
        plt.imshow(img)
    # 2.画出reconstruct的图
    inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
    model = AE().cuda()
    model.load_state_dict(torch.load('./checkpoints/last_checkpoint.pth'))
    model.eval()
    latents, recs = model(inp)
    recs = ((recs + 1) / 2).cpu().detach().numpy()
    recs = recs.transpose(0, 2, 3, 1) # 是preprocess的逆过程
    for i, img in enumerate(recs):
        plt.subplot(2, 6, 6 + i + 1, xticks=[], yticks=[])
        plt.imshow(img)

    plt.tight_layout()
    """

    # problem3
    checkpoints_list = sorted(glob.glob('checkpoints/checkpoint_*.pth'),
                              key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    # load data
    trainX = np.load('trainX.npy')
    trainX_preprocessed = preprocess(trainX)
    dataset = ImageDataset(trainX_preprocessed)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    points = []
    model = AE().cuda()
    with torch.no_grad():
        for i, checkpoint in enumerate(checkpoints_list):
            print('[{}/{}] {}'.format(i + 1, len(checkpoints_list), checkpoint))
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            err = 0
            n = 0
            for x in dataloader:
                x = x.cuda()
                _, rec = model(x)
                err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
                n += x.flatten().size(0)
            print('Reconstruction error (MSE):', err / n)
            latents = inference(X=valX, model=model)
            pred, X_embedded = predict(latents)
            acc = cal_acc(valY, pred)
            print('Accuracy:', acc)
            points.append((err / n, acc))
    ps = list(zip(*points))
    plt.figure(figsize=(6, 6))
    plt.subplot(211, title='Reconstruction error (MSE)').plot(ps[0])
    plt.subplot(212, title='Accuracy (val)').plot(ps[1])
    plt.show()
