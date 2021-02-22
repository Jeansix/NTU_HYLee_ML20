import torch
import torch.nn as nn
import pandas as pd
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data import *
from train import *
from model import *
from inference import *

source_dir = 'real_or_drawing/train_data'
target_dir = 'real_or_drawing/test_data'
extractor_dir = f'extractor_model.bin'
predictor_dir = f'predictor_model.bin'

if __name__ == "__main__":
    # data
    source_dataset = ImageFolder(source_dir, transform=source_transform)
    target_dataset = ImageFolder(target_dir, transform=target_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
    print("size of source_dataset is " + str(source_dataset.__len__()))
    print("size of target_dataset is " + str(target_dataset.__len__()))

    # model
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier().cuda()

    # loss
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # optimizer
    opt_F = optim.Adam(feature_extractor.parameters())
    opt_C = optim.Adam(label_predictor.parameters())
    opt_D = optim.Adam(domain_classifier.parameters())

    # hyper parameters:
    n_epoch = 200
    train(feature_extractor, label_predictor, domain_classifier, class_criterion, domain_criterion, opt_F, opt_C, opt_D,
          source_dataloader, target_dataloader, extractor_dir, predictor_dir, lamb=0.1, n_epoch=n_epoch)

    # inference(test_dataloader, extractor_dir, predictor_dir)
