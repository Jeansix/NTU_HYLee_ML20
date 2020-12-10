import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from w2v import train_word2vec
from preprocess import *
from data import *
from utils import *
from train import *
from test import *
from model import *
data_prefix = './data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_with_label = os.path.join(data_prefix, 'training_label.txt')
train_no_label = os.path.join(data_prefix, 'training_nolabel.txt')
testing_data = os.path.join(data_prefix, 'testing_data.txt')
w2v_path = os.path.join(data_prefix, 'w2vall.model')

sen_len = 30
fix_embedding = True  # fix embedding during training
batch_size = 128
epoch = 5
lr = 0.001
model_dir = data_prefix  # model directory for checkpoint model

# READ DATA
print('loading training data...')
train_x, train_y = load_training_data(train_with_label)
train_nolabel_x = load_training_data(train_no_label, hasLabel=False)
print('loading testing data...')
test_x = load_testing_data(testing_data)
#model = train_word2vec(train_x + train_nolabel_x + test_x)
#model.save(w2v_path)

# TRAIN PART
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(train_y)
# split data
X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]
# encapsulate data
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)  # num_workers indicates the number of threads

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0)
# prepare model
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=250, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)
training(model, batch_size, epoch, lr, model_dir, train_loader, val_loader, device)
# TEST PART
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
outputs = testing(model, batch_size, test_loader, device)

tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
print("save csv ...")
tmp.to_csv(os.path.join(data_prefix, 'predict.csv'), index=False)
print("Finish Predicting")
