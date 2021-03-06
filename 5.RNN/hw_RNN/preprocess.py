from torch import nn
from gensim.models import Word2Vec
import torch


class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # make embedding for a new word
        vec = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vec)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vec], 0)

    def make_embedding(self, load=True):
        if load:
            print("loading word to vec model...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # make dictionary of word2idx:word=>idx
        # make list of idx2word:idx=>word
        # make list of word2vector:idx=>embedding
        for i, word in enumerate(self.embedding.wv.vocab):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words:{}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            # print('sentence count #{}'.format(i+1),end='\r')
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx['<UNK>'])
            sentence_idx = self.pad_sequence(sentence_idx)  # to a fixed length
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)
