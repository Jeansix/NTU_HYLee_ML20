import os
import json
import torch


class Preprocess:
    def __init__(self, root, embed_dim, max_seq_len):
        """
        :param root: 项目根目录
        :param embed_dim: 词嵌入维度
        :param max_seq_len: 固定每个sequence的长度
        """
        self.root = root
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')
        self.cn_embedding = None
        self.en_embedding = None

    def get_dictionary(self, language):
        """
        加载中文/英文字典
        :param language:
        :return:
        """
        word2int_dir = os.path.join(self.root, f'word2int_{language}.json')
        int2word_dir = os.path.join(self.root, f'int2word_{language}.json')
        with open(word2int_dir, 'r', encoding="utf-8") as f:
            word2int = json.load(f)
        with open(int2word_dir, 'r', encoding="utf-8") as f:
            int2word = json.load(f)
        return word2int, int2word

    def pad_seq(self, seq):
        if len(seq) < self.max_seq_len:
            gap = self.max_seq_len - len(seq)
            for _ in range(gap):
                seq.append('<PAD>')
        else:
            seq = seq[:self.max_seq_len]
        assert len(seq) == self.max_seq_len
        return seq

    def seq2idx(self, sentences_en, sentences_cn):
        cn_seq_idxs = []
        en_seq_idxs = []
        for seq in sentences_cn:
            seq = self.pad_seq(seq)
            seq_idx = [self.word2int_cn[x] if x in self.word2int_cn else self.word2int_cn['<UNK>'] for x in seq]
            cn_seq_idxs.append(seq_idx)
        for seq in sentences_en:
            seq = self.pad_seq(seq)
            seq_idx = [self.word2int_en[x] if x in self.word2int_en else self.word2int_en['<UNK>'] for x in seq]
            en_seq_idxs.append(seq_idx)
        return torch.tensor(cn_seq_idxs), torch.tensor(en_seq_idxs)

    def init_embedding(self):
        cn_length = len(self.word2int_cn)
        en_length = len(self.word2int_en)
        cn_embedding = torch.empty((cn_length, self.embed_dim))
        en_embedding = torch.empty((en_length, self.embed_dim))
        self.cn_embedding = torch.nn.init.xavier_normal_(cn_embedding, gain=1)
        self.en_embedding = torch.nn.init.xavier_normal_(en_embedding, gain=1)


if __name__ == "__main__":
    process = Preprocess('./cmn-eng', 200, 20)
    # process.init_embedding()
    print(len(process.word2int_en))
    print(len(process.word2int_cn))
    print(len(process.int2word_en))
    print(len(process.int2word_cn))
