import torch.nn
from preprocess import *
from nltk.translate.bleu_score import sentence_bleu
from model import *


def load_data(path):
    """
    加载数据
    :param path: 数据存放的路径
    :return:en_sentences英文句子的list,首尾用<BOS>,<EOS>封装
    :return:cn_sentences中文句子的list,首尾用<BOS>,<EOS>封装
    """
    en_sentences = []
    cn_sentences = []
    total_len = 0
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            seq = line.strip().split('\t')
            total_len += len(seq[0].strip().split(' '))
            total_len += len(seq[1].strip().split(' '))
            en_sentences.append(['<BOS>'] + seq[0].strip().split(' ') + ['<EOS>'])
            cn_sentences.append(['<BOS>'] + seq[1].strip().split(' ') + ['<EOS>'])
    return en_sentences, cn_sentences


def save_model(model, path, step):
    """
    保存训练好的模型（的参数）
    :param model: 训练好的模型
    :param path: 模型保存路径
    :param step: 训练步数
    :return:
    """
    print("saving the best model:epoch_{}".format(step))
    torch.save(model.state_dict(), f'{path}/best_model.ckpt')
    return


def load_model(model, path):
    """
    加载训练好的模型（的参数）
    :param model: 定义的模型
    :param path: 模型所在路径
    :return: model:训练好的模型的参数
    """
    print(f'loading model from {path}')

    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    return model


def build_model(config, en_vocab_size, cn_vocab_size):
    """

    :param config: 参数配置
    :param en_vocab_size: 英文字典大小，是向量的维度
    :param cn_vocab_size: 中文字典大小，是向量的维度
    :return: model:定义的模型
    :return: optimizer:定义的优化器
    :return: loss_fn: 定义的损失函数
    """
    encoder = Encoder(en_vocab_size, config.embed_dim, config.hidden_dim, config.n_layers, config.dropout,
                      config.bidirectional)
    decoder = Decoder(cn_vocab_size, config.embed_dim, config.hidden_dim, config.n_layers, config.dropout,
                      config.is_attn)
    model = Seq2Seq(encoder, decoder, config.device)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    return model, optimizer, loss_fn


def idx2sentences(outputs, int2word):
    """
    将outputs(index形式）转换成句子(word形式）
    :param outputs: array，每个元素是一个index
    :param int2word: dict,将index转化为word
    :return: sentences:list,每个元素是用list表示的句子
    """
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word != '<EOS>':
                sentence.append(sentence)
            else:
                break
        sentences.append(sentence)
    return sentences


def compute_bleu(sentences, targets):
    """
    计算BLEU分数
    :param sentences: pred result
    :param targets: true result
    :return: score:bleu score of sentences
    """
    score = 0
    assert len(sentences) == len(targets)

    def cut_token(sentence):
        split_sent = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                split_sent.append(token)
            else:
                split_sent += [word for word in token]
        return split_sent

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += score_bleu([target], sentence, weights=(1, 0, 0, 0))
    return score


if __name__ == '__main__':
    train_dir = 'cmn-eng/training.txt'
    val_dir = 'cmn-eng/validation.txt'
    test_dir = 'cmn-eng/testing.txt'

    en_train, ch_train = load_data(train_dir)
    en_val, ch_val = load_data(val_dir)
    en_test, ch_test = load_data(test_dir)

    print("length of train:{}".format(len(en_train)))
    print("length of val:{}".format(len(en_val)))
    print("length of test:{}".format(len(en_test)))
    print(en_train[10], ch_train[10], en_val[0], ch_val[0], en_test[66], ch_test[66], sep='\n')
