import torch

import utils
import tagger1


if __name__ == '__main__':
    pos_train_set, word2index, index2word, label2index, index2label = utils.parse_POS('./pos/train', window_size=2)
    # ner_train_set, words2index, index2words, _, _ = utils.parse_NER('./ner/train', window_size=2)
    # test_set = utils.parse_test_file('./pos/test', window_size=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr = 1e-3
    n_epochs = 10

    vocab_size = len(word2index.keys())
    embedd_size = 50
    num_words = 5

    hidden_dim = 128
    out_dim = len(label2index.keys())

    model = tagger1.Tagger1Model(vocab_size, embedd_size, num_words, hidden_dim, out_dim)

    tagger1.train_model(pos_train_set, model, n_epochs, lr, device, word2index, label2index)

