import collections
import re
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc


def load_book_tokenize():
    def filter_chinese(x):
        punc = cn_punc + en_punc
        tmp = re.sub("[{}]+".format(punc), "", x) # remove punctutation
        remove_white_spaces = re.sub(r'[\r|\n|\t]', '', tmp) # remove white space
        # pure_words = re.sub(r'[0-9]+', '', remove_white_spaces) # remove numbers
        pure_words = re.sub(r'[^\u4e00-\u9fa5_^0-9]', '', remove_white_spaces) # filter out other characters except chinese
        
        return pure_words
    
    with open('./data.txt', 'r') as f:
        text = f.readlines()
    lines = list(map(filter_chinese, text))
    all_tokens = [c for line in lines for c in line]
            
    return lines, len(all_tokens), all_tokens
    

def divide_dataset(lines, corpus_size, train_size=0.7, val_size=0.2):
    train_size = int(train_size * corpus_size)
    val_size = int(val_size * corpus_size)
    
    # divide train dataset, validation dataset and test dataset
    train_tokens, val_tokens, test_tokens = [], [], []
    n = 0
    for line in lines:
        tokens = [c for c in line]
        n += len(tokens)
        if n < train_size:
            train_tokens.append(tokens)
        elif n >= train_size and n < train_size + val_size:
            val_tokens.append(tokens)
        else:
            test_tokens.append(tokens)
    
    return train_tokens, val_tokens, test_tokens


def count_corpus(lines):
    flattened_tokens = [c for line in lines for c in line]
    
    return collections.Counter(flattened_tokens)


class Vocab:
    def __init__(self, min_freq=10, reserved_tokens=None, tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        freq = count_corpus(tokens)
        self.freq_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        self.unk = 0
        res = ['unk'] + reserved_tokens
        res += [token for token, freq in self.freq_tokens if freq > min_freq and token not in res]

        self.index_to_token, self.token_to_index = [], {}
        for i, token in enumerate(res):
            self.index_to_token.append(token)
            self.token_to_index[token] = len(self.index_to_token) - 1

    def __len__(self):
        return len(self.index_to_token)
    
    def __getitem__(self, token):
        if not isinstance(token, (list, tuple)):
            return self.token_to_index.get(token, self.unk)
        return [self.__getitem__(item) for item in token]

    def to_tokens(self, index):
        if not isinstance(index, (list, tuple)):
            return self.index_to_token[index]
        return [self.to_tokens(item) for item in index]


def test():
    lines, corpus_size, all_tokens = load_book_tokenize()
    vocab_word = Vocab(tokens=all_tokens)
    print(len(vocab_word))
    print(vocab_word[('的', '我', '新', 'like', '陈', '难', '赢', '中', '对', '1', '2', '5', 'e', ',', '.', '。', '\n')])
    print(vocab_word.to_tokens([1, 2, 3, 100, 1000, 2000, 3000, 4000]))


def load_data():
    def get_vocab_corpus(tokens):
        vocab = Vocab(tokens=tokens)
        corpus = [vocab[token] for line in tokens for token in line]

        return vocab, corpus
    
    lines, corpus_size, all_tokens = load_book_tokenize()
    vocab, corpus = get_vocab_corpus(all_tokens)
    train_tokens, val_tokens, test_tokens = divide_dataset(lines, corpus_size)
    train_vocab, train_corpus = get_vocab_corpus(train_tokens)
    val_vocab, val_corpus = get_vocab_corpus(val_tokens)
    test_vocab, test_corpus = get_vocab_corpus(test_tokens)

    return vocab, corpus,\
           train_corpus, train_vocab,\
           val_vocab, val_corpus,\
           test_corpus, test_vocab
           

if __name__ == "__main__":
    test()
    # vocab, corpus,\
    # train_corpus, train_vocab,\
    # val_vocab, val_corpus,\
    # test_corpus, test_vocab = load_data()
    # print(len(vocab), len(corpus), len(train_vocab), len(train_corpus), len(val_vocab), len(val_corpus), len(test_vocab), len(test_corpus))