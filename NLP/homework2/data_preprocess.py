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
        pure_words = re.sub(r'[^\u4e00-\u9fa5]', '', remove_white_spaces) # filter out other characters except chinese
        
        return pure_words
    
    with open('./data.txt', 'r') as f:
        text = f.readlines()
        # lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in text]
    lines = list(map(filter_chinese, text))
    
    def seperate_dataset(lines):
        tokens = []
        for line in lines:
            tokens.append([c for c in line])
        all_tokens = [c for line in lines for c in line]
            
        return tokens, all_tokens
    
    train_num = int(len(lines) * 0.7)
    train_lines = lines[:train_num]
    test_lines = lines[train_num: ]
    
    tokens_train, all_tokens_train = seperate_dataset(train_lines)
    tokens_test, all_tokens_test = seperate_dataset(test_lines)
    
    return tokens_train, all_tokens_train, tokens_test, all_tokens_test


class Vocab:
    def __init__(self, min_freq=10, reserved_tokens=None, tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        freq = collections.Counter(tokens)
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
    tokens, all_tokens = load_book_tokenize()
    vocab_word = Vocab(tokens=all_tokens)
    print(len(vocab_word))
    print(vocab_word[('的', '我', '新', 'like', '陈', '难', '赢', '中', '对')])
    print(vocab_word.to_tokens([1, 2, 3, 100, 1000, 2000, 3000, 4000, 4580-1]))


def load_data():
    tokens_train, all_tokens_train, tokens_test, all_tokens_test = load_book_tokenize()
    train_vocab = Vocab(tokens=all_tokens_train)
    train_corpus = [train_vocab[token] for line in tokens_train for token in line]
    
    test_vocab = Vocab(tokens=all_tokens_test)
    test_corpus = [test_vocab[token] for line in tokens_test for token in line]

    return train_corpus, train_vocab, test_corpus, test_vocab