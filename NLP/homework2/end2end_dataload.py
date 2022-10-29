from data_preprocess import load_data
import random
from tqdm import tqdm
import torch


def seq_data_iter_random(corpus, step, batch_size):
    init_index = random.randint(0, step - 1)
    corpus = corpus[init_index:]
    num_sequence = (len(corpus) - 1) // step
    num_batch = num_sequence // batch_size
    sequence_index = [index for index in range(0, num_sequence * step, step)]
    random.shuffle(sequence_index)
    batch_index = [index for index in range(0, num_batch * batch_size, batch_size)]
    # random.shuffle(batch_index)
    
    generate_data = lambda idx: corpus[idx : idx + step]
    for i in batch_index:
        x = [generate_data(idx) for idx in sequence_index[i : i + batch_size]]
        y = [generate_data(idx + 1) for idx in sequence_index[i : i + batch_size]]
        yield torch.tensor(x), torch.tensor(y)


def seq_data_iter_sequential(corpus, step, batch_size):
    offset = random.randint(0, step)
    num_tokens = (len(corpus) - offset - 1) // batch_size * batch_size
    xs = torch.tensor(corpus[offset : offset + num_tokens])
    ys = torch.tensor(corpus[offset + 1 : offset + num_tokens + 1])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batch = xs.shape[1] // step
    for i in range(0, num_batch * step, step):
        x = xs[:, i : i + step]
        y = ys[:, i : i + step]
        yield x, y
        
        
def seq_data_iter_sequential_FNN(corpus, step, batch_size):
    offset = random.randint(0, step)
    num_tokens = (len(corpus) - offset - 1) // batch_size * batch_size
    xs = torch.tensor(corpus[offset : offset + num_tokens])
    ys = torch.tensor(corpus[offset + 1 : offset + num_tokens + 1])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batch = xs.shape[1] // step
    for i in range(0, num_batch * step, step):
        x = xs[:, i : i + step]
        y = ys[:, i + step - 1]
        yield x, y


class SeqDataLoader:
    def __init__(self, batch_size, step, use_random_sample, use_FNNML):
        self.step = step
        self.batch_size = batch_size
        if use_random_sample:
            self.sample_fn = seq_data_iter_random
        elif use_FNNML:
            self.sample_fn = seq_data_iter_sequential_FNN
        else:
            self.sample_fn = seq_data_iter_sequential
    
    
class SeqDataLoader_train(SeqDataLoader):
    def __init__(self, batch_size, step, use_random_sample, use_FNNML, train_corpus, train_vocab):
        super().__init__(batch_size, step, use_random_sample, use_FNNML)
        self.corpus = train_corpus
        self.vocab = train_vocab
    
    def __iter__(self):
        return self.sample_fn(self.corpus, self.step, self.batch_size)
    
    
class SeqDataLoader_val(SeqDataLoader):
    def __init__(self, batch_size, step, use_random_sample, use_FNNML, val_corpus, val_vocab):
        super().__init__(batch_size, step, use_random_sample, use_FNNML)
        self.corpus = val_corpus
        self.vocab = val_vocab
    
    def __iter__(self):
        return self.sample_fn(self.corpus, self.step, self.batch_size)


class SeqDataLoader_test(SeqDataLoader):
    def __init__(self, batch_size, step, use_random_sample, use_FNNML, test_corpus, test_vocab):
        super().__init__(batch_size, step, use_random_sample, use_FNNML)
        self.corpus = test_corpus
        self.vocab = test_vocab
    
    def __iter__(self):
        return self.sample_fn(self.corpus, self.step, self.batch_size)
    

def load_data_iter(batch_size, step, use_random_sample=False, use_FNNML=True):
    vocab, corpus,\
    train_corpus, train_vocab,\
    val_vocab, val_corpus,\
    test_corpus, test_vocab = load_data()
    train_data_iter = SeqDataLoader_train(batch_size, step, use_random_sample, use_FNNML, train_corpus, train_vocab)
    val_data_iter = SeqDataLoader_train(batch_size, step, use_random_sample, use_FNNML, val_corpus, val_vocab)
    test_data_iter = SeqDataLoader_train(batch_size, step, use_random_sample, use_FNNML, test_corpus, test_vocab)
    
    return train_data_iter, val_data_iter, test_data_iter, vocab, corpus
    

if __name__ == "__main__":
    train_data_iter, val_data_iter, test_data_iter, vocab, corpus = load_data_iter(batch_size=128, step=8)
    n = 0
    for x, y in tqdm(train_data_iter):
        n += 1
        if n == 4: break
        else:
            print(x, x.shape)
            print(y, y.shape)