from data_preprocess import load_data
import random
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
            print("use FNML sampling")
            self.sample_fn = seq_data_iter_sequential_FNN
        else:
            self.sample_fn = seq_data_iter_sequential
        self.train_corpus, self.train_vocab, self.test_corpus, self.test_vocab = load_data()
    
    
class SeqDataLoader_train(SeqDataLoader):
    def __init__(self):
        super().__init__()
    
    def __iter__(self):
        return self.sample_fn(self.train_corpus, self.step, self.batch_size)
    

class SeqDataLoader_test(SeqDataLoader):
    def __init__(self):
        super().__init__()
    
    def __iter__(self):
        return self.sample_fn(self.test_corpus, self.step, self.batch_size)


def load_data_iter(batch_size, step, use_random_sample=False, use_FNNML=False):
    train_data_iter = SeqDataLoader_train(batch_size, step, use_random_sample, use_FNNML)
    test_data_iter = SeqDataLoader_train(batch_size, step, use_random_sample, use_FNNML)
    
    return train_data_iter, test_data_iter