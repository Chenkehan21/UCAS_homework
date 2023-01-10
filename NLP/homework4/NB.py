from collections import Counter


def load_data(path='./NLP_TC/traindata.txt'):
    with open(path, 'r') as f:
        text = f.readlines()
        
    print("trainset size: ", len(text))
    
    data = {}
    vocab = []
    for line in text:
        line = line.split('\t')
        label = line[0]
        if label not in data.keys():
            data[label] = []
        data[label].append(line)
        
        content = line[1].strip().split(' ')
        content.insert(0, label)
        vocab += content
    
    vocab_freq = Counter(vocab)
    vocab_freq = dict(sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True))
    
    for key in data.keys():
        print("label: %s, size: %d" % (key, len(data[key])))
    print("vocab size: ", len(vocab_freq.keys()))
        
    return data, vocab_freq


