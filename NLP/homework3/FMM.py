import re
import collections


def load_dict():
    def preprocess_data(lines):
        res = []
        for line in lines:
            data = line.split(' ')[1:]
            for item in data:
                if item != '':
                    item = item.split('/')[0]
                    if '[' in item:
                        item = item.split('[')[1]
                    res.append(item)
        
        return res
                
    with open("./ChineseCorpus.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
    lines = [re.sub(r'[\r|\n|\t]', '', line) for line in lines] # remove white space
        
    return preprocess_data(lines)


def check_dict(dict):
    freq = collections.Counter(dict)
    freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    print(len(freq))
    
    return freq


def get_corpus():
    def data_preprocess(line):
        line = re.sub(r'[\r|\n|\t]', '', line) # remove white space
        line = re.sub(r'[a-zA-Z_\[_\]]', '', line) # remove part of speech notations
        line = line.replace(' ', '') # remove spaces in one line
        
        return line[20:] # drop data
        
    with open("./ChineseCorpus.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
    label_corpus = list(map(data_preprocess, lines))
    
    return label_corpus

def FMM(sentence, cn_dict, spliter='/'):
    cn_dict = list(set(cn_dict))
    cn_dict.sort(key=lambda x: len(x)) # sort according to length
    cn_dict.reverse()
    m = len(cn_dict[0])
    res = ""
    length = len(sentence)
    p = 0

    while True:
        n = length - p
        if n == 1:
            res += sentence[p] + spliter
            break
        if n < m: m = n
        w = sentence[p : p + m]
        while True:
            if w in cn_dict:
                res += w + spliter
                p += len(w)
                break
            else:
                if len(w) > 1:
                    w = w[:-1]
                else:
                    res += w + spliter
                    p += len(w)
                    cn_dict.append(w)
                    break
                
    return res


if __name__ == "__main__":
    cn_dict = load_dict()
    sentence1 = "这部电影很好看，我很喜欢它！"
    sentence2 = "他是研究生物化学的一位科学家。"
    sentence3 = "乒乓球拍卖完了。"
    res1 = FMM(sentence1, cn_dict)
    res2 = FMM(sentence2, cn_dict)
    res3 = FMM(sentence3, cn_dict)
    print(res1)
    print(res2)
    print(res3)