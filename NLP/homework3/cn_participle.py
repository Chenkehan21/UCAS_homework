import re
import collections
import random
from tqdm import tqdm
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc
import matplotlib.pyplot as plt

random.seed(123)


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
    total_num = len(lines)
    random.shuffle(lines)
    n = int(0.7 * total_num)
    train_lines = lines[: n]
    test_lines = lines[n: ]
    
    return preprocess_data(train_lines), test_lines


def get_corpus(lines):
    def data_preprocess(line):
        line = re.sub(r'[a-zA-Z_\[_\]]', '', line) # remove part of speech notations
        line = line.replace(' ', '') # remove spaces in one line
        
        return line[20:] # drop data

    label_corpus = list(map(data_preprocess, lines))
    label_corpus = [x for x in label_corpus if x != '']
    source_corpus = [line.replace('/', '') for line in label_corpus]
    
    return source_corpus, label_corpus


def FMM(sentence, cn_dict, max_len=7, spliter='/'):
    res = ""
    start = 0
    non_dictionary_words = 0
    single_words = 0
    total_cuts = 0
    end = min(start + max_len, len(sentence))
    while start < end:
        word = sentence[start : end]
        if word in cn_dict.keys() or len(word) == 1: # using a dict is faster than using a list!
            res += word + spliter
            total_cuts += 1
            start = end
            end = min(start + max_len, len(sentence))
        else:
            end -= 1
            
        if word not in cn_dict:
            non_dictionary_words += 1
        if len(word) == 1:
            single_words += 1
    
    return res, non_dictionary_words, single_words, total_cuts


def BMM(sentence, cn_dict, max_len=7 ,spliter='/'):
    res = ""
    start = 0
    non_dictionary_words = 0
    single_words = 0
    total_cuts = 0
    end = min(start + max_len, len(sentence))
    
    sentence = sentence[::-1]
    while start < end:
        word = sentence[start : end]
        word = word[::-1]
        if word in cn_dict or len(word) == 1:
            res += spliter + word[::-1]
            total_cuts += 1
            start = end
            end = min(start + max_len, len(sentence))
        else:
            end -= 1
        if word not in cn_dict:
            non_dictionary_words += 1
        if len(word) == 1:
            single_words += 1
    
    return res[::-1], non_dictionary_words, single_words, total_cuts


def Bi_MM(sentence, cn_dict):
    '''
    use FMM and BMM, choose a better result according to maximum match rule:
    (1) less Non-dictionary words
    (2) less single words
    (3) less total words
    '''
    
    FMM_cut, FMM_ndw, FMM_sw, FMM_tc = FMM(sentence, cn_dict) # ndw: non-dictionary words
    BMM_cut, BMM_ndw, BMM_sw, BMM_tc = BMM(sentence, cn_dict) # sw: single_words, tc: total_cuts
    if (FMM_ndw <= BMM_ndw) + (FMM_sw <= BMM_sw) + (FMM_tc <= BMM_tc) >= 2:
        res = FMM_cut
        ndw = FMM_ndw
        sw = FMM_sw
        tc = FMM_tc
    else:
        res = BMM_cut
        ndw = BMM_ndw
        sw = BMM_sw
        tc = BMM_tc
        
    return res, ndw, sw, tc
    
        
def test(cn_dict, source_corpus, label_corpus, CUT_func=FMM, name="FMM", BEP_iteration=0):
    cut, tc = [], []
    for sentence in tqdm(source_corpus, ncols=80):
        c, n, s, t = CUT_func(sentence, cn_dict)
        cut.append(c)
        tc.append(t)

    '''
    how to check right cut?
    FMM_cut: A/BC/D/E/B/C
    label:   AB/CD/E/BC
    
    need 2 steps:
    (1) check whether the part in label
    (2) check whether they are at the same position
    '''
    correct, num_cut, num_label = 0, 0, 0
    for i, line in tqdm(enumerate(cut), ncols=80):
        label = label_corpus[i].split('/')[:-1] # after split the last one will be ''
        label_sentence = label_corpus[i].replace('/', '')
        line = line.split('/')[:-1]
        num_cut += len(line)
        num_label += len(label)
        pos = -1
        for item in line:
            pos += len(item)
            if item in label and label_sentence[pos - len(item) + 1: pos + 1] == item:
                correct += 1
    
    precision = correct / num_cut
    recall = correct / num_label
    F1 = (2 * precision * recall) / (precision + recall)
    print("Precision: %.3f\nRecall: %.3f\nF1: %.3f"%(precision, recall, F1))
    with open('./%s_BEP%d.txt'%(name, BEP_iteration), 'w') as f:
        f.write("Test on %d sentences.\nPrecision: %.3f\nRecall: %.3f\nF1: %.3f"%(len(source_corpus), precision, recall, F1))
        f.write("\ncut result:\n%s"%cut[1])
    
    return tc


def BEP(cn_dict, iteration_times=1000):
    punc = cn_punc + en_punc
    cn_dict = [x for x in cn_dict if x not in punc]
    
    print("\n===== start BEP =====\n")
    for _ in tqdm(range(iteration_times), ncols=80):
        pairs = [cn_dict[i] + cn_dict[i + 1] for i in range(len(cn_dict) - 1)]
        pairs_query = [cn_dict[i : i + 2] for i in range(len(cn_dict) - 1)]
        pairs_freq = collections.Counter(pairs)
        freq = dict(sorted(pairs_freq.items(), key=lambda x: x[1], reverse=True))
        sub_word = list(freq.keys())[0]
        index = pairs.index(sub_word)
        words = pairs_query[index]
        p = 0
        q = p + 1
        while q < len(cn_dict):
            if cn_dict[p] == words[0] and cn_dict[q] == words[1]:
                cn_dict[p] = sub_word
                del cn_dict[q]
            p += 1
            q = p + 1
    cn_dict = dict.fromkeys(cn_dict, 0)
    
    return cn_dict


def plot_historgram(data:list, name, iterations:list):
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle("Test BEP for %s"%name)
    
    ax = fig.add_subplot(2, 2, 1)
    plt.title("without BEP")
    plt.hist(data[0], bins=len(set(data[0])))
    plt.grid()
    plt.xlabel("sentence length")
    plt.ylabel("number")
    
    ax = fig.add_subplot(2, 2, 2)
    plt.title("with BEP iteration %d"%iterations[0])
    plt.hist(data[1], bins=len(set(data[1])))
    plt.grid()
    plt.xlabel("sentence length")
    plt.ylabel("number")
    
    ax = fig.add_subplot(2, 2, 3)
    plt.title("with BEP iteration %d"%iterations[1])
    plt.hist(data[2], bins=len(set(data[2])))
    plt.grid()
    plt.xlabel("sentence length")
    plt.ylabel("number")
    
    ax = fig.add_subplot(2, 2, 4)
    plt.title("with BEP iteration %d"%iterations[2])
    plt.hist(data[3], bins=len(set(data[3])))
    plt.grid()
    plt.xlabel("sentence length")
    plt.ylabel("number")
    
    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.5, wspace=0.5)
    plt.show()
    plt.savefig("./%s.png"%name)


def main():
    cn_dict, test_corpus = load_dict()
    source_corpus, label_corpus = get_corpus(test_corpus)
    '''
    find one element in dict will faster than in list!
    since list read elements by bias, while dict read elements by key(hash map)!
    '''
    cn_dict2 = dict.fromkeys(cn_dict, 0)
    FMM_tc = test(cn_dict2, source_corpus, label_corpus, CUT_func=FMM, name='FMM', BEP_iteration=0)
    BMM_tc = test(cn_dict2, source_corpus, label_corpus, CUT_func=BMM, name='BMM', BEP_iteration=0)
    Bi_MM_tc = test(cn_dict2, source_corpus, label_corpus, CUT_func=Bi_MM, name='Bi_MM', BEP_iteration=0)

    FMM_tcs = [FMM_tc]
    BMM_tcs = [BMM_tc]
    Bi_MM_tcs = [Bi_MM_tc]
    iteration_times = [100, 1000, 10000]
    for iteration in iteration_times:
        cn_dict3 = BEP(cn_dict, iteration)
        FMM_tc = test(cn_dict3, source_corpus, label_corpus, CUT_func=FMM, name='FMM', BEP_iteration=iteration)
        BMM_tc = test(cn_dict3, source_corpus, label_corpus, CUT_func=BMM, name='BMM', BEP_iteration=iteration)
        Bi_MM_tc = test(cn_dict3, source_corpus, label_corpus, CUT_func=Bi_MM, name='Bi_MM', BEP_iteration=iteration)
        FMM_tcs.append(FMM_tc)
        BMM_tcs.append(BMM_tc)
        Bi_MM_tcs.append(Bi_MM_tc)
    
    plot_historgram(FMM_tcs, "FMM", iteration_times)
    plot_historgram(BMM_tcs, "BMM", iteration_times)
    plot_historgram(Bi_MM_tcs, "Bi_MM", iteration_times)


if __name__ == "__main__":
    # main()
    cn_dict, test_corpus = load_dict()
    cn_dict = dict.fromkeys(cn_dict)
    sentence1 = "这部电影很好看，我很喜欢它！"
    sentence2 = "他是研究生物化学的一位科学家。"
    sentence3 = "乒乓球拍卖完了。"
    res1 = Bi_MM(sentence1, cn_dict)[0]
    res2 = Bi_MM(sentence2, cn_dict)[0]
    res3 = Bi_MM(sentence3, cn_dict)[0]
    print(res1)
    print(res2)
    print(res3)