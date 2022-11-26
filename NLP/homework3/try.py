def BEP(sentence, text, spliter='/'):
    res = ""
    sentence = sentence.split('/')[:-1]
    pairs = [sentence[i] + sentence[i + 1] for i in range(len(sentence) - 1)]
    pairs_query = [sentence[i : i + 2] for i in range(len(sentence) - 1)]
    print(pairs)
    pair_num = dict.fromkeys(pairs, 0)
    print(pair_num)
    for key in pair_num.keys():
        pair_num[key] = text.count(key)
    pair_num = sorted(pair_num.items(), key=lambda x: x[1], reverse=True)
    print(pair_num)
    sub_word = pair_num[0][0]
    sub_word_freq = [item[1] for item in pair_num]
    index = pairs.index(sub_word)
    
    if sub_word_freq[0] == 0:
        return sentence
    else:
        for i, item in enumerate(pairs_query):
            if i == index:
                res += item[0] + item[1] + spliter
            if i - 1 == index:
                continue
            else:
                res += item[0] + spliter
                
        return BEP(res, text)
    
    
def test_BEP(sentence, CUT_func=FMM):
    cn_dict, test_corpus = load_dict()
    source_corpus, label_corpus = get_corpus(test_corpus)
    text = []
    for line in label_corpus:
        text += line.split('/')[:-1]
    print(text[:10])
    cut = CUT_func(sentence, cn_dict)
    BEP_sentence = BEP(cut, text)
    print("raw sentence: ", sentence)
    print("cut sentence: ", cut)
    print("BEP sentence: ", BEP_sentence)
    
    return BEP_sentence


def FMM(sentence, cn_dict, spliter='/'):
    # cn_dict = list(set(cn_dict))
    # cn_dict.sort(key=lambda x: len(x)) # sort according to length
    # cn_dict.reverse()
    print(cn_dict[0])
    m = len(cn_dict[0])
    m = 7
    res = ""
    length = len(sentence)
    p = 0

    while True:
        n = length - p
        if n == 0:
            break
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