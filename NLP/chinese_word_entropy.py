from inspect import getfile
import numpy as np
from collections import Counter
import os
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc
import re
from concurrent.futures import ThreadPoolExecutor
import time


all_content = ''


def get_file(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    time.sleep(2)
    # clean file
    res = ''
    for content in contents:
        tmp = re.sub("[{}]+".format(punc), "", content) # remove punctutation
        clean_content = re.sub(r'\s+', '', tmp) # remove white space
        pure_words = re.sub(r'[0-9]+', '', clean_content)
        res += pure_words
        # res += clean_content
    global all_content 
    all_content += res.strip()

    return res.strip()


# def data_clean(contents):
#     res = ''
#     for content in contents:
#         tmp = re.sub("[{}]+".format(punc), "", content) # remove punctutation
#         clean_content = re.sub(r'\s+', '', tmp)
#         res += clean_content

#     return res.strip()


def append_data(path, first_append, n = 2):
    all_contents = ''
    if first_append:
        with ThreadPoolExecutor(n) as t:
            for file_name in all_file_names[0 : n]:
                print(file_name)
                t.submit(get_file, path, file_name)
                # content = get_file(path, file_name)
                # clean_content = data_clean(content.result)
                # all_contents += clean_content
    else:
        with ThreadPoolExecutor(n) as t:
            for file_name in all_file_names[n - 2 : n]:
                t.submit(get_file, path, file_name)
                # print(file_name)
                # clean_content = data_clean(content)
                # all_contents += clean_content


    return all_contents


# 666666 chinese utf-8 letters is about 2MB 666666 * 3
# 2000000 english letters is 2MB
def calculate_cn_word_entropy(num_filesize=1, en_cn="cn"):
    if en_cn == "cn":
        letters_2mb = 666666
        words_per_book_avg = 743362
    elif en_cn == "en":
        letters_2mb = 2000000
        words_per_book_avg = 743362
    num_books = int(2 + num_filesize * letters_2mb / words_per_book_avg)
    end_size = letters_2mb * num_filesize
    append_data(root_path, first_append=True, n=num_books)
    while end_size - len(all_content) > 300:
        num_books += 2
        append_data(root_path, first_append=False, n=num_books)
    content = all_content[:end_size]
    word_num_table = Counter(content)
    # print(word_num_table)
    total = sum(list(word_num_table.values()))
    word_probability = {k: v / total for k, v in word_num_table.items()}
    p = np.array(list(word_probability.values()))
    log_p = np.log2(p)
    H = -np.dot(p, log_p.reshape((-1, 1))).item()
    print(H)

    return H


def calculate_entropy_different_file_size(num_filesize=20):
    all_H = []
    for file_size in range(1, num_filesize):
        print(file_size)
        all_H.append(calculate_cn_word_entropy(file_size))
    print(all_H)


if __name__ == "__main__":
    root_path = "D:\\corpus_en"
    save_path = "/mnt/d/UCAS/web_crawler/"
    punc = cn_punc + en_punc
    all_file_names = os.listdir(root_path)
    # calculate_entropy_different_file_size(num_filesize=15)
    # calculate_cn_word_entropy(200)
    # get_file(root_path, all_file_names[0])
    t = time.time()
    append_data(root_path, True, 15)
    t2 = time.time()
    print(all_content, len(all_content), t2 - t)