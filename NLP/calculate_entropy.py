from inspect import getfile
import numpy as np
from collections import Counter
import os
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc
import re
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt


all_content = ''


def get_file(path, file_name, en_cn='cn'):
    file_path = os.path.join(path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    
    # clean file
    res = ''
    for content in contents:
        tmp = re.sub("[{}]+".format(punc), "", content) # remove punctutation
        remove_white_spaces = re.sub(r'[\r|\n|\t]', '', tmp) # remove white space
        pure_words = re.sub(r'[0-9]+', '', remove_white_spaces)
        if en_cn == 'en':
            lower_case = pure_words.lower()
            pure_words = re.sub(r'[^a-z^\s]', '', lower_case)
        elif en_cn == 'cn':
            pure_words = re.sub(r'[^\u4e00-\u9fa5]', '', pure_words)

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


def append_data(path, first_append, n = 2, en_cn='cn'):
    all_contents = ''
    if first_append:
        with ThreadPoolExecutor(n) as t:
            for file_name in all_file_names[0 : n]:
                # print(file_name)
                t.submit(get_file, path, file_name, en_cn)
                # content = get_file(path, file_name)
                # clean_content = data_clean(content.result)
                # all_contents += clean_content
    else:
        with ThreadPoolExecutor(n) as t:
            for file_name in all_file_names[n - 2 : n]:
                t.submit(get_file, path, file_name, en_cn)
                # print(file_name)
                # clean_content = data_clean(content)
                # all_contents += clean_content
    print("n: ", n)

    return all_contents


# 666666 chinese utf-8 letters is about 2MB 666666 * 3
# 2000000 english letters is 2MB
def calculate_cn_word_entropy(num_filesize=1, en_cn="cn"):
    if en_cn == "cn":
        letters_2mb = 666666
        words_per_book_avg = 743362
    elif en_cn == "en":
        letters_2mb = 2000000
        words_per_book_avg = 603448
    num_books = int(2 + num_filesize * letters_2mb / words_per_book_avg)
    # print("num_books: ", num_books)
    end_size = int(letters_2mb * num_filesize)
    append_data(root_path, first_append=True, n=num_books, en_cn=en_cn)
    while end_size - len(all_content) > 300 and num_books < len(all_file_names):
        num_books += 2
        append_data(root_path, first_append=False, n=num_books, en_cn=en_cn)
    content = all_content[:end_size]
    word_num_table = Counter(content)
    # print(word_num_table, len(word_num_table))
    total = sum(list(word_num_table.values()))
    word_probability = {k: v / total for k, v in word_num_table.items()}
    word_probability_list = sorted(word_probability.items(), key=lambda x: x[1], reverse=True)
    for i, value in enumerate(word_probability_list):
        if i <= 30:
            print(value)
    # print(word_probability, len(word_probability))
    p = np.array(list(word_probability.values()))
    log_p = np.log2(p)
    H = -np.dot(p, log_p.reshape((-1, 1))).item()
    print(H)

    return H


def calculate_entropy_different_file_size(num_filesize=20, en_cn='cn'):
    all_H = []
    for file_size in range(1, num_filesize):
        print(file_size)
        all_H.append(calculate_cn_word_entropy(file_size, en_cn=en_cn))
    print(all_H)

    return all_H


def plot_result(all_H, title):
    num_H = len(all_H)
    x = list(range(2, 2 * num_H + 1, 2))
    plt.plot(x, all_H)
    plt.grid()
    plt.title(title)
    plt.xlabel("file size(MB)")
    plt.ylabel("entropy")
    plt.show()


if __name__ == "__main__":
    # root_path = "/mnt/d/UCAS/web_crawler/english_books2/"
    root_path = "/mnt/d/UCAS/web_crawler/chinese_novels/"
    save_path = "/mnt/d/UCAS/web_crawler/"
    punc = cn_punc + en_punc
    all_file_names = os.listdir(root_path)
    # calculate_entropy_different_file_size(num_filesize=15)
    # calculate_cn_word_entropy(200)
    # get_file(root_path, all_file_names[0])
    # t = time.time()
    # append_data(root_path, True, 150)
    # t2 = time.time()
    # print(len(all_content), t2 - t)
    H = calculate_cn_word_entropy(num_filesize=30, en_cn="cn")
    # all_H = calculate_entropy_different_file_size(30, en_cn='cn')
    # plot_result(all_H, "English Entropy")