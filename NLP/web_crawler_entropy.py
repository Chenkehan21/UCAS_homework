import requests
from bs4 import BeautifulSoup
from retrying import retry
import pandas as pd
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp


@retry(stop_max_attempt_number=10)
def get_html(url):
    try:
        headers = {
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/105.0.0.0 Mobile Safari/537.36 Edg/105.0.1343.33",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,en-GB;q=0.6",
        }
        html = requests.get(url, headers=headers,timeout=10)
        html.encoding = 'utf-8'
        html_code = html.text
        soup = BeautifulSoup(html_code, 'html.parser')

        return soup
    except:
        print("get %s fail"%url)

        return None
    

# @retry(stop_max_attempt_number=10)
def get_one_page_title_url_list(url, not_add_csv=True):
    soup = get_html(url)
    punctuation_check = cn_punc + en_punc
    if soup != None:
        div_list = soup.find_all(name='div', attrs="fmKVwHzYQ")
        title_list, url_list = [], []
        for item in div_list:
            res = item.find('a')
            title = res.string
            book_url = res['href']
            for i in punctuation_check:
                title = title.replace(i, '')
            if len(title) > 0 and len(book_url) > 0:
                title_list.append(title)
                url_list.append(root_url + book_url[1:])
        title_url = dict(zip(title_list, url_list))
        title_url_pd = pd.DataFrame(list(title_url.items()), columns=["title", "url"])
        with open('./title_url3.csv', mode='a', encoding='utf_8_sig', errors='ignore', newline='') as f:         
            title_url_pd.to_csv(f, index=False, header=not_add_csv)
        print(url + " success!")
        
        return title_url
    else:
        return {}


def get_all_title_url_list(url):
    title_url = get_one_page_title_url_list(url, not_add_csv=True)
    with ThreadPoolExecutor(50) as t:
        for page_number in range(2, 825):
            t.submit(get_one_page_title_url_list, url + str(page_number), False)
    print("\ndone!")
    

# def get_all_title_url_list(root_url):
#     page_number = 1
#     page_url = root_url
#     all_title_url = {}
#     new_page_title_url = get_one_page_title_url_list(page_url)
#     if new_page_title_url != {}:
#         print("successfully get page %d"%page_number)
#     all_title_url.update(new_page_title_url)
#     while True:
#         last_page_title_url = new_page_title_url
#         page_number += 1
#         page_url = root_url + str(page_number) + '/'
#         new_page_title_url = get_one_page_title_url_list(page_url)
#         if new_page_title_url != {}:
#             print("successfully get page %d"%page_number)
#         '''
#         Judge whether reach the final page.
#         method: if reach the final + 1 page, we will get the same thing. So we can 
#         check whether new page title urls are the same with last page title urls
#         '''
#         if new_page_title_url != {} and last_page_title_url != {} \
#             and all(x in last_page_title_url.keys() for x in new_page_title_url.keys()):
#             print("reach final page")
#             break
#         all_title_url.update(new_page_title_url)

#         if page_number % 10 == 0:
#             with open('./all_title_list_%d.pkl'%page_number, "wb") as f:
#                 pickle.dump(all_title_url, f)

#     return all_title_url


# def get_book_content_list():
#     pass


def get_content_url(title_url):
    soup = get_html(title_url)
    if soup == None:
        return None
    else:
        read_url = soup.find(name='li', attrs='eUWPgsxrR').find('a')['href']
        return read_url


#<a href="javascript:void(0);">没有了</a>
def get_one_page_content(url):
    content_soup = get_html(url)
    if content_soup == None:
        print("get content fail")
        return None, None
    else:
        content = content_soup.find(name='div', attrs='aQHuOqEcj').text
        return content, content_soup


def get_nextpage(soup):
    next_url = soup.find('li', 'wEamlypTs').find('a')['href']
    if 'javascript:void(0)' in next_url:
        return False
    else:
        return next_url


def get_one_book(book_title, url):
    num = 0
    read_url = get_content_url(url)
    if read_url != None:
        with open(save_path + '%s.txt'%book_title, 'a', encoding='utf-8') as f:
            while read_url:
                content_url = root_url + read_url[1:]
                content, content_soup = get_one_page_content(content_url)
                if content != None:
                    f.write(content)
                    print("%s page %d success"%(book_title, num))
                else:
                    print("%s page %d fail"%(book_title, num))
                if content_soup != None:
                    read_url = get_nextpage(content_soup)
                    num += 1
                else:
                    continue
        print("%s finish!"%book_title)
    else:
        print("get %s fail"%book_title)


# use multithread again
def get_all_books():
    with open('./title_url2.csv', 'r', encoding='utf-8', errors='ignore') as f:
        all_title_urls = pd.read_csv(f)
    with ThreadPoolExecutor(500) as t:
        for _, value in all_title_urls[101:601].iterrows():
            book_title, book_url = value['title'], value['url']
            t.submit(get_one_book, book_title, book_url)
    print("\nAll books downloaded!")


if __name__ == "__main__":
    source_url = "https://m.moyanxsw.com/quanben/sort/"
    root_url = "https://m.moyanxsw.com/"
    save_path = "/mnt/d/UCAS/web_crawler/chinese_novels/"
    # get_all_title_url_list(source_url)
    # get_one_book("灵兽养殖笔记", "https://m.moyanxsw.com/lingshouyangshibiji/")
    get_all_books()