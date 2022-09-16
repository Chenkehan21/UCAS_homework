import requests
from bs4 import BeautifulSoup
from retrying import retry
import pickle
from threading import Thread, current_thread


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
def get_one_page_title_url_list(url):
    soup = get_html(url)
    if soup != None:
        div_list = soup.find_all(name='div', attrs="fmKVwHzYQ")
        title_list, url_list = [], []
        for item in div_list:
            res = item.find('a')
            title_list.append(res.string)
            url_list.append(res['href'])
        title_url = dict(zip(title_list, url_list))

        return title_url
    else:
        return {}


def get_all_title_url_list(root_url):
    page_number = 1
    page_url = root_url
    all_title_url = {}
    new_page_title_url = get_one_page_title_url_list(page_url)
    if new_page_title_url != {}:
        print("successfully get page %d"%page_number)
    all_title_url.update(new_page_title_url)
    while True:
        last_page_title_url = new_page_title_url
        page_number += 1
        page_url = root_url + str(page_number) + '/'
        new_page_title_url = get_one_page_title_url_list(page_url)
        if new_page_title_url != {}:
            print("successfully get page %d"%page_number)
        '''
        Judge whether reach the final page.
        method: if reach the final + 1 page, we will get the same thing. So we can 
        check whether new page title urls are the same with last page title urls
        '''
        if new_page_title_url != {} and last_page_title_url != {} \
            and all(x in last_page_title_url.keys() for x in new_page_title_url.keys()):
            print("reach final page")
            break
        all_title_url.update(new_page_title_url)

        if page_number % 10 == 0:
            with open('./all_title_list_%d.pkl'%page_number, "wb") as f:
                pickle.dump(all_title_url, f)

    return all_title_url


def get_book_content_list(url):
    title_url = get_title_url_list(url)
    for book_title, url in title_url.items():
        book_url = first_url + url
        soup = get_html(book_url)
        content_url = soup.find(name='div', attrs='lRSQphVsn').find_all('li')



# def get_title(url):
#     soup = get_html(url)
#     title = soup.find('div', class_="container").find('h1').string

#     return title
    

def get_one_page_content(url):
    pass


def get_nextpage(url):
    pass


def save_txt(root_url, first_url, save_path):
    pass


if __name__ == "__main__":
    root_url = "https://m.moyanxsw.com/quanben/sort/"
    first_url = "https://m.moyanxsw.com/"
    save_path = ""

    save_txt(root_url, first_url, save_path)