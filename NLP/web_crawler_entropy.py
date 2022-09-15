import requests
from bs4 import BeautifulSoup
from retrying import retry


@retry(stop_max_attempt_number=10)
def get_html(url):
    try:
        html = requests.get(url, timeout=3)
        html_code = html.text
        soup = BeautifulSoup(html_code, 'html.parser')

        return soup
    except:
        print("fail")


@retry(stop_max_attempt_number=10)
def get_one_page_title_url_list(url):
    soup = get_html(url)
    div_list = soup.find_all(name='div', attrs="fmKVwHzYQ")
    title_list, url_list = [], []
    for item in div_list:
        res = item.find('a')
        title_list.append(res.string)
        url_list.append(res['href'])
    title_url = dict(zip(title_list, url_list))

    return title_url


def get_all_title_url_list(root_url):
    page_number = 1
    page_url = root_url
    print(page_url)
    all_title_url = {}
    new_page_title_url = get_one_page_title_url_list(page_url)
    print("successfully get page %d"%page_number)
    all_title_url.update(new_page_title_url)
    while True:
        last_page_title_url = new_page_title_url
        page_number += 1
        page_url = root_url + str(page_number) + '/'
        new_page_title_url = get_one_page_title_url_list(page_url)
        print("successfully get page %d"%page_number)
        '''
        Judge whether reach the final page.
        method: if reach the final + 1 page, we will get the same thing. So we can 
        check whether new page title urls are the same with last page title urls
        '''
        if all(x in last_page_title_url.keys() for x in new_page_title_url.keys()):
            print("reach final page")
            break
        all_title_url.update(new_page_title_url)

    return all_title_url, last_page_title_url, new_page_title_url


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