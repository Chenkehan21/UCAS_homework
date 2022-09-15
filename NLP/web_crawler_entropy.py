import requests
from bs4 import BeautifulSoup


def get_html(url):
    try:
        html = requests.get(url, timeout=3)
        html_code = html.text
        soup = BeautifulSoup(html_code, 'html.parser')
        soup.find_all()

        return soup
    except:
        print("fail")


def get_title_url_list(url):
    soup = get_html(url)
    div_list = soup.find_all(name='div', attrs="fmKVwHzYQ")
    title_list, url_list = [], []
    for item in div_list:
        res = item.find('a')
        title_list.append(res.string)
        url_list.append(res['href'])
    title_url = dict(zip(title_list, url_list))

    return title_url


def get_book_section_list(url):
    pass



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
    root_url = ""
    first_url = ""
    save_path = ""

    save_txt(root_url, first_url, save_path)