import requests
from bs4 import BeautifulSoup
from retrying import retry
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from zhon.hanzi import punctuation
import re


@retry(stop_max_attempt_number=10)
def get_html(url):
    try:
        html = requests.get(url, timeout=10)
        html.encoding = 'utf-8'
        html_code = html.text
        soup = BeautifulSoup(html_code, 'html.parser')

        return soup
    except:
        print("get %s fail"%url)

        return None


def get_one_page_list(url, not_add_csv=True):
    soup = get_html(url)
    if soup != None:
        cat_block = soup.find('div', 'cat_block').find_all('li', 'item_li')
        content_url = []
        for item in cat_block:
            part_url = item.find('a')['href']
            full_url = url.split('?')[0] + part_url
            content_url.append(full_url)
        res = pd.DataFrame(content_url, columns=['url'])
        with open('./english_novels.csv', mode='a', encoding='utf-8', errors='ignore', newline='') as f:         
            res.to_csv(f, index=False, header=not_add_csv)
        print(url + ", success")
        return content_url
    else:
        print("get %s fail"%url)
        
        return None



def get_all_lists(url):
    content_url = get_one_page_list(url, not_add_csv=True)
    with ThreadPoolExecutor(200) as t:
        for page_number in range(189, 186, -1):
            t.submit(get_one_page_list, url[:-3] + str(page_number), False)
    print("\ndone!")


def get_one_page_content(url):
    soup = get_html(url)
    content = soup.find('div', 'HJHuaCi').text
    content2 = re.sub("[{}]+".format(punctuation), "", content)
    content3 = re.sub("[\u4e00-\u9fa5]", "", content2).strip()

    return content3


def get_all_content(url):
    pass


if __name__ == "__main__":
    last_url = "http://m.enread.com/index.php?mid=2&catid=5&page=190"
    # get_all_lists(last_url)