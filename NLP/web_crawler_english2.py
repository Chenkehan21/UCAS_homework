import requests
from bs4 import BeautifulSoup
from retrying import retry
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
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


def get_top100_30_days(url, not_add_csv=True):
    soup = get_html(url)
    top100_30_days = soup.find_all('ol')[4].find_all('a')
    book_titles, content_urls = [], []
    for item in top100_30_days:
        book_title = item.text
        book_url = item['href']
        num = book_url.split('/')[-1]
        content_url = f"https://www.gutenberg.org/files/{num}/{num}-h/{num}-h.htm"
        book_titles.append(book_title)
        content_urls.append(content_url)
    title_content_urls = dict(zip(book_titles, content_urls))
    title_content_urls_pd = pd.DataFrame(list(title_content_urls.items()), columns=['title', 'url'])
    with open('./english_title_content_url.csv', mode='a', encoding='utf_8', errors='ignore', newline='') as f:         
        title_content_urls_pd.to_csv(f, index=False, header=not_add_csv)

    print(url + " success!")

    return title_content_urls_pd


def get_all_top100(url, not_add_csv=True):
    soup = get_html(url)
    tmp = soup.find_all('ol')
    all_top100 = []
    for item in [tmp[i] for i in (0, 2, 4)]:
        all_top100 += item.find_all('a')
    book_titles, content_urls = [], []
    for item in all_top100:
        book_title = item.text.split('(')[0].strip()
        book_url = item['href']
        num = book_url.split('/')[-1]
        content_url = f"https://www.gutenberg.org/files/{num}/{num}-h/{num}-h.htm"
        book_titles.append(book_title)
        content_urls.append(content_url)
    title_content_urls = {"title": book_titles, "url": content_urls}
    title_content_urls_pd = pd.DataFrame(title_content_urls)
    title_content_urls_pd = title_content_urls_pd.drop_duplicates('title', keep='first')
    with open('./english_title_content_url2.csv', mode='a', encoding='utf_8', errors='ignore', newline='') as f:         
        title_content_urls_pd.to_csv(f, index=False, header=not_add_csv)

    print(url + " success!")

    return title_content_urls_pd



def get_content(url, book_title):
    soup = get_html(url)
    if soup == None:
        return None
    else:
        contents = soup.find_all('p')
        with open(save_path + '%s.txt'%book_title, 'a', encoding='utf-8') as f:
            for content in contents:
                raw_text = content.text
                text = " ".join(re.split("\s+", raw_text, flags=re.UNICODE)).strip() + '\r\n'
                f.write(text)
            print("%s download success"%book_title)

        return content


def get_all_content():
    with open('./english_title_content_url2.csv', 'r', encoding='utf-8', errors='ignore') as f:
        all_title_urls = pd.read_csv(f)
    with ThreadPoolExecutor(100) as t:
        for _, value in all_title_urls.iterrows():
            book_title, book_url = value['title'], value['url']
            t.submit(get_content, book_url, book_title)
    print("\nFINISH!")


if __name__ == "__main__":
    root_url = "https://www.gutenberg.org"
    book_title_url = "https://www.gutenberg.org/browse/scores/top#books-last30"
    top_url = "https://www.gutenberg.org/browse/scores/top"
    save_path = "/mnt/d/UCAS/web_crawler/english_books2/"
    # get_all_top100(top_url)
    get_all_content()