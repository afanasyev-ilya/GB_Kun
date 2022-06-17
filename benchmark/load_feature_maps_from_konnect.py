import pickle
import requests
import argparse
import sys
import time
from progress.bar import IncrementalBar
from urllib.request import urlopen
from bs4 import BeautifulSoup

requests_get_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'}

def get_category_names():
    url = "http://konect.cc/categories/"
    # html = urlopen(url).read()
    # soup = BeautifulSoup(html, features="html.parser")
    html= requests.get(url, headers=requests_get_headers)
    soup = BeautifulSoup(html.text, features="html.parser")
    category_names = []

    for name in soup.find_all('a'):
        category_name = name.get('href')
        if (not category_name in category_names) and category_name[0] != '/' and category_name[-1] == '/':
            category_names.append(category_name)

    return category_names

def get_graph_names(category_names):
    bad_urls = ['../../', '../', '/', 'https://www.paypal.com/donate?hosted_button_id=Q9JY2FB3AFHR6', '/networks/flickr-groupmemberships/', '/networks/dbpedia-country/']
    graph_urls = []
    for category_name in category_names:
        category_url = "http://konect.cc/categories/" + category_name
        #        html = urlopen(category_url).read()
        #        soup = BeautifulSoup(html, features="html.parser")
        html= requests.get(category_url, headers=requests_get_headers)
        soup = BeautifulSoup(html.text, features="html.parser")
        for name in soup.find_all('a'):
            url = name.get('href')
            graph_url = category_url + url
            if (not graph_url in graph_urls) and (not url in bad_urls) and (url != "/categories/" + category_name):
                graph_urls.append(graph_url)
    return graph_urls

def get_name(soup):
    return soup.select('h1')[0].text.strip()

def get_tsv_link(soup):
    #        print(soup.find_all('h1'))
    for name in soup.find_all('a'):
        link = name.get('href')
        if link == None:
            continue
        if 'tsv' in link:
            return link
    return None

def extract_number(line):
    digits_list = [int(s) for s in line if s.isdigit()]
    return int(''.join(map(str, digits_list)))

def extract_float(line):
    if line is None:
        return -1.0
    number_section = line.split('=')[1]
    required_float = float(number_section.split()[0].replace(",", ""))
    return required_float


def find_info_on_page(text, pattern):
    for line in text.splitlines():
        if pattern in line:
            return line
    return None

def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

def extract_category(line):
    return remove_prefix(line, "Category")

def get_graph_info(graph_url):
    #html = urlopen(graph_url).read()
    html= requests.get(graph_url, headers=requests_get_headers)
    soup = BeautifulSoup(html.text, features="html.parser")

    #    print(html)

    if "404" in str(html):
        return None

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # drop blank lines
    page_text = '\n'.join(chunk for chunk in chunks if chunk)

    ans = {}

    size = extract_number(find_info_on_page(page_text, "Size"))
    volume = extract_number(find_info_on_page(page_text, "Volume"))
    avg_degree = extract_number(find_info_on_page(page_text, "Average degree"))
    category = extract_category(find_info_on_page(page_text, "Category"))
    exponent = extract_float(find_info_on_page(page_text, "Power law exponent"))
    percentile = extract_float(find_info_on_page(page_text, "90-Percentile effective diameter"))
    # Adding a new parameter.
    # [name] = extract_number(find_info_on_page(page_text, "[name]"))
    # If it does not work, then you need to write your own function which
    # extracts parameter you need from the html code.
    #
    # Do not forget to add new parameter to the returned dictionary
    tsv_link = get_tsv_link(soup)

    download_link = None
    if (tsv_link != None):
        download_link = graph_url + tsv_link


    # Add your parameter here.
    return {"tsv_link": download_link, "size": size, "volume": volume, "avg_degree": avg_degree, "category": category, "exponent": exponent, "percentile": percentile}

def add_parameter(parameter, soup):
    ans = {parameter: ''}

    return ans

def get_info_for_all_graphs(graph_urls, cnt):
    am = 0
    ans = {}
    iter = 0
    if cnt != None:
        bar = IncrementalBar('Progress', max = int(cnt))
    for graph_url in graph_urls:
        print(iter, " of ", len(graph_urls))
        iter = iter + 1
        try:
            # html = urlopen(graph_url).read()
            html= requests.get(graph_url, headers=requests_get_headers)
        except:
            print("ERROR WITH " + graph_url + '\n')
            continue
        soup = BeautifulSoup(html.text, features="html.parser")
        ret = get_graph_info(graph_url)
        if ret != None and ret['tsv_link'] != None and ret['volume'] > 2000000 and ret['volume'] < 66000000:
            ans[get_name(soup)] = ret
            am += 1

        if ret != None and cnt != None:
            bar.next()
            if int(cnt) == am:
                bar.finish()
                return ans

    return ans


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-c', '--cnt')
    parser.add_argument ('-f', '--file')
    return parser

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    cnt = namespace.cnt
    output_file = namespace.file
    if output_file == None:
        output_file = 'dict.pickle'

    category_names = get_category_names()
    graph_urls = get_graph_names(category_names)
    dict = get_info_for_all_graphs(graph_urls, cnt)

    txt_output_file = 'txt_' + output_file

    if not '.txt' in txt_output_file:
        txt_output_file += '.txt'

    f = open(txt_output_file, 'w')

    for name in dict:
        tsv_link = dict[name]['tsv_link']
        if tsv_link != None:
            pos1 = tsv_link.find('tsv.')
            pos2 = tsv_link.find('.tar')
            f.write('\'' + name + '\': {\'link\': \'' + tsv_link[pos1 + 4:pos2] + '\'},\n')


    with open(output_file, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# You can add new parameter in get_graph_info() function.
