import pickle
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

def get_category_names():
    url = "http://konect.cc/categories/"
   # html = urlopen(url).read()
   # soup = BeautifulSoup(html, features="html.parser")
    html= requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
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
        html= requests.get(category_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
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
    html= requests.get(graph_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
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
    return {"tsv_link": download_link, "size": size, "volume": volume, "avg_degree": avg_degree, "category": category}

def add_parameter(parameter, soup):
    ans = {parameter: ''}
    
    return ans

def get_info_for_all_graphs(graph_urls):
    cnt = 0 # !
    ans = {}
    for graph_url in graph_urls:     
        try:
           # html = urlopen(graph_url).read()
            html= requests.get(graph_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'})
        except:
            print("ERROR WITH " + graph_url + '\n')
            continue
        soup = BeautifulSoup(html.text, features="html.parser")
        ret = get_graph_info(graph_url)
        if ret != None:
            ans[get_name(soup)] = ret
            cnt += 1

        if cnt == 10: # !
            return ans # !

    return ans

def main():
    category_names = get_category_names()
    graph_urls = get_graph_names(category_names)
    dict = get_info_for_all_graphs(graph_urls)
    print(dict)
    with open('dict.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


main()

# You can add new parameter in get_graph_info() function.
