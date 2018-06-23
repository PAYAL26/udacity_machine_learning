from pytrends.request import TrendReq
import pandas as pd
import re
import numpy as np
import sys
import time
from tqdm import tqdm
#Helper Code Taken from https://github.com/RicardoMoya/Scraping_Proxy_Tor
from ConnectionManager import ConnectionManager 
cm = ConnectionManager()

#This function scrapes data from google trends 
#It changes its IP every request to get over the daily limit request which enforced by google
def get_google_trends_data(title,page):
    ip = cm.new_identity()
    proxy = {"http": "socks5://127.0.0.1:9050", "https": "socks5://127.0.0.1:9050"}
    pytrends = TrendReq(proxies = proxy)
    kw_list = [title]
    print("Getting Data for: {}".format(title))
    pytrends.build_payload(kw_list, cat=0, timeframe='2015-07-01 2016-03-01', geo='', gprop='')
    data = pytrends.interest_over_time()
    if data.empty:
        return data
    pytrends.build_payload(kw_list, cat=0, timeframe='2016-03-02 2016-11-01', geo='', gprop='')
    data = data.append(pytrends.interest_over_time())
    pytrends.build_payload(kw_list, cat=0, timeframe='2016-11-02 2016-12-31', geo='', gprop='')
    data = data.append(pytrends.interest_over_time())
    data = data.drop(columns=['isPartial']).transpose()
    data['Page'] = page
    data.set_index('Page')
    return data
    
def main(args):
    #Regex to get the title
    REGEX = '(.*)_(.*)\.(.*)\.org_(.*)'
    #File with all the en titles
    csv = pd.read_csv('Modified.csv',header=0)
    #Checkpoint from previous run
    gd = pd.read_csv('Google_trends4.csv')
    gd = gd.set_index(['Page'])
    gd = gd.loc[gd.index.dropna()]
    csv = csv.set_index('Page')
    pages = csv.index.values
    print(len(pages))
    count = 0
    titles = {}
    count = 0
    print('Extracting Titles')
    for page in tqdm(pages):
        groups = re.match(REGEX, page)
        if groups.group(2) == 'en':
            var = titles.get(groups.group(0))
            if not var == None:
                count += 1
            titles[groups.group(0)] = groups.group(1).replace('_',' ')
        else:
            #remove from original data-set
            csv = csv.drop(page)


    print('Getting Google Trends Data')
    kw_list = []
    pages = []
    flag = 0
    pytrends_data = pd.DataFrame()
    for page, title in tqdm(titles.items()):
        print(flag)
        print(title)
        print(str(page))
        try:
            gd.loc[page]
            print('Found')
            continue
        except Exception as e:
            print('Not Found')
            pass
        try:
            data = get_google_trends_data(title=title,page=page)
        except Exception as e:
            print('Exception Raised')
            print(str(e))
            continue
        if data.empty:
            csv = csv.drop(page)
            continue
        if flag == 0:
            pytrends_data = pd.DataFrame.from_records(data)
            pytrends_data.set_index('Page')
            flag += 1
            continue
        pytrends_data = pytrends_data.append(pd.DataFrame.from_records(data))
        flag += 1
        if flag > 10:
            print(pytrends_data.shape)
            print('Saving Google Trends Data To File')
            pytrends_data.to_csv('Google_trends1.csv')
            print('Saving Data To File')
            csv.to_csv('Modified.csv')
            flag = 1
            continue

    print(pytrends_data.shape)
    print('Saving Google Trends Data To File')
    pytrends_data.to_csv('Google_trends.csv')
    print('Saving Data To File')
    csv.to_csv('Modified.csv')

if __name__ == '__main__':
    main(sys.argv[1:])
