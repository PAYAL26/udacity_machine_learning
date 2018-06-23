import re
import pandas as pd
import sys
import numpy as np
import pickle
import tensorflow as tf


regex_pattern = re.compile('(.+)_([a-z][a-z]\.)?((?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.mediawiki\.org))_([a-z_-]+?)$')
#Wikipedia traffic data for only en area pages from Jul 2015 to Sept 2017 
#arround 12825 pages for whom also google trends data is available
TRAFFIC_DATA_CSV = '/Data/Final_Traffic_Data.csv'
#Google trends data from  10 Jul 2015 to 31 Dec 2016
GOOGLE_TRENDS_TRAIN_DATA_CSV = '/Data/Final_Google_Trends_Train.csv'
#Google trends data from  10 Jan 2017 to Sept 2017
GOOGLE_TRENDS_TEST_DATA_CSV = '/Data/Google_trends_Test_Final.csv'
#Pickel file where the features DataFrame will be stored
RAW_FEATURES_DATA_PATH = '/output/raw_features.pkl'

#Matches the index Pages and extracts agent, site, title
def get_features(pages):
    agents = []
    sites = []
    titles = []

    #iterating through each page to get feature list for each 
    for page in pages:
        groups = regex_pattern.fullmatch(page)
        titles.append(groups.group(1))
        sites.append(groups.group(3))
        agents.append(groups.group(4))

    df = pd.DataFrame({'agent': agents, 'site': sites,
                    'title': titles, 'Page': pages })
    return df

#Read CSV data and do minimal processing on it
def read_csv_data(path):
    data_frame = pd.read_csv(path)
    data_frame = data_frame.set_index('Page')
    data_frame.columns = pd.to_datetime(data_frame.columns)
    data_frame = data_frame.sort_index()
    return np.log1p(data_frame.fillna(0))


#Saves extracted Features in a pickel file
def save_features(feature_dict):
    with open(RAW_FEATURES_DATA_PATH, mode='wb') as file:
        pickle.dump(feature_dict, file)


def main(args):

    #Read all csv files
    traffic_data_frame = read_csv_data(TRAFFIC_DATA_CSV)
    trends_train_data_frame = read_csv_data(GOOGLE_TRENDS_TRAIN_DATA_CSV)
    trends_test_data_frame = read_csv_data(GOOGLE_TRENDS_TEST_DATA_CSV)
    start_date = traffic_data_frame.columns[0]
    end_date = traffic_data_frame.columns[-1]

    features = get_features(traffic_data_frame.index.values).set_index('Page')

    #Measure The popularity of the page by taking median 
    page_median = traffic_data_frame.median(axis=1)
    page_median = (page_median - page_median.mean()) / page_median.std()

    #Splitting the traffic data to match google trends test and train data sets 
    train_traffic_data_frame = np.split(traffic_data_frame, [len(trends_train_data_frame.columns)], axis=1)[0]
    test_traffic_data_frame = np.split(traffic_data_frame, [len(trends_train_data_frame.columns)], axis=1)[1]
    
    #Final feature dict to be save and used in the Model
    feature_tensors = dict(
        train_hits   = train_traffic_data_frame,
        test_hits    = test_traffic_data_frame,
        indexes      = traffic_data_frame.index.values,
        titles       = features['title'],
        agent        = features['agent'],
        site         = features['site'],
        page_median  = page_median,
        google_trends_train = trends_train_data_frame,
        google_trends_test  = trends_test_data_frame

    )

    save_features(feature_tensors)


if __name__ == '__main__':
    main(sys.argv[1:])