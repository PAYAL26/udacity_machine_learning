from keras import metrics
from keras.optimizers import Adagrad
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU, Flatten
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import argparse
import pickle
import numpy as np
import pandas as pd
from keras.callbacks import History 
from tqdm import tqdm
from os.path import isfile
from keras import losses
# from keras.plot_model import plot
history = History()

#Log Files and Cache files
LOG_DIR_PATH = '/output/logs/'
RAW_FEATURES_DATA_PATH = '/data/raw_features.pkl'
META_DATA_PATH = '/data/meta.pkl'
final_encoder_file = '/data/encoder.pkl'
final_decoder_file = '/data/decoder.pkl'
test_encoder_file = '/data/test_encoder.pkl'
test_decoder_file = '/data/test_decoder.pkl'

#SMAPE Loss
def smape(true, predicted):
    cost = 2/len(true)
    lst = []
    for i in range(len(true)):
        num = abs(f[i]-a[i])
        denom = f[i]+a[i]
        lst.append(num/denom)
        
    return const*sum(lst)

#Read The pickel files saved by the features.py file
def get_pickel_file(filename):
    with open(filename, mode='rb') as file:
        raw_file = pickle.load(file)
    return raw_file

#Encode categorical features(site,agent,page) 
def encode_categorical_data(data):
    le = LabelEncoder()
    data = le.fit(data).transform(data)
    data = data.reshape(-1,1)
    enc = OneHotEncoder(dtype = np.float32)
    enc.fit(data).transform(data)
    return data

#Creates The input for encoders
#takes traffic data and Google trends Data as input
#Returns encoder inputs
def make_encoder_inputs(traffic_hits, google_trends):
    #Making input data for training encoder 1
    final_train_encoder_input1 = np.ndarray((traffic_hits.shape[0],traffic_hits.shape[1],5))
    for i in tqdm(range(traffic_hits.shape[0])):
        
        num_days = traffic_hits.shape[1]

        inpt = pd.DataFrame(traffic_hits.iloc[i])
        inpt.columns = ['traffic']
        inpt.index.name = 'Date'

        inpt['google_trends'] = pd.DataFrame(google_trends.iloc[i])
        
        idx = np.empty((num_days), dtype=np.float32 )
        for j in range(num_days):
            idx[j] = float(train_features['indexes'][i][0])
        inpt['indexes'] = pd.Series(np.array(idx, dtype=np.float32), index=inpt.index)
        
        idx = np.empty((num_days), dtype=np.float32 )
        for j in range(num_days):
            idx[j] = float(train_features['agent'][i][0])
        inpt['agent'] = pd.Series(np.array(idx, dtype=np.float32), index=inpt.index)
        
        idx = np.empty((num_days), dtype=np.float32 )
        for j in range(num_days):
            idx[j] = float(train_features['site'][i][0])
        inpt['site'] = pd.Series(np.array(idx, dtype=np.float32), index=inpt.index)
        reshaped = inpt.values.reshape((1,inpt.shape[0], inpt.shape[1]))
        final_train_encoder_input1 = np.append(final_train_encoder_input1, reshaped, axis=0)
    return final_train_encoder_input1

#Creates Decoder Inputs
# Takes traffic Data and Google trends data as input
# Returns the target out put and decoder inputs
def make_decoder_inputs(traffic_hits, google_trends):
    #Making input data for training encoder 1
    final_train_decoder_input1 = np.ndarray((traffic_hits.shape[0],traffic_hits.shape[1],4))
    expected_hits = np.ndarray((traffic_hits.shape[0],traffic_hits.shape[1]))
    for i in tqdm(range(traffic_hits.shape[0])):
        
        num_days = traffic_hits.shape[1]
        arr = pd.DataFrame(traffic_hits.iloc[i]).values.reshape(1,num_days)
        expected_hits = np.append(expected_hits, arr, axis=0)

        inpt = pd.DataFrame(google_trends.iloc[i])
        inpt.columns = ['google_trends']
        inpt.index.name = 'Date'
        
        idx = np.empty((num_days), dtype=np.float32 )
        for j in range(num_days):
            idx[j] = float(train_features['indexes'][i][0])
        inpt['indexes'] = pd.Series(np.array(idx, dtype=np.float32), index=inpt.index)
        
        idx = np.empty((num_days), dtype=np.float32 )
        for j in range(num_days):
            idx[j] = float(train_features['agent'][i][0])
        inpt['agent'] = pd.Series(np.array(idx, dtype=np.float32), index=inpt.index)
        
        idx = np.empty((num_days), dtype=np.float32 )
        for j in range(num_days):
            idx[j] = float(train_features['site'][i][0])
        inpt['site'] = pd.Series(np.array(idx, dtype=np.float32), index=inpt.index)
        reshaped = inpt.values.reshape((1,inpt.shape[0], inpt.shape[1]))
        final_train_decoder_input1 = np.append(final_train_decoder_input1, reshaped, axis=0)
    print('Done')
    return final_train_decoder_input1, expected_hits


train_features = get_pickel_file(RAW_FEATURES_DATA_PATH)

#Removing Unuable data (in-order to keep the encoder and decoder inputs same for both train and test phase)
hits = train_features['train_hits']
hits.drop(hits.columns[:44], axis=1, inplace=True)
train_features['train_hits'] = hits

hits = train_features['google_trends_train']
hits.drop(hits.columns[:44], axis=1, inplace=True)
train_features['google_trends_train'] = hits

#Encoding categorical data
train_features['indexes'] = encode_categorical_data(train_features['indexes'])
train_features['agent'] = encode_categorical_data(train_features['agent'])
train_features['site'] = encode_categorical_data(train_features['site'])

#Spliting the traffic dataset into 2 training sets and one test set
main_split1 = np.split(train_features['train_hits'], [253], axis=1)[0]
main_split2 = np.split(train_features['train_hits'], [253], axis=1)[1]

train_encoder_hits1 = np.split(main_split1, [186], axis=1)[0]
train_decoder_hits1 = np.split(main_split1, [186], axis=1)[1]

train_encoder_hits2 = np.split(main_split2, [186], axis=1)[0]
train_decoder_hits2 = np.split(main_split2, [186], axis=1)[1]

#Spliting the google trends dataset for training
google_split1 = np.split(train_features['google_trends_train'], [253], axis=1)[0]
google_split2 = np.split(train_features['google_trends_train'], [253], axis=1)[1]

train_google_encoder_hits1 = np.split(google_split1, [186], axis=1)[0]
train_google_decoder_hits1 = np.split(google_split1, [186], axis=1)[1]

train_google_encoder_hits2 = np.split(google_split2, [186], axis=1)[0]
train_google_decoder_hits2 = np.split(google_split2, [186], axis=1)[1]

#Preparing test splits
test_encoder = np.split(train_features['test_hits'], [186], axis=1)[0]
test_decoder = np.split(train_features['test_hits'], [186], axis=1)[1]

#Spliting google trends data for testing
test_google_encoder_hits = np.split(train_features['google_trends_test'], [186], axis=1)[0]
test_google_decoder_hits = np.split(train_features['google_trends_test'], [186], axis=1)[1]

#Checking For Cached Data Before making the entire input data
if isfile(final_encoder_file):
    final_train_encoder_input1 = get_pickel_file(final_encoder_file)
else:
    #Making input data for training encoder 1
    final_train_encoder_input1 = make_encoder_inputs(train_encoder_hits1, train_google_encoder_hits1)
    with open('/output/encoder.pkl', mode='wb') as file:
        pickle.dump(final_train_encoder_input1, file)

if isfile(final_decoder_file):
    final_train_decoder_input1, expected_hits = get_pickel_file(final_decoder_file)
else:
    #Making input data for training decoder 1
    final_train_decoder_input1, expected_hits = make_decoder_inputs(train_decoder_hits1, train_google_decoder_hits1)
    with open('/output/decoder.pkl', mode='wb') as file:
        pickle.dump((final_train_decoder_input1,expected_hits), file)


#Building Model
encoder_inputs = Input(shape=(186,5), name='Encoder_Input', dtype='float32')
#Encoder
encoder_l1 = GRU(512, activation='selu', return_sequences=True)
encoder_outputs = encoder_l1(encoder_inputs)

encoder_l2 = GRU(512,activation='selu', return_state=True)
encoder_outputs, encoder_state = encoder_l2(encoder_outputs)

#Decoder
decoder_inputs = Input(shape=(67,4), name='Decoder_Input', dtype='float32')

decoder_l1 = GRU(512,activation='selu',  return_sequences=True, return_state=True, name='decoder1')
decoder_outputs, state_h = decoder_l1(decoder_inputs, initial_state=encoder_state)

decoder_l2 = GRU(512,activation='selu', return_sequences=True, return_state=True, name='decoder2')
decoder_outputs, _ = decoder_l2(decoder_outputs)

flatten = Flatten()
flatten_output = flatten(decoder_outputs)
decoder_dense = Dense(67, activation='selu')
decoder_outputs = decoder_dense(flatten_output)

#Keras functional API
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#Compiling
model.compile(optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0), loss=losses.mean_squared_error, metrics=['accuracy'])
hist = model.fit([final_train_encoder_input1, final_train_decoder_input1], expected_hits,
          epochs=10,
          validation_split=0.2)

if isfile(test_encoder_file):
    final_train_encoder_input1 = get_pickel_file(test_encoder_file)
else:
    #Making input data for test encoder 
    final_test_encoder_input1 = make_encoder_inputs(train_encoder_hits1, train_google_encoder_hits1)
    with open('/output/test_encoder.pkl', mode='wb') as file:
        pickle.dump(final_test_encoder_input1, file)

if isfile(test_decoder_file):
    final_test_decoder_input1, expected_hits = get_pickel_file(test_decoder_file)
else:
    #Making input data for test encoder 
    final_test_decoder_input1, expected_hits = make_decoder_inputs(train_decoder_hits1, train_google_decoder_hits1)
    with open('/output/test_decoder.pkl', mode='wb') as file:
        pickle.dump((final_test_decoder_input1,expected_hits), file)

hist1 = model.evaluate([final_test_encoder_input1,final_test_decoder_input1], expected_hits)


#later use for visualisation
with open('/output/history.pkl', mode='wb') as file:
        pickle.dump(history.history, file)


# plot(hist)
