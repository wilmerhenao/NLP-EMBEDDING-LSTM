import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
import gc
from datetime import datetime, timedelta
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews

pd.set_option('max_columns', 50)


class SequenceGenerator:
    def __init__(self, df, num_cols, cat_cols, lstm_cols=[], window=10, batch_size=32, train=True):
        self.batch_size = batch_size
        self.train = train
        self.window = window
        # self.data = df.sort_values(by=['time'])
        self.data = df
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.lstm_cols = lstm_cols
        self.cols = num_cols + cat_cols
        self.y01 = self.data.returnsOpenNextMktres10.map(lambda x: 0 if x < 0 else 1)

    def generate(self):
        while True:
            X, y, d, r, u = {'num': [], 'lstm': []}, [], [], [], []
            for cat in cat_cols:
                X[cat] = []
            # Subtract batch_size and window to make sure that I don't cross the boundary
            for seq in range(0, self.data.shape[0] - self.batch_size - self.window, self.batch_size):
                # Todo tabuleado desde aca hasta el def step function
                X['num'] = self.data[self.num_cols].iloc[
                           (seq + self.window):(seq + self.window + self.batch_size)].values
                # The next for loop will enter the LSTM part. I only select members of length of the window
                for subseq in range(seq, seq + self.batch_size):
                    X['lstm'].append(self.data[self.lstm_cols].iloc[seq:(seq + self.window)].values)
                for cat in cat_cols:
                    X[cat] = self.data[cat].iloc[(seq + self.window):(seq + self.window + self.batch_size)].values
                y = self.y01.iloc[(seq + self.window):(seq + self.window + self.batch_size)].values
                # d = self.data.time.iloc[(seq + self.window):(seq + self.window + self.batch_size)]
                # r = self.data.returnsOpenNextMktres10.iloc[(seq + self.window):(seq + self.window + self.batch_size)]
                # u = self.data.universe.iloc[(seq + self.window):(seq + self.window + self.batch_size)]

                X_ = {'num': np.array(X['num']), 'lstm': np.array(X['lstm'])}
                for cat in cat_cols:
                    X_[cat] = np.array(X[cat])
                y_ = np.array(y)
                # r_, u_, d_ = np.array(r),np.array(u), np.array(d)
                X, y, d, r, u = {'num': [], 'lstm': []}, [], [], [], []
                for cat in cat_cols:
                    X[cat] = []
                if self.train:
                    yield X_, y_
                else:
                    yield X_, y_, r_, u_, d_

    def steps(self):
        # get number of steps per epoch
        steps = 0
        num_sequences = self.data.shape[0] - self.window
        steps += num_sequences // self.batch_size
        return steps


env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
# Remove data before 2009
start = datetime(2015, 1, 1, 0, 0, 0).date()
market_train = market_train.loc[market_train['time'].dt.date >= start].reset_index(drop=True)
news_train = news_train.loc[news_train['time'].dt.date >= start].reset_index(drop=True)
# Preprocess some news and categorical data, also remove some columns that are not used
print('preprocessing news...')


def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline', 'firstCreated', 'sourceTimestamp',
    ]
    news_train.drop(drop_list, axis=1, inplace=True)
    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train


news_train = preprocess_news(news_train)
print('Done')
# Let's add some magic to the data count the number of assetcodes in the news bef. I destroy this data below
news_train['countAssetCodes'] = [i.count('.') for i in news_train['assetCodes']]
print('Unstack the news...')


# Now I'm going to unstack the news
def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)] * len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df


index_df = unstack_asset_codes(news_train)
print('Done')
# and merge the news on this frame
print('Merge the news on this frame...')


def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack


news_unstack = merge_news_on_index(news_train, index_df)
del news_train, index_df
gc.collect()
print('Done')
# Group by date and asset using simple mean
print(
    'Group news by date and asset using a simple mean (Think this better there are several news per day per asset)...')


def group_news(news_frame):
    news_frame['date'] = news_frame.time.dt.date  # Add date column

    aggregations = ['mean']
    gp = news_frame.groupby(['assetCode', 'date']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date']}
    return gp.astype(float_cols)


news_agg = group_news(news_unstack)
del news_unstack;
gc.collect()
print('Done')
# Finally, merge on assetCode and Date
print('Merge both datasets based on asset code and date...')
market_train['date'] = market_train.time.dt.date
market_train = market_train.merge(news_agg, how='left', on=['assetCode', 'date'])
del news_agg
gc.collect()
print('Done')

cat_cols = ['assetCode']
lstm_cols = ['returnsClosePrevMktres1']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']
#num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsClosePrevRaw10']
news_num_cols = ['bodySize', 'companyCount', 'sentenceCount', 'wordCount', 'firstMentionSentence',
                 'relevance', 'sentimentClass', 'sentimentNegative',
                 'sentimentNeutral','sentimentPositive','sentimentWordCount','noveltyCount12H',
                 'noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D','volumeCounts12H',
                 'volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D', 'countAssetCodes']
news_num_cols = []
num_cols += [word + '_mean' for word in news_num_cols]
newsnum = len(num_cols)

from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=12)

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]

for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets

from sklearn.preprocessing import StandardScaler

# market_train[num_cols] = market_train[num_cols].fillna(market_train[num_cols].mean())
market_train[num_cols] = market_train[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
print('Done')

print('Defining the architecture...')
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, LSTM
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32,activation='relu')(categorical_logits)
#categorical_logits = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(categorical_logits)

numerical_inputs = Input(shape=(len(num_cols),), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(128,activation='tanh')(numerical_logits)

# LSTM part
lstm_inputs = Input(shape = (10, len(lstm_cols)), name='lstm')
lstm_logits = LSTM(16)(lstm_inputs)

logits = Concatenate()([numerical_logits,categorical_logits,lstm_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs] + [lstm_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)
model.summary()
print('Done')

mytrain = SequenceGenerator(market_train.iloc[train_indices], num_cols, cat_cols, lstm_cols)
myvalid = SequenceGenerator(market_train.iloc[val_indices], num_cols, cat_cols, lstm_cols)
#del market_train
gc.collect()
train_steps = mytrain.steps()
test_steps = myvalid.steps()
print('Done')

from keras.callbacks import EarlyStopping, ModelCheckpoint
print('Train the Neural Network Model...')
check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
#model.fit(X_train,y_train.astype(int),
#          validation_data=(X_valid,y_valid.astype(int)),
#          epochs=2,
#          verbose=True,
#          callbacks=[early_stop,check_point])
model.fit_generator(mytrain.generate(),
          validation_data=myvalid.generate(),
          epochs=1,
          steps_per_epoch=train_steps,
          validation_steps=test_steps,
          callbacks=[early_stop,check_point])#, workers=4, use_multiprocessing=True)
print('Done')

def get_input(market_train):
    X_num = market_train[num_cols]
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train[cat_cols]
    y = (market_train['returnsOpenNextMktres10'] >= 0).values
    r = market_train['returnsOpenNextMktres10'].values
    u = market_train['universe']
    d = market_train['time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(mytrain.data)
del mytrain
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(myvalid.data)
del myvalid
gc.collect()

# distribution of confidence that will be used as submission
print('Evaluating confidence that will be used as submission...')
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()
print('Done.')

# calculation of actual metric that is used to calculate final score
print('Calculating actual metric that is used to calculate the final score...')
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print('Done')
print(score_valid)

days = env.get_prediction_days()

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days += 1
    print(n_days, end=' ')
    t = time.time()
    # merging treatments
    news_obs_df = preprocess_news(news_obs_df)
    news_obs_df['countAssetCodes'] = [i.count('.') for i in news_obs_df['assetCodes']]
    index_df = unstack_asset_codes(news_obs_df)
    news_unstack = merge_news_on_index(news_obs_df, index_df)
    del news_obs_df, index_df
    gc.collect()
    news_agg = group_news(news_unstack)
    del news_unstack;
    gc.collect()
    market_obs_df['date'] = market_obs_df.time.dt.date
    market_obs_df = market_obs_df.merge(news_agg, how='left', on=['assetCode', 'date'])
    del news_agg;
    gc.collect()
    ###########################
    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    # market_obs_df[num_cols] = market_obs_df[num_cols].fillna(market_obs_df[num_cols].mean())
    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num': X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values

    prep_time += time.time() - t

    t = time.time()
    market_prediction = model.predict(X_test)[:, 0] * 2 - 1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() - t

    t = time.time()
    preds = pd.DataFrame({'assetCode': market_obs_df['assetCode'], 'confidence': market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds, how='left').drop('confidenceValue', axis=1).fillna(
        0).rename(columns={'confidence': 'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')

plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()