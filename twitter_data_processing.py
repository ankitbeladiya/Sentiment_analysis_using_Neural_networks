import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split



def create_feature_sets_and_labels(data,featureSize):
    data = pd.read_csv(data, encoding='utf_8', names=['rating', 'id', 'date', 'query', 'user', 'tweet'])
    data = data[['rating', 'tweet']].iloc[800000-int(featureSize/2):800000+int(featureSize/2)]
    print('Processing ', len(data), ' reviews')
    X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['rating'], random_state=0,test_size = 0.1)
    vect = CountVectorizer(max_df=featureSize,ngram_range=(1,2),stop_words='english')
    return vect.fit_transform(X_train), vect.transform(X_test), y_train.map({0:[0,1],4:[1,0]}), y_test.map({0:[0,1],4:[1,0]})


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = create_feature_sets_and_labels('data/training.csv',featureSize=100000)
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([X_train, X_test, y_train, y_test], f)

        print('done')


