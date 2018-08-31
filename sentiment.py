import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


hm_lines = 10000000

def sample_handling(sample, classification):

	featureset = []
	c = []
	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			featureset.append(l)
			c.append(classification)
	features =pd.DataFrame(data = [featureset,c])
	return features.T

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	pos = sample_handling('pos.txt',np.array([1,0]))
	neg = sample_handling('neg.txt',np.array([0,1]))
	features = pd.concat([pos,neg])
	features = features.sample(n = len(features))
	features.reset_index(drop = True ,inplace = True)
	X_train, X_test, y_train, y_test = train_test_split(features[0], features[1], random_state=0)
	vect = CountVectorizer(min_df = 50,max_df=10000000,ngram_range=(1,2)).fit(features[0])
	# X_train_vec = vect.transform(X_test)

	# print(vect.transform(X_test).toarray().shape)
	return vect.transform(X_train).toarray(), vect.transform(X_test).toarray(), y_train, y_test

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([X_train, X_test, y_train, y_test], f)




