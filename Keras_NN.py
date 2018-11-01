import numpy as np
import pickle
from twitter_data_processing import create_feature_sets_and_labels
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy
import keras

X_train, X_test, y_train, y_test = create_feature_sets_and_labels('data/training.csv',1000000)



model = Sequential()
model.add(Dense(555,input_dim=X_train.shape[1],activation='relu',name='Layer_1'))
model.add(Dense(350,activation='relu',name='Layer_2'))
model.add(Dense(200,activation='relu',name='Layer_3'))
model.add(Dense(2,activation='softmax',name='Out_put'))
print(model.summary())

logger = keras.callbacks.TensorBoard(
    log_dir='logs',
    write_graph=True
)

model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
model.fit(X_train,np.array([x for x in y_train]),shuffle=True,epochs=100,batch_size=10000,verbose=2,callbacks=[logger])

test_error_rate = model.evaluate(X_test, np.array([x for x in y_test]), verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate[1]))

model.save('saved_model/model')

logger = keras.callbacks.TensorBoard()