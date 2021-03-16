#### Training 
import numpy as np
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
import text_preparation
import config


x_train,y_train = text_preparation.Base()
seq_length = config.Seq_length
epochs = config.epochs
batch_size = config.batch_size


### defining the Model
model = Sequential()
model.add(LSTM(64,input_shape=(seq_length,1)))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1],activation="softmax"))

## compile the model
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)

model.save("/home/sai/Documents/new ml/Text Generator/models/model.h5")

