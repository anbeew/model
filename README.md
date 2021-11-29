import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops.gen_dataset_ops import model_dataset
from tensorflow.python.ops.variables import model_variables

directory=r'C:\datasets\orange'
categories=['fresh','rotten']
for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        #plt.imshow(img_array)
        #plt.show()
        break
    break
#print(img_array)

img_size=(70,70)
new_array=cv2.resize(img_array, img_size)
print(new_array.shape)
#plt.imshow(new_array)
#plt.show()

training_data=[]
def create_training_data():
    for category in categories:
        path=os.path.join(directory,category)
        class_names=categories.index(category)
        for img in os.listdir(path):
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array, img_size)
                training_data.append([new_array,class_names])

create_training_data()

#print(len(training_data))
import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

x=[]
y=[]
for features,label in training_data:
    x.append(features)
    y.append(label)
X = np.array(x).reshape(-1, 70, 70, 3)

import pickle
pickle_out=open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open('y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

#X[1]

#y[1]

pickle_in=open('X.pickle', 'rb')
X= pickle.load(pickle_in)
pickle_in=open('y.pickle', 'rb')
y = pickle.load(pickle_in)
y = np.array(y)
X = X/225.0
def create_model():
    model=Sequential([
        layers.Flatten(input_shape=(70,70,3)),
        layers.Dense(128,activation="relu"),
        layers.Dense(2),
    ])
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
    model.fit(X, y, batch_size=16, epochs=10,verbose=2, validation_split=0.1)
    return model
model = create_model()
score = model.evaluate(X,y, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
model.save('saved_model/my_model')
