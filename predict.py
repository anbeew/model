import tensorflow as tf
from tensorflow.keras.models import  load_model
import cv2
import numpy as np
import pickle

height=70
width=70

#load the model
model=load_model('saved_model/my_model', compile=True)
#load the labels
pickle_in=open('y.pickle', 'rb')
y = pickle.load(pickle_in)
y = np.array(y)
class_names=['Fresh','rotten']

def resize(filepath):
	img_array = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)  # read in the image
	new_array = cv2.resize(img_array, (height,width))  # resize image to match model's expected sizing
	img=new_array.reshape(-1, height, width, 3)/255.0
	return   img# return the image with shaping that TF wants.

prediction=model.predict([resize('try2org.jpg')])
#print(prediction)
index=np.argmax(prediction)
y=y[index]
img=cv2.imread('try2org.jpg')
cv2.putText(img,f'prediction: {class_names[y]}', (60,50), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0,255,0),2)
cv2.imshow('img resized',img)
cv2.waitKey(0)



