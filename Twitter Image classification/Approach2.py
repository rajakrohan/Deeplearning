#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import np_utils


# In[21]:


FILE_PATHS = [
    
    "/home/du4/14CS30029/dlab/annotations/california_wildfires_final_data.tsv",
    "/home/du4/14CS30029/dlab/annotations/hurricane_harvey_final_data.tsv",
    "/home/du4/14CS30029/dlab/annotations/hurricane_irma_final_data.tsv",
    "/home/du4/14CS30029/dlab/annotations/hurricane_maria_final_data.tsv",
    "/home/du4/14CS30029/dlab/annotations/iraq_iran_earthquake_final_data.tsv",
    "/home/du4/14CS30029/dlab/annotations/mexico_earthquake_final_data.tsv",
    "/home/du4/14CS30029/dlab/annotations/srilanka_floods_final_data.tsv"
]


# In[5]:


data = []
l = []
for file in FILE_PATHS:
    df = pd.read_csv(file,sep="\t")
    #print (df.columns)
    for index, item in df.iterrows():
        #print ('label : ', item['image_human'], str(item['image_human']))
        #break
        if str(item['image_human']) == 'infrastructure_and_utility_damage':
            l = [item['image_path'],1]
        else:
            l = [item['image_path'],0]
        data.append(l)


# In[6]:


df = pd.DataFrame(data)
df


# In[ ]:


df.groupby(1).count()


# In[ ]:


minw=100000
minh=100000
maxw=0
maxh=0
dim = []
for index, row in df.iterrows():
    im = Image.open(row[0])
    width, height = im.size
    dim.append((row[0],width,height,row[1]))
    if row[1] == 1:
        im.rotate(45, expand=True)
        width, height = im.size
        dim.append((row[0],width,height,row[1]))
        im.rotate(-45, expand=True)
        width, height = im.size
        dim.append((row[0],width,height,row[1]))


# In[ ]:


newDf = pd.DataFrame(dim, columns=['path' , 'w', 'h','l'])


# In[ ]:


newDf = newDf.query('w > 320 & h > 240')


# In[ ]:


newDf.groupby('l').count()


# In[ ]:


newDf = newDf.drop(newDf.query('l == 0').sample(frac = 0.2).index)


# In[ ]:


newDf.groupby('l').count()


# In[ ]:


matrix_data = []
for index, row in newDf.iterrows():
    x = load_img(row[0], target_size=(320,240))    
    x = (1.0/255)*img_to_array(x)
    #x = np.expand_dims(x, axis=0)
    l = [x,row[3]]
    matrix_data.append(l)


# In[ ]:


fd = pd.DataFrame(matrix_data, columns=['pixels' , 'label'])
(X, y) = (fd['pixels'],fd['label'])
X.shape


# In[ ]:


x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# df = pd.DataFrame(np.random.randn(17056, 2))
# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]
x_train = np.array(x_train)
x_test = np.array(x_test)
# print(train.shape)
#X_train = pd.DataFrame(x_train, columns=['pixels'])
#X_test = pd.DataFrame(x_test, columns=['pixels'])
#y_train = pd.DataFrame(Y_train, columns=['pixels'])
#y_test = pd.DataFrame(Y_test, columns=['pixels'])
#x_train[0][0].shape
#x_train = x_train.reshape(x_train.shape[0], 1)
#X_test = x_test.values.reshape(x_test.shape[0], 1)
# train = []
# for all in x_train:
#     train.append(all.reshape(3,320, 240))# 
#X_test.shape

train = []
test = []
for arr in x_train:
    tmp = arr.reshape(320, 240,3)
    train.append(tmp)
    
for arr in x_test:
    tmp = arr.reshape(320, 240,3)
    test.append(tmp)
    
    
train = np.array(train)
test = np.array(test)
# train[0].shape


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(320, 240,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[18]:


#batch_size to train
batch_size = 20
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 10

hist = model.fit(train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
                verbose=1, validation_data=(test, Y_test))

#Evaluating on validation set for Computing loss and accuracy :


# In[19]:


import matplotlib.pyplot as plt
import matplotlib
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)


# In[28]:


plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# In[ ]:


plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# In[ ]:




score = model.evaluate(test, Y_test,  verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(test[1:5]))
print(Y_test[1:5])


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
fname = "name.hdf5"
model.save_weights(fname,overwrite=True)


# In[33]:


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(320, 240,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model


# In[41]:


from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_trained_model('/home/du4/14CS30029/dlab/best.h5')
img_width, img_height = 320, 240
# load the model we saved
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
from IPython.display import Image,display
display(Image(filename='test02.jpg'))

img = image.load_img('test02.jpg', target_size=(img_width, img_height))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images)
print ('This image has been classified as class  ',classes.flatten())


# In[ ]:





# In[ ]:




