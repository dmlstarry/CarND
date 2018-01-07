
# coding: utf-8

# In[1]:


import pandas as pd
#import keras
import numpy as np


# **Reading data set**

# In[3]:


df = pd.read_csv('driving_log.csv')
df.columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'break', 'speed']
df = df[df['speed']>20]
df = df.drop(['break','speed','throttle'],axis=1)
#df['steering_angle'] = df['steering_angle']*100
df = df.reset_index(drop=True)
df


# **Merge data set**

# In[5]:


df_c = df.iloc[:,[0,3]]
df_c.columns =['Imgpath','steering_angle']
df_r = df.loc[:,["right",'steering_angle']]
df_r['steering_angle'] -=0.1
df_r.columns =['Imgpath','steering_angle']
df_l =df.loc[:,["left",'steering_angle']]
df_l['steering_angle'] +=0.1
df_l.columns =['Imgpath','steering_angle']
df=pd.concat([df_c,df_r,df_l])

df.reset_index(drop=True)


# In[6]:


from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size=0.1, random_state=0)
print(df_train.shape)
print(df_val.shape)


# In[21]:


import random
import cv2
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.utils import shuffle
def generator(data,batch_size=64):
    '''Sometimes there are no local images, and actively skip them'''

    num=data.shape[0]
    while 1:
        
        data = shuffle(data)
        for offset in range(0,num,batch_size):
            image = []
            steer = []
            batch_data = data.values[offset:offset+batch_size]
            for i,imginfo in enumerate(batch_data):
                orimg = cv2.imread(imginfo[0])
                try:
                    img = cv2.cvtColor(orimg,cv2.COLOR_BGR2RGB)
                except:
                    continue
                image.append(img)
                steer.append(imginfo[1])
            X = np.array(image)
            Y = np.array(steer)
            yield (X,Y)

def Exgenerator(data,batch_size=64):

    num=data.shape[0]
    while 1:
        
        data = shuffle(data)
        for offset in range(0,num,batch_size):
            image = []
            steer = []
            batch_data = data.values[offset:offset+batch_size]
            for i,imginfo in enumerate(batch_data):
                orimg = cv2.imread(imginfo[0])
                try:
                    img = cv2.cvtColor(orimg,cv2.COLOR_BGR2RGB)
                except:
                    continue
                img =cv2.flip(img,1)
                image.append(img)
                steer.append(imginfo[1]*-1)
            X = np.array(image)
            Y = np.array(steer)
            yield (X,Y)            
        


# In[24]:


a,b = next(Exgenerator(df_train))
plt.imshow(a[1])
print(b[1])


# In[13]:


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Conv2D,Cropping2D,MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #normalize the data
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[14]:


model.summary()


# In[26]:


history=model.fit_generator(generator(df_train),steps_per_epoch=576,epochs=3,validation_data=generator(df_val),validation_steps=57)


# In[25]:


history=model.fit_generator(Exgenerator(df_train),steps_per_epoch=576,epochs=3,validation_data=Exgenerator(df_val),validation_steps=57)


# In[27]:


model.save("dml.h5")


# In[ ]:


from keras.models import load_model 

model =load_model("dml1.h5")



# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

