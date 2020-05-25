#!/usr/bin/env python
# coding: utf-8


# In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential

sys.stderr = stderr
# 

# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D( filters = 32,
                       kernel_size=(3,3),
                       activation = 'relu',
                       input_shape=(64,64,3)))


# 

# In[8]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


model.add(Flatten())


# In[10]:


model.add(Dense(units =128, activation = 'relu'))


# In[11]:


model.add(Dense(units=1, activation = 'sigmoid'))


# In[12]:


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[13]:


from keras_preprocessing.image import ImageDataGenerator


# In[14]:

save = sys.stdout
sys.stdout = open("output.txt", "w+")
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'C:\\Users\\DeLL\\Desktop\\Cat and dog dataset DL\\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'C:\\Users\\DeLL\\Desktop\\Cat and dog dataset DL\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

sys.stdout.close()
sys.stdout = save

history = model.fit(
        training_set,
        steps_per_epoch=100,
        epochs=5,
        validation_data=test_set,
        validation_steps=10,
        verbose=0
        )


# In[22]:


print ("Accuracy of the trained model is : {} %".format ( 100 * history.history['val_accuracy'][-1])) 


# In[19]:


model.save('dog and cat model.h5')


# In[ ]:




