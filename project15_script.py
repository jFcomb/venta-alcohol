#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam


# In[2]:


#Carga de datos
df = pd.read_csv('/datasets/faces/labels.csv')

#Carga de datos de entrenamient
def load_train(path):
    
    train_datagen = ImageDataGenerator(rescale = 1/255)
    
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe = df,
        directory = '/datasets/faces/final_files/',
        x_col = 'file_name',
        y_col = 'real_age',
        target_size = (224,224),
        batch_size = 32,
        class_mode = 'raw', #numpy array of values in y_col column(s),
        seed = 12345,
    )
 
    return train_gen_flow


# In[4]:


#Carga de datos de prueba
def load_test(path):
    
    test_datagen = ImageDataGenerator(rescale = 1/255)
    
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe = df,
        directory = '/datasets/faces/final_files/',
        x_col = 'file_name',
        y_col = 'real_age',
        target_size = (224,224),
        batch_size = 32,
        class_mode = 'raw', #numpy array of values in y_col column(s)
        seed = 12345,
    )

    return test_gen_flow


# In[6]:


#Creación del modelo
def create_model(input_shape):
    print(input_shape)
    backbone = ResNet50(
        input_shape=input_shape,
#         classes=1000,
        weights='imagenet', 
        include_top=False
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
#     model.add(Flatten())
#     model.add(BatchNormalization())
    model.add(Dense(units=256, activation = 'relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(.2))
    model.add(Dense(units=128,activation = 'relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(.5))
    model.add(Dense(units=16,activation = 'relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(.5))
    model.add(Dense(units=1,kernel_initializer='normal'))
    
    
    optimizer = Adam(lr = 0.0002)
    model.compile(
    optimizer = 'adam',
    loss = 'mean_absolute_error',
    metrics = ['mean_absolute_error']
    )
    
    model.summary()

    return model


# In[ ]:


#Entrenamiento del modelo
def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    
    if steps_per_epoch is None:     
        steps_per_epoch = len(train_data) 
    if validation_steps is None:     
        validation_steps = len(test_data)
    
    model.fit(train_data,
          validation_data=test_data,
          batch_size=batch_size,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps,
          verbose=2) 

    return model

