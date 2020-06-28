#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import VGG16


# In[2]:


base_model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (224, 224, 3))
base_model.layers[0].__class__.__name__
base_model.layers[0].input


# In[3]:


base_model.summary()


# In[5]:


for layer in base_model.layers:
    layer.trainable=False
for (i,layer) in enumerate(base_model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[6]:


from keras.layers import Dense, Flatten
from keras.models import Sequential

top_model = base_model.output
top_model = Flatten()(top_model)
top_model = Dense(512, activation='relu')(top_model)   
top_model = Dense(256, activation='relu')(top_model)   
top_model = Dense(128, activation='relu')(top_model) 
top_model = Dense(2, activation='softmax')(top_model)  
top_model


# In[7]:


from keras.models import Model
model = Model(inputs=base_model.input, outputs=top_model)


# In[8]:


model.output


# In[9]:


model.layers


# In[10]:


model.summary()


# In[11]:


from keras.preprocessing.image import ImageDataGenerator


# In[12]:


train_datagen = ImageDataGenerator(
    
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[13]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[14]:


train_data="C:/Users/DELL-PC/Desktop/mlops-ws/trainingimages"
train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(224, 224),
        class_mode='categorical')


# In[15]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_data="C:/Users/DELL-PC/Desktop/mlops-ws/testingimages"
test_generator = test_datagen.flow_from_directory(
        test_data,
        target_size=(224, 224),
        class_mode='categorical',
        shuffle=False)


# In[16]:


from keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(lr=0.0001),
                 loss = 'categorical_crossentropy',
                 metrics =['accuracy']
                )


# In[17]:


model.fit_generator(train_generator, epochs=2,steps_per_epoch=20, validation_data=test_generator,
                                validation_steps = 53)


# In[5]:


from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from numpy import array, expand_dims


# In[8]:


import cv2
model = load_model("vggsafe.h5")
testing_image = "C:/Users/DELL-PC/Desktop/mlops-ws/testingimages/lucky/image0.jpg"
image = load_img(testing_image, target_size=(224, 224))
image.show(testing_image)

image = array(image)
image = expand_dims(image, axis=0)
if model.predict(image)[0][0] > 0.9:
    print("not wearing helmet")
if model.predict(image)[0][1] > 0.9:
    print("wearing helmet")


# In[ ]:





# In[29]:





# In[ ]:





# In[ ]:




