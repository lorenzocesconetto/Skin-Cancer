
# coding: utf-8

# # Notebook 4.1 - MobileNet v1

# ### 1. Setup constants

# In[1]:


from constants import *


# ### 2. Imports and notebook setup

# In[2]:


# Set up multiple outputs for cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Printing with markdown
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


# In[3]:


# Default imports
import os
import seaborn as sn
import re
import glob
import random
import shutil
from send2trash import send2trash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import pylab as pl
import cv2

# Tensorflow
import tensorflow
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import top_k_categorical_accuracy
from keras.utils.np_utils import to_categorical
# Sklearn
from sklearn.metrics import confusion_matrix


# ### 3. Create model + Data Augmentation on the fly

# #### 3.1 Set version to avoid overwirting previous files

# In[4]:


# Set version to avoid overwirting previous files
base_name = 'mobile_net_v'
version = 0
while any([bool(re.match(base_name + str(version), x)) for x in os.listdir('.')]):
    version += 1
print('version', version)


# #### 3.2 Model Constants

# In[5]:


# Constants
TRAIN_BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 10

num_train_samples = len(glob.glob(os.path.join(TRAIN_PATH, '*', '*.' + IMG_FORMAT)))
train_steps = np.ceil(num_train_samples / TRAIN_BATCH_SIZE)


# #### 3.4 Load all train images from disk to memory

# In[6]:


X_train = []
y_train = []

for cls in BINARY_CLASSES:
    counter = 0
    for img in glob.glob(os.path.join(TRAIN_PATH, cls, '*.' + IMG_FORMAT)):
        counter += 1
        X_train.append(tensorflow.keras.applications.mobilenet.preprocess_input(plt.imread(img)))
    y_train = y_train + [(1 if cls == 'mel' else 0)] * counter

X_train = np.array(X_train)
# y_train = to_categorical(np.array(y_train).reshape(-1, 1))


# In[7]:


# # Data flow
# datagen = ImageDataGenerator(preprocessing_function = \
#                              tensorflow.keras.applications.mobilenet.preprocess_input)

# train_flow = datagen.flow_from_directory(train_path,
#                                          target_size=(image_size, image_size),
#                                          batch_size=train_batch_size)


# In[8]:


# train_flow.class_indices


# #### 3.5 Load validation image to memory

# In[7]:


X_val = []
y_val = []

for cls in BINARY_CLASSES:
    counter = 0
    for img in glob.glob(os.path.join(VAL_PATH, cls, '*.' + IMG_FORMAT)):
        counter += 1
        X_val.append(tensorflow.keras.applications.mobilenet.preprocess_input(plt.imread(img)))
    y_val = y_val + [(1 if cls == 'mel' else 0)] * counter

X_val = np.array(X_val)
# y_val = to_categorical(np.array(y_val).reshape(-1, 1))


# In[8]:


# _ = plt.imshow(X_val[1090])


# #### 3.3 Transfer Learning MobileNet

# In[9]:


# Load MobileNet
mobile = tensorflow.keras.applications.mobilenet.MobileNet()


# In[10]:


# Set architecture
x = mobile.layers[-6].output

# x = Dropout(0.4)(x)
# x = Dense(1024, activation='tanh')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='tanh')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=mobile.input, outputs=predictions)


# In[11]:


model.summary()


# In[12]:


# Leave only the last 11 layers trainable.
# 1 drop + 1 dense: 22, 16, 9
# 2 drop + 2 dense: 11, 18
# 3 drop + 3 dense: 13
# 2 drop + 2 BatchNormalization + 2 dense: 20

for layer in model.layers[:-20]:
    layer.trainable = False


# In[13]:


# name, name_scope
# dir(model.layers[-22])
model.layers[-20].name


# In[14]:


model.compile(optimizer=optimizers.SGD(lr=0.001, nesterov=True), 
              loss='binary_crossentropy', metrics=['accuracy'])


# #### 3.6 Train Model

# In[15]:


checkpoint = ModelCheckpoint(base_name + str(version) + '_{val_acc:.2f}.h5', monitor='val_acc', verbose=1, 
                             save_best_only=True, save_weights_only=False)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=2,
                                   verbose=1, mode='max', min_lr=0.0001, cooldown=2)
                              
callbacks_list = [checkpoint, reduce_lr]


# In[17]:


history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=TRAIN_BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=(X_val, y_val)
                   )


# In[19]:


plt.style.use("ggplot")
_ = plt.figure(figsize=(15, 10))
_ = plt.plot(np.arange(0, NUM_EPOCHS), history.history["loss"], label="train_loss")
_ = plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_loss"], label="val_loss")
_ = plt.plot(np.arange(0, NUM_EPOCHS), history.history["acc"], label="train_acc")
_ = plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_acc"], label="val_acc")
_ = plt.title("Erro e Acurácia")
_ = plt.xlabel("# Época")
_ = plt.ylabel("Erro & Acurácia")
_ = plt.legend(loc="upper right")

plt.savefig(base_name + str(version) + '.png')


# ### 3.7 Confusion matrix

# In[ ]:


probas = model.predict(X_val)
classes = np.argmax(probas, axis=1)


# In[37]:


cm = confusion_matrix(y_true=np.argmax(y_val, axis=1), y_pred=classes , labels=NUMERIC_CLASSES)

df_cm = pd.DataFrame(cm, index=NUMERIC_CLASSES)
_ = plt.figure(figsize = (10,7))
_ = sn.heatmap(df_cm, annot=True)

plt.savefig(base_name + str(version) + '_confusion_matrix.png')


# In[25]:





# In[ ]:




