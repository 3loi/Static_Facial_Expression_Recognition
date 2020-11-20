#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

from __future__ import print_function, division
from util import *
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from imutils import face_utils
import imutils
import dlib
import cv2
import json
from shutil import rmtree
from config import affectnet_config as config
from pyimagesearch.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor, SimplePreprocessor, MeanPreprocessor, CropPreprocessor
from keras.models import load_model


# In[3]:



images = glob('/path/to/images/jpg/A/*/*.jpg')
print('Found {} images'.format( len(images)))


# In[4]:


means = json.loads(open(config.DATASET_MEAN).read())
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

#load the pretrained network
model = load_model(config.MODEL_PATH)


# In[5]:


predictions = []
for image in tqdm(images[:100]):
    face = face_get(image)
    crop = sp.preprocess(face)
    crop = mp.preprocess(crop)
    crop = np.expand_dims(crop, axis=0)
    #crop = np.array([iap.preprocess(c) for c in crop], dtype = "float32")
    pred = model.predict(crop)
    predictions.append([image, pred.mean(axis=0)])


# In[6]:


neu = [pred[1][0] for pred in predictions]
hap = [pred[1][1] for pred in predictions]
sad = [pred[1][2] for pred in predictions]
ang = [pred[1][3] for pred in predictions]
imgs = [pred[0] for pred in predictions]


# In[7]:


df = pd.DataFrame({'image':imgs, 'Neutral':neu, 'Happy':hap, 'Sad':sad, 'Angry':ang})
df = df.sort_values(['image'])
df.to_csv("results.csv", index=False)


# In[ ]:




