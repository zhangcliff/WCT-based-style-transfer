#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:23:19 2018

@author: zwx
"""
import pandas as pd
import numpy as np
import tensorflow as tf
#from skimage.io import imread,imshow,imsave
from keras.preprocessing import image
from glob import glob
from tqdm import tqdm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

images_path='train2014/'
images_list=glob(images_path+'*.jpg')
num=len(images_list)

tfrecords_filename =  'tfrecords/train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

img_width=224
img_height=224

for image_path in tqdm(images_list):
    img = image.load_img(image_path,target_size=[224,224,3])
    img = image.img_to_array(img).astype(np.uint8)
    img_raw = img.tostring()
    feature = {'image_raw':_bytes_feature(img_raw)}
    example=tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())    
       
writer.close()
print 'record done'

