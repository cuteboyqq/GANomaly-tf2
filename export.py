# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:55:29 2022

@author: User
"""

import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

def representative_dataset():
  for _ in range(100):
      #data = random.randint(0, 1)
      #yield [data]
      data = np.random.rand(32)*2
      yield [data.astype(np.float32)]


def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

infer_data_dir = r'C:\factory_data\2022-08-26\f_384_2min\crops_1cls'
shuffle = False
img_height = 64
img_width = 64
batch_size_ = 64

infer_dataset = tf.keras.utils.image_dataset_from_directory(
  infer_data_dir,
  #validation_split=0.1,
  #subset="validation",
  shuffle=shuffle,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size_)

infer_dataset = infer_dataset.map(process)


saved_model_dir = r'C:\GitHub_Code\cuteboyqq\GANomaly\GANomaly-tf2\ckpt\G'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory

import os
os.makedirs('./export_model',exist_ok=True)

quantize_mode = 'float32'

if quantize_mode == 'int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    # Save the model.
    with open(r'./export_model/G-int8.tflite', 'wb') as f:
      f.write(tflite_model)
elif quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    with open(r'./export_model/G-float16.tflite', 'wb') as f:
      f.write(tflite_model)
elif quantize_mode == 'float32':
    tflite_model = converter.convert()
    # Save the model.
    with open(r'./export_model/G-float32.tflite', 'wb') as f:
      f.write(tflite_model)
else:
    print('[ERROR] No suuch quatization mode : {}'.format(quantize_mode))
