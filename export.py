# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:55:29 2022

@author: User
"""

import tensorflow as tf

saved_model_dir = r'C:\GitHub_Code\cuteboyqq\GANomaly\GANomaly-tf2\ckpt\G'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('G.tflite', 'wb') as f:
  f.write(tflite_model)