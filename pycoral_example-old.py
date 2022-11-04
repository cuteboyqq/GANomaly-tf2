#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:46:08 2022

@author: ali
"""

import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'G-uint8-new_edgetpu.tflite')
label_file = os.path.join(script_dir, 'imagenet_labels.txt')
image_file = os.path.join(script_dir, 'parrot.jpg')

# Initialize the TF interpreter
print('Start interpreter')
interpreter = edgetpu.make_interpreter(model_file)
print('End interpreter')

print('Start allocate_tensors')
interpreter.allocate_tensors()
print('End allocate_tensors')

input_details = interpreter.get_input_details()  # inputs
output_details = interpreter.get_output_details()  # outputs 
print('input details : \n{}'.format(input_details))
print('output details : \n{}'.format(output_details))
# Resize the image
#size = common.input_size(interpreter)
#image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
#common.set_input(interpreter, image)
#interpreter.invoke()
#classes = classify.get_classes(interpreter, top_k=1)

# Print the result
#labels = dataset.read_label_file(label_file)
#for c in classes:
  #print('%s: %.5f' % (labels.get(c.id, c.id), c.score))