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

def Pycoral_Edgetpu():
    # Specify the TensorFlow model, labels, and image
    #script_dir = pathlib.Path(__file__).parent.absolute()
    script_dir = r'/home/ali/Desktop/GANomaly-tf2/export_model'
    model_file = os.path.join(script_dir, 'G-uint8-20221104_edgetpu.tflite')
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
import platform
import subprocess
import warnings
from pathlib import Path
import cv2
import numpy as np
def detect(w,tflite=False,edgetpu=True):
    if tflite or edgetpu:# https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
        try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
            from tflite_runtime.interpreter import Interpreter, load_delegate
            print('try successful')
        except ImportError:
            print('ImportError')
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
        if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
            print(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
            
            delegate = {
                'Linux': 'libedgetpu.so.1',
                'Darwin': 'libedgetpu.1.dylib',
                'Windows': 'edgetpu.dll'}[platform.system()]
            interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            
            
            # Initialize the TF interpreter
            #print('Start interpreter')
            #interpreter = edgetpu.make_interpreter(w)
            #print('End interpreter')
            
        else:  # TFLite
            print(f'Loading {w} for TensorFlow Lite inference...')
            interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs 
        print('input details : \n{}'.format(input_details))
        print('output details : \n{}'.format(output_details))
    return interpreter

def detect_image(w, im, interpreter=None, tflite=False,edgetpu=True):
    INFER=False
    ONLY_DETECT_ONE_IMAGE=True
    if interpreter is None:
        print('interpreter is None, get interpreter now')
        interpreter = detect(w,tflite,edgetpu)
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs 
        #print('input details : \n{}'.format(input_details))
        #print('output details : \n{}'.format(output_details))
    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs 
    '''
    if tflite or edgetpu:# https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
        try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
            from tflite_runtime.interpreter import Interpreter, load_delegate
            #print('try successful')
        except ImportError:
            #print('ImportError')
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
        if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
            #print(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
            
            #delegate = {
                #'Linux': 'libedgetpu.so.1',
                #'Darwin': 'libedgetpu.1.dylib',
                #'Windows': 'edgetpu.dll'}[platform.system()]
            
            #interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            
            # Initialize the TF interpreter
            print('Start interpreter')
            interpreter = edgetpu.make_interpreter(w)
            print('End interpreter')
            
            
        else:  # TFLite
            #print(f'Loading {w} for TensorFlow Lite inference...')
            interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs 
        print('input details : \n{}'.format(input_details))
        print('output details : \n{}'.format(output_details))
       ''' 
    #import tensorflow as tf
    from PIL import Image
    from matplotlib import pyplot as plt
    # Lite or Edge TPU

    
    if INFER:
        input_img = im
        #im = tf.transpose(im, perm=[0,1,2,3])
        #im = tf.squeeze(im)
        #plt.imshow(im)
        #plt.show()
    elif ONLY_DETECT_ONE_IMAGE:
        im = cv2.imread(im)
        im = cv2.resize(im, (64, 64))
        
        #input_img = im
        #cv2.imshow('ori_image',im)
        #cv2.imwrite('ori_image.jpg',im)
        #cv2.waitKey(10)
        
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #im = im/255.0
    #im = (im).astype('int32')
    #image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    #img = img[np.newaxis, ...].astype(np.float32)
    #print("calibration image {}".format(img[i]))
    #img = img / 255.0
    
    #im = Image.fromarray((im * 255).astype('uint8'))
    im = im[np.newaxis, ...].astype(np.float32)
    print('im : {}'.format(im.shape))
    input_img = im
    im = im/255.0
    #im = tf.expand_dims(im, axis=0)
    #im = im.numpy()
    
    #print('im:{}'.format(im.shape))
    #print('im: {}'.format(im))
    input = input_details[0]
    int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model (np.uint8)
    #int32 = input['dtype'] == np.int32  # is TFLite quantized uint8 model (np.uint8)
    #print('input[dtype] : {}'.format(input['dtype']))
    if int8:
        #print('is TFLite quantized uint8 model')
        scale, zero_point = input['quantization']
        im = (im / scale + zero_point).astype(np.uint8)  # de-scale
        #im = im.astype(np.uint8)
        print('after de-scale {}'.format(im))
    interpreter.set_tensor(input['index'], im)
    interpreter.invoke()
    y = []
    gen_img = None
    for output in output_details:
        x = interpreter.get_tensor(output['index'])
        #print(x.shape)
        #print(x)
        if x.shape[1]==64:
            #print('get out images')
            
            scale, zero_point = output['quantization']
            
            x = (x.astype(np.float32)-zero_point) * scale  # re-scale
            #x = x.astype(np.float32)
            #x = tf.squeeze(x)
            #x = x.numpy()
            gen_img = x*255
            
            gen_img = np.squeeze(gen_img)
            #print('after squeeze & numpy x : {}'.format(x))
            cv2.imshow('out_image',gen_img)
            cv2.imwrite('out_image.jpg',gen_img)
            cv2.waitKey(10)
            #gen_img = renormalize(gen_img)
            #gen_img = tf.transpose(gen_img, perm=[0,1,2])
            #plt.imshow(gen_img)
            #plt.show()
        if int8:
            scale, zero_point = output['quantization']
            x = (x.astype(np.float32)-zero_point) * scale  # re-scale
            #x = x.astype(np.float32)
            #gen_img = tf.squeeze(gen_img)
            #gen_img = gen_img.numpy()
        y.append(x)
    y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
    #gen_img = y[0]
    print('input image : {}'.format(input_img))
    print('input image : {}'.format(input_img.shape))
    print('gen_img : {}'.format(gen_img))
    print('gen_img : {}'.format(gen_img.shape))
    latent_i = y[0]
    latent_o = y[1]
    _g_loss = g_loss(input_img, gen_img, latent_i, latent_o)
    #_g_loss = 888
    print('g_loss : {}'.format(_g_loss))
    #print(y)
    return _g_loss, gen_img

def g_loss(input_img, gen_img, latent_i, latent_o):
    
    def l1_loss(A,B):
        return np.mean((abs(A-B)).flatten())
    def l2_loss(A,B):
        return np.mean(sqrt((A-B)*(A-B)).flatten())
    # tf loss
    #l2_loss = tf.keras.losses.MeanSquaredError()
    #l1_loss = tf.keras.losses. MeanAbsoluteError()
    #bce_loss = tf.keras.losses.BinaryCrossentropy()
    
    # adversarial loss (use feature matching)
    #l_adv = l2_loss
    # contextual loss
    l_con = l1_loss
    # Encoder loss
    l_enc = l2_loss
    # discriminator loss
    #l_bce = bce_loss
    
    #err_g_adv = l_adv(feat_real, feat_fake)
    err_g_con = l_con(input_img, gen_img)
    #err_g_enc = l_enc(latent_i, latent_o)
    err_g_enc = 0
    g_loss = err_g_con * 50 + \
             err_g_enc * 1
    return g_loss

if __name__=="__main__":
    PYCORAL = False
    DETECT = False
    DETECT_IMAGE = True
    if DETECT:
        w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        detect(w,tflite=False,edgetpu=True)
    if PYCORAL:
        Pycoral_Edgetpu()
        
    if DETECT_IMAGE:
        
        im = r'/home/ali/Desktop/factory_data/crops_1cls/line/ori_video_ver2121.jpg'
        #im = r'/home/ali/Desktop/factory_data/crops_2cls_small/noline/ori_video_ver244.jpg'
        
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        #w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104.tflite'
        w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        loss, gen_image = detect_image(w, im, tflite=False,edgetpu=True)