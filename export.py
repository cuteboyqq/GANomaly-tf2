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

import cv2
def representative_data_gen(fimage,input_size):
  #fimage = open(FLAGS.dataset).read().split()
  for input_value in range(10):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      #image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
      img_in = original_image[np.newaxis, ...].astype(np.float32)
      print("calibration image {}".format(fimage[input_value]))
      yield [img_in]
    else:
      continue


def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

infer_data_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_1cls'
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


import os
os.makedirs('./export_model',exist_ok=True)

def convert_tflite_oldversion(saved_model_dir):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
    quantize_mode = 'int8'
    export_tflite_model = False
    if export_tflite_model:
        if quantize_mode == 'int8':
            print('Start convert to int8 tflite model')
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.allow_custom_ops = True
            tflite_model = converter.convert()
            # Save the model.
            with open(r'./export_model/G-int8.tflite', 'wb') as f:
              f.write(tflite_model)
        elif quantize_mode == 'float16':
            print('Start convert to float16 tflite model')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.allow_custom_ops = True
            tflite_model = converter.convert()
            with open(r'./export_model/G-float16.tflite', 'wb') as f:
              f.write(tflite_model)
        elif quantize_mode == 'float32':
            print('Start convert to float32 tflite model')
            tflite_model = converter.convert()
            # Save the model.
            with open(r'./export_model/G-float32.tflite', 'wb') as f:
              f.write(tflite_model)
        else:
            print('[ERROR] No suuch quatization mode : {}'.format(quantize_mode))
    
'''See official document at https://coral.ai/docs/edgetpu/compiler/#system-requirements'''

'''
==================
Edge TPU Compiler
===================
    The Edge TPU Compiler (edgetpu_compiler) is a command line tool that compiles a TensorFlow Lite model (.tflite file) 
    into a file that's compatible with the Edge TPU. This page describes how to use the compiler and a bit about how it works.
    
    Before using the compiler, be sure you have a model that's compatible with the Edge TPU. For compatibility details, read
    https://coral.ai/docs/edgetpu/models-intro/#compatibility-overview
==========================
System requirements
==========================
    The Edge TPU Compiler can be run on any modern Debian-based Linux system. Specifically, you must have the following:
    
    64-bit version of Debian 6.0 or higher, or any derivative thereof (such as Ubuntu 10.0+)
    x86-64 system architecture
If your system does not meet these requirements, try our web-based compiler using Google Colab.
===============
Download
===============
    You can install the compiler on your Linux system with the following commands:
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    
    sudo apt-get update
    
    sudo apt-get install edgetpu-compiler
========
Usage
=========
    edgetpu_compiler [options] model...
'''
import platform
import subprocess
import warnings
from pathlib import Path
def export_edgetpu(file):
    file = Path(file)
    # YOLOv5 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    cmd = 'edgetpu_compiler --version'
    help_url = 'https://coral.ai/docs/edgetpu/compiler/'
    assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
    if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
        print(f'\n export requires Edge TPU compiler. Attempting install from {help_url}')
        sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
        for c in (
                'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
            subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    print(f'\n starting export with Edge TPU compiler {ver}...')
    #f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
    f = str(file).replace('.tflite', '-uint8_edgetpu.tflite')  # Edge TPU model
    #f_tfl = str(file).replace('-int8.tflite', '-int8.tflite')  # TFLite model
    f_tfl = str(file)
    #f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model
    #file_dir = '/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-Pytorch/model/img64_nz100'
    cmd = f"edgetpu_compiler -s -d -k 10 --out_dir {file.parent} {f_tfl}"
    #cmd = f"edgetpu_compiler {f_tfl}"
    #subprocess.run(cmd.split(), check=True)
    subprocess.run(cmd.split(), check=True)
    return f, None

import glob
def rep_data_gen():
    root = "/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_2cls_cyclegan"
    BATCH_SIZE = 1
    a = []
    for i in range(160):
        #inst = anns[i]
        #file_name = inst['filename']
        file_name = sorted(glob.glob(os.path.join(root, "B") + "/*.*"))
        img = cv2.imread(file_name[i])
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        #img = img[np.newaxis, ...].astype(np.float32)
        #print("calibration image {}".format(img[i]))
        #img = img / 255.0
        img = img.astype(np.float32)
        #yield [img]
        a.append(img)
    a = np.array(a)
    print(a.shape) # a is np array of 160 3D images
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(BATCH_SIZE):
        print(i)
        yield [i]


def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 32, 32, 3)
      yield [data.astype(np.float32)]
      #yield [data.astype(np.uint8)]

'''code example is at https://www.tensorflow.org/lite/performance/post_training_quantization
        find the samw error issues https://github.com/google-coral/edgetpu/issues/453
        wrong data type error issues https://stackoverflow.com/questions/52530724/python-tensorflow-lite-error-cannot-set-tensor-got-tensor-of-type-1-but-expecte
    cthis code convert no error
    https://stackoverflow.com/questions/57877959/what-is-the-correct-way-to-create-representative-dataset-for-tfliteconverter
    '''
def export_tflite(saved_model_dir, int8=True):
    # YOLOv5 TensorFlow Lite export
    import tensorflow as tf

    #LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    #batch_size, ch, *imgsz = list(im.shape)  # BCHW
    #f = str(file).replace('.pt', '-fp16.tflite')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    #converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    #converter.target_spec.supported_types = [tf.float16] 
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    if int8:
        converter.experimental_new_converter = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.representative_dataset = representative_dataset
        converter.representative_dataset = rep_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8 successul
        converter.inference_output_type = tf.int8  # or tf.uint8 successful
    else: # uint8
        converter.experimental_new_converter = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.representative_dataset = representative_dataset
        converter.representative_dataset = rep_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.uint8 successul
        converter.inference_output_type = tf.uint8  # or tf.uint8 successful
    #if nms or agnostic_nms:
        #converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_quant_model = converter.convert()
    f=''
    if int8:
        f='./export_model/G-int8-new.tflite'
    else:
        f='./export_model/G-uint8-new.tflite'
    open(f, "wb").write(tflite_quant_model)
    
    import numpy as np
    import tensorflow as tf
    
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=f)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('interpreter.get_input_details()')
    print(interpreter.get_input_details())
    print('interpreter.get_output_details()')
    print(interpreter.get_output_details())
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    if int8:
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.int8)
    else:
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
        
    return f, None


def detect(w,tflite=False,edgetpu=True):
    if tflite or edgetpu:# https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
        try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
            from tflite_runtime.interpreter import Interpreter, load_delegate
            #print('try successful')
        except ImportError:
            #print('ImportError')
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
        if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
            print(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
            delegate = {
                'Linux': 'libedgetpu.so.1',
                'Darwin': 'libedgetpu.1.dylib',
                'Windows': 'edgetpu.dll'}[platform.system()]
            interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
        else:  # TFLite
            print(f'Loading {w} for TensorFlow Lite inference...')
            interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs 
        print('input details : \n{}'.format(input_details))
        print('output details : \n{}'.format(output_details))


def g_loss(input_img, gen_img, latent_i, latent_o):
    # loss
    l2_loss = tf.keras.losses.MeanSquaredError()
    l1_loss = tf.keras.losses.MeanAbsoluteError()
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

def detect_image(w, im, tflite=False,edgetpu=True):
    if tflite or edgetpu:# https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
        try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
            from tflite_runtime.interpreter import Interpreter, load_delegate
            #print('try successful')
        except ImportError:
            #print('ImportError')
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
        if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
            print(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
            delegate = {
                'Linux': 'libedgetpu.so.1',
                'Darwin': 'libedgetpu.1.dylib',
                'Windows': 'edgetpu.dll'}[platform.system()]
            interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
        else:  # TFLite
            print(f'Loading {w} for TensorFlow Lite inference...')
            interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs 
        print('input details : \n{}'.format(input_details))
        print('output details : \n{}'.format(output_details))
        
        import tensorflow as tf
        from PIL import Image
        # Lite or Edge TPU
        im = cv2.imread(im)
        im = cv2.resize(im, (32, 32))
        cv2.imshow('ori_image',im)
        cv2.imwrite('ori_image.jpg',im)
        cv2.waitKey(100)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = im/255.0
        im = (im).astype('int32')
        #image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        #img = img[np.newaxis, ...].astype(np.float32)
        #print("calibration image {}".format(img[i]))
        #img = img / 255.0
        
        #im = Image.fromarray((im * 255).astype('uint8'))
        im = tf.expand_dims(im, axis=0)
        im = im.cpu().numpy()
        input_img = im
        #print('im:{}'.format(im.shape))
        #print('im: {}'.format(im))
        input = input_details[0]
        int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model (np.uint8)
        #int32 = input['dtype'] == np.int32  # is TFLite quantized uint8 model (np.uint8)
        print('input[dtype] : {}'.format(input['dtype']))
        if int8:
            print('is TFLite quantized uint8 model')
            scale, zero_point = input['quantization']
            im = (im * scale + zero_point).astype(np.uint8)  # de-scale
        interpreter.set_tensor(input['index'], im)
        interpreter.invoke()
        y = []
        for output in output_details:
            x = interpreter.get_tensor(output['index'])
            print(x.shape)
            print(x)
            if x.shape[1]==32:
                print('get out images')
                
                scale, zero_point = output['quantization']
                
                x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                x = tf.squeeze(x)
                x = x.numpy()
                
                cv2.imshow('out_image',x)
                cv2.imwrite('out_image.jpg',x)
                cv2.waitKey(100)
            if int8:
                scale, zero_point = output['quantization']
                gen_img = (x.astype(np.float32) - zero_point) * scale  # re-scale
                
                
            y.append(x)
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        gen_img = y[0]
        latent_i = y[1]
        latent_o = y[2]
        _g_loss = g_loss(input_img, gen_img, latent_i, latent_o)
        print('g_loss : {}'.format(_g_loss))
        #print(y)
        return y

if __name__=="__main__":
    saved_model_dir = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/ckpt/G'
    
    INT8=False #True
    EDGETPU=False #True
    DETECT=False
    DETECT_IMAGE=True
    print('convert int8.tflite :{}\nconvert edgetpu.tflite:{}\ndetect:{}\ndetect_image:{}'.format(INT8,EDGETPU,DETECT,DETECT_IMAGE))
    
    if INT8:
        saved_model_dir = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/ckpt/G'
        export_tflite(saved_model_dir, int8=False)
    
    if EDGETPU:
        f = export_edgetpu(r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite')
        
    if DETECT:
        w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        detect(w,tflite=False,edgetpu=True)
    if DETECT_IMAGE:
        im = r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_1cls/line/ori_video_ver246.jpg'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        y = detect_image(w, im, tflite=True,edgetpu=False)