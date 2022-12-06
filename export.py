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
#=====================================================================
'''
import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
'''
#=====================================================================

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
      img_in = original_image[np.newaxis, ...].astype(np.int8)
      print("calibration image {}".format(fimage[input_value]))
      yield [img_in]
    else:
      continue


def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

infer_data_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/2022-11-24/crops_line'
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
def export_edgetpu(file,file2=None,two_models=False):
    file = Path(file)
    if two_models:
        file2 = Path(file2)
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
    f = str(file).replace('.tflite', '-int8_edgetpu.tflite')  # Edge TPU model
    #f_tfl = str(file).replace('-int8.tflite', '-int8.tflite')  # TFLite model
    f_tfl = str(file)
    
    if two_models:
        f2 = str(file2).replace('.tflite', '-int8_edgetpu.tflite')  # Edge TPU model
        #f_tfl = str(file).replace('-int8.tflite', '-int8.tflite')  # TFLite model
        f2_tfl = str(file2)
    #f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model
    #file_dir = '/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-Pytorch/model/img64_nz100'
    if two_models:
        cmd = f"edgetpu_compiler {file} {file2}"
    else:
        cmd = f"edgetpu_compiler -s -d -k 10 --out_dir {file.parent} {f_tfl}"
    #cmd = f"edgetpu_compiler --out_dir {file.parent} {f_tfl}"
    #cmd = f"edgetpu_compiler {f_tfl}"
    #subprocess.run(cmd.split(), check=True)
    subprocess.run(cmd.split(), check=True)
    return f, None

import glob
def rep_data_gen():
    root = "/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/2022-11-24/crops_line"
    BATCH_SIZE = 1
    a = []
    file_name = sorted(glob.glob(os.path.join(root, "line") + "/*.*"))
    for i in range(7800):
        #inst = anns[i]
        #file_name = inst['filename']
        #print(file_name[i])
        img = cv2.imread(file_name[i])
        img = cv2.resize(img, (32, 32))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        
        img = np.transpose(img, [1, 2, 0])
        img = np.expand_dims(img, axis=0).astype(np.float32)
        #img = img.astype(np.float32)
        img /= 255
        #yield [im]
        #----------------------------------------------------------------------------------------------------------
        #Alister train GANomaly do not convert BGR2RGB, so note below img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #----------------------------------------------------------------------------------------------------------
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        #img = img[np.newaxis, ...].astype(np.float32)
        #print("calibration image {}".format(img[i]))
        #=============================
        #2022-11-04 add normalization
        #=============================
        #img = img / 255.0
        #img = img.astype(np.float32) 
        #yield [img]
        a.append(img)
    a = np.array(a)
    #print(a.shape) # a is np array of 160 3D images
    #img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    #for i in img.take(BATCH_SIZE):
        #print(i)
        #yield [i]
    for i in a:
        #print(i)
        yield[i]


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
    
    quintize infomation : https://zhuanlan.zhihu.com/p/79744430
    '''
def export_tflite(saved_model_dir, int8=True, name='20221111',experimental_new_converter=True):
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
    converter.representative_dataset = rep_data_gen
    if int8:
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.representative_dataset = representative_dataset
        converter.representative_dataset = rep_data_gen
        #converter.target_spec.supported_types = []
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8 successul
        converter.inference_output_type = tf.int8  # or tf.uint8 successful
        converter.experimental_new_converter = experimental_new_converter #unable to convert to edgetpu.tflite when True
    else: # uint8
        converter.experimental_new_converter = experimental_new_converter #unable to convert to edgetpu.tflite when True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.representative_dataset = representative_dataset
        converter.representative_dataset = rep_data_gen
        #converter.target_spec.supported_types = []
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.uint8 successul
        converter.inference_output_type = tf.uint8  # or tf.uint8 successful
    #if nms or agnostic_nms:
        #converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_quant_model = converter.convert()
    f=''
    if int8:
        f='./export_model/'+ name + '-G-int8.tflite'
    else:
        f='./export_model/' + name + '-G-uint8.tflite'
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
    return interpreter


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

def detect_image(w, im, interpreter=None, tflite=False,edgetpu=True):
    INFER=True
    ONLY_DETECT_ONE_IMAGE=False
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
    import tensorflow as tf
    from PIL import Image
    from matplotlib import pyplot as plt
    # Lite or Edge TPU

    
    if INFER:
        input_img = im
        #im = tf.transpose(im, perm=[0,1,2,3])
        im = tf.squeeze(im)
        #plt.imshow(im)
        #plt.show()
    elif ONLY_DETECT_ONE_IMAGE:
        im = cv2.imread(im)
        im = cv2.resize(im, (128, 128))
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
    im = tf.expand_dims(im, axis=0)
    im = im.numpy()
    
    #print('im:{}'.format(im.shape))
    #print('im: {}'.format(im))
    input = input_details[0]
    int8 = input['dtype'] == np.int8  # is TFLite quantized uint8 model (np.uint8)
    #int32 = input['dtype'] == np.int32  # is TFLite quantized uint8 model (np.uint8)
    #print('input[dtype] : {}'.format(input['dtype']))
    if int8:
        #print('is TFLite quantized uint8 model')
        scale, zero_point = input['quantization']
        im = (im / scale + zero_point).astype(np.int8)  # de-scale
        #print('after de-scale {}'.format(im))
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
            x = tf.squeeze(x)
            x = x.numpy()
            gen_img = x
            #print('after squeeze & numpy x : {}'.format(x))
            #cv2.imshow('out_image',gen_img)
            #cv2.imwrite('out_image.jpg',gen_img)
            #cv2.waitKey(10)
            #gen_img = renormalize(gen_img)
            #gen_img = tf.transpose(gen_img, perm=[0,1,2])
            #plt.imshow(gen_img)
            #plt.show()
        if int8:
            scale, zero_point = output['quantization']
            x = (x.astype(np.float32)-zero_point) * scale  # re-scale
            #gen_img = tf.squeeze(gen_img)
            #gen_img = gen_img.numpy()
        y.append(x)
    y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
    #gen_img = y[0]
    #print('input image : {}'.format(input_img))
    #print('input image : {}'.format(input_img.shape))
    #print('gen_img : {}'.format(gen_img))
    #print('gen_img : {}'.format(gen_img.shape))
    latent_i = y[0]
    latent_o = y[1]
    _g_loss = g_loss(input_img, gen_img, latent_i, latent_o)
    #print('g_loss : {}'.format(_g_loss))
    #print(y)
    return _g_loss, gen_img
    
def infer(test_dataset, w, SHOW_MAX_NUM, show_img, data_type, tflite, edgetpu):
    interpreter = detect(w,tflite,edgetpu)
    show_num = 0
    
    loss_list = []
    dataiter = iter(test_dataset)
    #for step, (images, y_batch_train) in enumerate(test_dataset):
    cnt=1
    os.makedirs('./runs/detect/tflite_model',exist_ok=True)
    while(show_num < SHOW_MAX_NUM):
        images, labels = dataiter.next()
        #latent_i, fake_img, latent_o = self.G(images)
        #self.input = images
        
        #self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
        #self.pred_real, self.feat_real = self.D(self.input)
        #self.pred_fake, self.feat_fake = self.D(self.gen_img)
        #g_loss = self.g_loss()
        
        g_loss,fake_img = detect_image(w, images, interpreter, tflite=True,edgetpu=False)
        
        
        #g_loss = 0.0
        #print("input")
        #print(self.input)
        #print("gen_img")
        #print(self.gen_img)
        #images = renormalize(images)
        #fake_img = renormalize(fake_img)
        #fake_img = self.gen_img
        #images = images.cpu().numpy()
        #fake_img = fake_img.cpu().numpy()
        #fake_img = self.gen_img
        #print(fake_img.shape)
        #print(images.shape)
        if show_img:
            #plt = self.plot_images(images,fake_img)
            if data_type=='normal':
                file_name = 'infer_normal' + str(cnt) + '.jpg'
            else:
                file_name = 'infer_abnormal' + str(cnt) + '.jpg'
            #file_path = os.path.join('./runs/detect',file_name)
            #plt.savefig(file_path)
            cnt+=1
        if data_type=='normal':
            print('{} normal: {}'.format(show_num,g_loss.numpy()))
        else:
            print('{} abnormal: {}'.format(show_num,g_loss.numpy()))
        loss_list.append(g_loss.numpy())
        show_num+=1
        #if show_num%20==0:
            #print(show_num)
    return loss_list
    

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def renormalize(tensor):
    minFrom= tf.math.reduce_min(tensor)
    maxFrom= tf.math.reduce_max(tensor)
    minTo = 0
    maxTo = 1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

def plot_loss_distribution(SHOW_MAX_NUM,positive_loss,defeat_loss):
    # Importing packages
    import matplotlib.pyplot as plt2
    # Define data values
    x = [i for i in range(SHOW_MAX_NUM)]
    y = positive_loss
    z = defeat_loss
    print(x)
    print(positive_loss)
    print(defeat_loss)
    # Plot a simple line chart
    #plt2.plot(x, y)
    # Plot another line on the same chart/graph
    #plt2.plot(x, z)
    plt2.scatter(x,y,s=1)
    plt2.scatter(x,z,s=1) 
    os.makedirs('./runs/detect/tflite_model-20221110',exist_ok=True)
    file_path = os.path.join('./runs/detect/tflite_model-20221110','loss_distribution.jpg')
    plt2.savefig(file_path)
    plt2.show()

if __name__=="__main__":
    saved_model_dir = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/ckpt-128-nz400-ndf64-ngf64/G'
    
    INT8=False #True
    EDGETPU=True#True
    DETECT=False
    DETECT_IMAGE=False
    INFER = False
    print('convert int8.tflite :{}\nconvert edgetpu.tflite:{}\ndetect:{}\ndetect_image:{}\ninfer:{}'.format(INT8,EDGETPU,DETECT,DETECT_IMAGE,INFER))
    
    if INT8:
        saved_model_dir = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/ckpt/G'
        export_tflite(saved_model_dir, int8=True, name='ckpt-32-nz100-ndf64-ngf64-20221205-prelu-upsample',experimental_new_converter=True)
    
    if EDGETPU:
        tflite_model_path = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/ckpt-32-nz100-ndf64-ngf64-20221205-prelu-upsample-G-int8.tflite'
        tflite_model2_path = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/best-int8.tflite'
        f = export_edgetpu(tflite_model_path,tflite_model2_path,True)
        
    if DETECT:
        w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        detect(w,tflite=False,edgetpu=True)
    if DETECT_IMAGE:
        im = r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_1cls/line/ori_video_ver246.jpg'
        #im = r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/noline/ori_video_ver244.jpg'
        
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-20221104.tflite'
        y = detect_image(w, im, tflite=True,edgetpu=False)
        
        
    if INFER:
        test_data_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_line'
        abnormal_test_data_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_noline'
        (img_height, img_width) = (64,64)
        batch_size_ = 1
        shuffle = False
        
        test_dataset = tf.keras.utils.image_dataset_from_directory(
          test_data_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
        
        test_dataset = test_dataset.map(process)
        
        
        test_dataset_abnormal = tf.keras.utils.image_dataset_from_directory(
          abnormal_test_data_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
        
        test_dataset_abnormal = test_dataset_abnormal.map(process)
        
        w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-64nz200-20221110.tflite'
        
        SHOW_MAX_NUM = 1800
        
        show_img = False
        
        line_data_type = 'normal'
        noline_data_type = 'abnormal'
        
        #line_loss = infer(test_dataset, w, SHOW_MAX_NUM, show_img, line_data_type,tflite=True,edgetpu=False)
        
        #noline_loss = infer(test_dataset_abnormal, w, SHOW_MAX_NUM, show_img, noline_data_type,tflite=True,edgetpu=False)
        
        #line_loss = [0.5650214, 2.3491344, 0.69566053, 0.71863496, 0.579496, 0.65742105, 0.58306456, 0.58523935, 0.54923475, 0.55835944, 0.55193174, 0.52843744, 0.54969496, 1.5579801, 0.5451242, 0.5798215, 0.5540409, 0.61214614, 0.4705852, 0.55790067, 0.4896323, 0.5853224, 0.49816343, 0.57353055, 0.6882696, 0.52991354, 0.5840684, 0.5635927, 0.60677016, 0.5624922, 0.6046249, 0.5648086, 0.68368435, 0.55768627, 0.6608205, 1.4619833, 0.613113, 0.63328385, 0.6424586, 0.67471063, 0.6917342, 0.6864369, 0.5287891, 0.72114503, 0.5298847, 0.7298554, 0.70623493, 0.49191704, 0.71425056, 0.50450015, 0.7308067, 0.544648, 0.7674603, 0.52629524, 0.77319556, 0.49243402, 0.785572, 1.5098771, 0.51055276, 0.7786599, 0.58397335, 0.7804005, 0.53428274, 0.79534614, 0.5740105, 0.79278594, 0.6297307, 0.79143727, 0.71381515, 0.6232612, 0.8195745, 0.63422215, 0.82685816, 0.6137264, 0.8827746, 0.6394875, 0.91532284, 0.7103795, 0.97049016, 1.4564924, 0.7391544, 0.94942003, 0.75423783, 0.9721614, 0.7987396, 0.97340757, 0.8256844, 1.042261, 0.8192703, 1.1085831, 0.7418083, 0.8558225, 1.0387577, 0.91265833, 1.0089612, 0.7688968, 0.9787159, 0.73191464, 0.94286996, 0.68838984, 0.9230475, 1.4265924, 0.6889716, 0.90220815, 0.6633534, 0.92116857, 0.6920129, 0.96464336, 0.66664666, 0.95941496, 0.7010184, 0.9426497, 0.9748722, 0.7411781, 0.68744195, 0.90543085, 0.7755904, 0.91699225, 0.66020924, 0.98314285, 0.6676475, 0.867286, 0.68449694, 0.9681741, 1.4235162, 0.64761883, 0.9136874, 0.68610734, 0.8763615, 0.6551388, 0.9727274, 0.6818045, 0.92655206, 0.6520839, 0.8504361, 0.72695714, 0.7764994, 0.8863229, 0.6545628, 0.93776864, 0.7022826, 0.9129123, 0.69955987, 0.91017187, 0.6896497, 0.93672854, 1.4011338, 0.72974646, 0.8490659, 0.7365477, 0.8981095, 0.69046414, 0.9199714, 0.68323344, 1.0222235, 0.6898201, 0.7992392, 0.7210897, 0.6933414, 0.82174176, 0.71026766, 0.98695046, 0.7366902, 0.993732, 0.73456275, 0.89132524, 0.74505824, 1.0729072, 1.3403316, 0.7341829, 1.049393, 0.7418576, 0.99635476, 0.7373564, 0.9442432, 0.73589164, 1.0603355, 0.76373005, 1.0695106, 0.7269886, 0.7440204, 1.0617807, 0.739194, 1.0395894, 0.73846316, 1.0217553, 0.7367395, 0.9101324, 0.7182463, 0.9755921, 1.332875, 0.7557557, 0.941684, 0.7180864, 1.0056939, 0.7549544, 0.9510466, 0.7517061, 0.92735857, 0.75519097, 0.85558367, 0.78959763, 0.73460275, 0.90200055, 0.76623195, 0.8430274, 0.73314977, 0.9077214, 0.75777924, 0.8702818, 0.75817585, 0.84995, 1.3051404, 0.74724483, 0.75595415, 0.73885226, 0.7446619, 0.7401221, 0.7164187, 0.7475465, 0.71408844, 0.7778598, 0.754039, 2.223426, 0.7165631, 0.775594, 0.76334506, 0.93962157, 0.7263856, 1.0504725, 0.7574981, 0.82262504, 1.1030955, 0.73205805, 1.1520025, 1.3352337, 0.7173982, 0.7111388, 0.7114859, 0.7072573, 0.70051056, 0.6973118, 0.6816718, 0.6831128, 0.7005859, 0.6922059, 0.7296517, 0.6939433, 0.7164303, 0.6676675, 0.7027561, 0.64608437, 0.6816183, 0.6686382, 0.6744185, 0.69671386, 0.60844964, 1.2836486, 0.594649, 0.5979578, 0.5704683, 0.566873, 0.60400385, 0.57551897, 0.5579716, 0.536335, 0.52914256, 0.5161001, 0.7081856, 0.53300476, 0.5095493, 0.54776275, 0.5133707, 0.5101222, 0.48162732, 0.46265042, 0.46400827, 0.4765736, 0.5191056, 1.3446625, 0.48395902, 0.4915077, 0.4825593, 0.44994617, 0.4555656, 0.4807117, 0.4794308, 0.48420513, 0.48374376, 0.47936374, 0.6846946, 0.46388394, 0.63429457, 0.5098639, 0.61441094, 0.51624095, 0.67244023, 0.46856338, 0.69869024, 0.5080808, 0.7105575, 1.2709578, 0.4723872, 0.6841154, 0.4912544, 0.6508096, 0.49057263, 0.6661287, 0.46465275, 0.60215294, 0.46153125, 0.60611004, 0.7293311, 0.46150434, 0.67152333, 0.4557955, 0.6263772, 0.4819177, 0.58976793, 0.4539099, 0.5613739, 0.5020189, 0.6082321, 1.2642192, 0.48021126, 0.5823717, 0.46988633, 0.554726, 0.45808116, 0.6328828, 0.47796908, 0.5876698, 0.44023842, 0.59226173, 0.95009655, 0.68953764, 0.43139997, 0.5999211, 0.46426272, 0.5818334, 0.4727249, 0.57870126, 0.47952226, 0.6356436, 0.4618254, 0.5932357, 1.2761151, 0.43348527, 0.5949282, 0.43302062, 0.5478677, 0.45992833, 0.6742538, 0.46338615, 0.5570234, 0.45446438, 0.5629306, 0.7320216, 0.46506318, 0.5458527, 0.45711607, 0.6012847, 0.46555904, 0.50305575, 0.48752353, 0.4928737, 0.48753312, 0.51277727, 1.3525375, 0.5194906, 0.52338946, 0.55495745, 0.5053169, 0.553224, 0.5394199, 0.5040166, 0.5829331, 0.5233465, 0.56861645, 0.7255003, 0.5628772, 0.5570183, 0.55974835, 0.5705596, 0.5830572, 0.58006805, 0.6175591, 0.589643, 0.6399556, 0.65515757, 1.2615087, 0.6299354, 0.6447916, 0.62386066, 0.6493162, 0.63715476, 0.67385864, 0.6420822, 0.66149, 0.64447, 0.74970084, 0.7265892, 0.6356703, 0.72810656, 0.66405153, 0.7866929, 0.6524957, 0.8343909, 0.703101, 0.7916328, 0.74736017, 0.78827685, 1.2841623, 0.810896, 0.76764643, 0.6693374, 0.7435286, 0.63823426, 0.72547525, 0.5997485, 0.7257609, 0.6218286, 0.7164393, 0.693163, 0.63189554, 0.71798986, 0.6209628, 0.71474695, 0.6101278, 0.72663724, 0.58855826, 0.7036385, 0.60629034, 0.7080921, 1.2376262, 0.5841807, 0.71657825, 0.59170276, 0.727551, 0.5713413, 0.6816728, 0.62813365, 0.71631527, 0.66413, 0.74203444, 2.2009869, 0.7335636, 0.6168895, 0.6999377, 0.61304235, 0.60186523, 0.60038924, 0.63052905, 0.61182266, 0.7102762, 0.6490134, 0.6129288, 1.188696, 0.5979916, 0.6827527, 0.6060961, 0.676699, 0.5840472, 0.6473769, 0.59679884, 0.6968237, 0.6194554, 0.6787259, 0.72842216, 0.56982106, 0.6618463, 0.5873271, 0.66945285, 0.61898124, 0.56133074, 0.69714445, 0.65524757, 0.66934013, 0.6477901, 1.3282617, 0.68198806, 0.696598, 0.5780475, 0.68022597, 0.7041426, 0.70353866, 0.6647586, 0.56650335, 0.6447344, 0.57499176, 0.70373386, 0.6972505, 0.5542673, 0.71237373, 0.56367916, 0.69300205, 0.5054844, 0.68986386, 0.6336471, 0.4901113, 0.65070677, 1.3245897, 0.48755985, 0.6974307, 0.505098, 0.6512774, 0.4652027, 0.6450203, 0.5348242, 0.6525633, 0.48443535, 0.60104597, 0.67987967, 0.48069945, 0.63516676, 0.46442655, 0.6210619, 0.475444, 0.63193154, 0.45183435, 0.62254405, 0.44401637, 0.62905765, 1.312293, 0.43827733, 0.62782276, 0.43770355, 0.6147843, 0.4462099, 0.59351975, 0.43262634, 0.59100586, 0.45950097, 0.6128489, 0.6884206, 0.45810795, 0.6035641, 0.43493766, 0.6010829, 0.45754108, 0.5741545, 0.44683832, 0.5402168, 0.451456, 0.59628206, 1.2579775, 0.45529705, 0.5573087, 0.4482258, 0.5914942, 0.45042792, 0.55094874, 0.46730804, 0.55937254, 0.4897861, 0.5178827, 1.0624288, 0.6844877, 0.47820026, 0.46382704, 0.4620174, 0.48154575, 0.4444574, 0.4902306, 0.4494479, 0.4995351, 0.45372266, 0.4846402, 1.1848364, 0.46984032, 0.5054833, 0.47063705, 0.52269816, 0.48428854, 0.5619977, 0.4773187, 0.5369903, 0.47227368, 0.5615863, 0.6725482, 0.48496836, 0.6120124, 0.4996977, 0.569574, 0.48277205, 0.56262994, 0.48805457, 0.5526878, 0.49825293, 0.56403744, 1.1816734, 0.5139579, 0.5527851, 0.49836105, 0.55291325, 0.46170503, 0.58621657, 0.4748967, 0.4757921, 0.6118346, 0.59015155, 0.6901964, 0.46642172, 0.5935898, 0.4971079, 0.6756113, 0.4727332, 0.6241887, 0.46829796, 0.7124616, 0.47631916, 0.649843, 1.2403841, 0.47556716, 0.6847836, 0.47019076, 0.7861493, 0.46547377, 0.72676456, 0.47445047, 0.72891, 0.501625, 0.7777833, 0.6757629, 0.45416752, 0.77287954, 0.44834325, 0.80226743, 0.46178427, 0.8305263, 0.45105824, 0.7716963, 0.45479637, 0.7146842, 1.1949053, 0.4470947, 0.8308437, 0.44999808, 0.83959174, 0.52567536, 0.8028304, 0.46014357, 0.71589375, 0.46685693, 0.6761558, 0.68581134, 0.47733983, 0.8536544, 0.49589872, 0.7740307, 0.49466607, 0.7634415, 0.48451003, 0.9087483, 0.5190816, 0.86068326, 1.1561477, 0.49714902, 1.1964241, 0.49617052, 0.524869, 0.49051082, 0.494887, 0.47486117, 0.46403548, 0.47199443, 0.500635, 2.1806142, 0.66048175, 0.49400398, 0.47970274, 0.4916692, 0.4751177, 0.5170903, 0.46402574, 0.45180976, 0.4450689, 0.4365084, 0.47179312, 1.2028824, 0.46846172, 0.4672831, 0.47836262, 0.4818421, 0.47179675, 0.47412637, 0.47945416, 0.4841147, 0.46528456, 0.45627132, 0.6589305, 0.4620213, 0.4744307, 0.46558505, 0.49467602, 0.495935, 0.45960173, 0.48059317, 0.46052206, 0.5023392, 0.4643563, 1.1329926, 0.46702954, 0.5015796, 0.49593127, 0.48513427, 0.5244063, 0.49094948, 0.523864, 0.4797856, 0.47134534, 0.47770348, 0.6560828, 0.48794037, 0.48685855, 0.4621963, 0.5104687, 0.49202833, 0.49379525, 0.49561974, 0.5069556, 0.47489908, 0.5014335, 1.1542168, 0.49542394, 0.48788917, 0.5196068, 0.49106902, 0.49112913, 0.51104116, 0.49161556, 0.5040241, 0.47173426, 0.47442392, 0.6855802, 0.4761926, 0.49392796, 0.51145124, 0.50160056, 0.5043508, 0.51740885, 0.5033256, 0.5040634, 0.5150934, 0.54839903, 1.209395, 0.5246871, 0.53192925, 0.5314036, 0.5375862, 0.5791991, 0.5710441, 0.53895485, 0.57677114, 0.600072, 0.5638864, 0.6614548, 0.59057456, 0.5314453, 0.50168943, 0.51701695, 0.5461273, 0.5688169, 0.5675705, 0.52345717, 0.55831504, 0.5143018, 1.1718081, 0.51399606, 0.5269096, 0.5673253, 0.5584077, 0.5286503, 0.55216897, 0.5223105, 0.5307484, 0.5230492, 0.5422701, 1.309265, 0.70488423, 0.53424186, 0.5748574, 0.50587165, 0.49862638, 0.4778314, 0.46768716, 0.49140784, 0.48694974, 0.48518664, 1.1934555, 1.1472119, 0.4493349, 1.1620861, 0.46942252, 1.1531765, 0.49056298, 0.90355504, 0.48594657, 0.97524554, 0.47892913, 1.0652453, 0.6378703, 0.48781967, 0.99512756, 0.4712237, 0.8753296, 0.49158382, 0.8822529, 0.45545182, 0.91699773, 0.47445762, 0.97977424, 1.1913676, 0.48432904, 0.8524795, 0.48637724, 0.86455494, 0.43187672, 0.94668007, 0.46326312, 0.85129964, 0.47116438, 0.8213656, 0.64071536, 0.47183287, 0.8498823, 0.49679297, 0.8326716, 0.4660458, 0.81441927, 0.47910064, 0.87467504, 0.49050236, 0.84752727, 1.1311802, 0.48549744, 0.82170993, 0.47924668, 0.8224171, 0.4626814, 0.79955685, 0.47336835, 0.8403828, 0.47812042, 0.81951934, 0.6290349, 0.45077178, 0.81156653, 0.47010952, 0.90279675, 0.4747129, 0.88273317, 0.48235458, 1.000876, 0.47204462, 0.9341054, 1.112366, 0.47688818, 0.81283134, 0.53129977, 0.87829745, 0.5052832, 0.8007532, 0.47474903, 0.7683724, 0.47469428, 0.8606013, 0.65128577, 0.48870718, 0.83066154, 0.49156803, 0.78667766, 0.4653574, 0.7700885, 0.49709255, 0.808866, 0.47516972, 0.7330478, 1.1333607, 0.49474695, 0.74070764, 0.49653473, 0.7531224, 0.49217302, 0.73796314, 0.5476208, 0.7528938, 0.49608546, 0.7289502, 2.1283064, 0.6817309, 0.50610614, 0.783653, 0.51742256, 0.80498385, 0.50314337, 0.8191736, 0.5085935, 0.76722354, 0.5122653, 0.7666669, 1.142782, 0.48290017, 0.706743, 0.5159788, 0.7698188, 0.48181388, 0.7162163, 0.5248, 0.7767207, 0.5352678, 0.67800295, 0.66540706, 0.5831772, 0.7192057, 0.5243663, 0.76491475, 0.5273603, 0.69175875, 0.53821623, 0.73808825, 0.52227813, 0.69699997, 1.161415, 0.5414786, 0.6830661, 0.52973056, 0.6844747, 0.54163617, 0.6677201, 0.5273989, 0.75026137, 0.5605487, 0.6994592, 0.68567044, 0.5221614, 0.64997655, 0.52365464, 0.6580323, 0.5537408, 0.67456317, 0.5329463, 0.72677445, 0.5367231, 0.7088639, 1.2059215, 0.5589226, 0.69221973, 0.49746168, 0.70179814, 0.5311666, 0.66351146, 0.52948457, 0.6825378, 0.52601683, 0.6703503, 0.7420579, 0.52645665, 0.6263323, 0.5272713, 0.62069285, 0.48750997, 0.6641049, 0.55733824, 0.60320294, 0.5551795, 0.6805292, 1.2711749, 0.4934622, 0.6405267, 0.5329168, 0.58149266, 0.55505407, 0.61787343, 0.5336207, 0.5947243, 0.5047674, 0.5738033, 0.7892342, 0.5098699, 0.5430693, 0.5080724, 0.54741746, 0.52502686, 0.5813072, 0.51891494, 0.5206379, 0.49485508, 0.5229217, 1.3126906, 0.49703008, 0.5210236, 0.5232369, 0.48984438, 0.5370479, 0.5100169, 0.527659, 0.5228146, 0.56125265, 0.52485967, 1.7618626, 0.759191, 0.5827507, 0.5532225, 0.5962262, 0.5503949, 0.5108259, 0.5322402, 0.5323822, 0.5300336, 0.6182123, 0.5428572, 1.3188275, 0.6208577, 0.54855645, 0.62095404, 0.5301109, 0.605808, 0.6072186, 0.6446209, 0.61039317, 0.63513505, 0.58604985, 0.711579, 0.6311703, 0.57621634, 0.64560294, 0.6313274, 0.66063684, 0.62737745, 0.6625258, 0.6167084, 0.6727414, 0.7126513, 1.3167728, 0.65935814, 0.74465907, 0.66721046, 0.73989815, 0.6573993, 0.68683696, 0.6476053, 0.6956692, 0.64544857, 0.7735023, 0.59100956, 0.6806715, 0.7615577, 0.688259, 0.73793733, 0.638016, 0.81283677, 0.6385649, 0.7404304, 0.6409431, 0.77393687, 1.2318649, 0.69084346, 0.7740709, 0.6465047, 0.8657448, 0.6675059, 0.8128902, 0.68931013, 0.7570848, 0.6599779, 0.8101711, 0.5514458, 0.6948635, 0.8123086, 0.6918902, 0.83692235, 0.6847242, 0.7304342, 0.67919123, 0.7827004, 0.67803895, 0.7888297, 1.0973182, 0.6496315, 0.7692661, 0.61014456, 0.73689747, 0.6289492, 0.79707223, 0.63574743, 0.73602337, 0.6319114, 0.7640428, 0.52762395, 0.5539578, 0.78602546, 0.6499529, 0.74983466, 0.56393343, 0.7578904, 0.6286848, 0.7571017, 0.6603565, 0.7466613, 1.1435655, 0.67115885, 0.82523555, 0.6516764, 0.74589914, 0.6260887, 0.7445273, 0.6911774, 0.8117078, 0.66735905, 0.74566054, 2.3010428, 1.926179, 0.53010637, 0.70032465, 0.74541223, 0.707479, 0.7479517, 0.70652103, 0.766184, 0.705787, 0.7693428, 0.7339627, 0.83637697, 1.0633574, 0.6569322, 0.84836, 0.8008324, 0.8370825, 0.78921765, 0.8094214, 0.74760723, 0.8052755, 0.76694715, 0.7529656, 0.540716, 0.8310153, 0.757127, 0.8294003, 0.8389484, 0.8560269, 0.81190515, 0.86663437, 0.84160334, 0.93210435, 0.8438927, 1.1761944, 0.8424393, 0.82301134, 0.86421716, 0.97398895, 0.77078056, 0.8329805, 0.77724683, 0.8637568, 0.8393773, 0.962295, 0.56451005, 0.87874943, 0.924756, 0.7576796, 0.8471645, 0.8356985, 0.77293754, 0.90644884, 0.7719526, 0.8306238, 0.77766275, 1.1128893, 0.91676146, 0.94547933, 0.9272741, 0.95770395, 0.82342345, 0.90916336, 0.93328774, 1.0069659, 0.8021265, 1.0202407, 0.56510645, 0.8429415, 1.1531208, 0.827154, 1.5363591, 0.9357783, 0.93898904, 0.8489965, 0.8452112, 0.9602947, 0.84532166, 1.172883, 1.0289539, 0.9630409, 0.8397914, 0.890385, 0.8104859, 0.79177356, 0.84271675, 0.79784167, 0.8120911, 0.8075554, 0.5817458, 0.77234495, 0.8265469, 0.8162668, 0.8280387, 0.86068475, 0.7883042, 0.8031953, 0.745465, 0.9051417, 0.70430356, 1.2244623, 0.87658054, 0.66565746, 0.81189305, 0.62640065, 0.6241974, 0.636319, 0.6453385, 0.5494762, 0.7029986, 0.52487755, 0.9777402, 0.6177133, 0.641439, 0.5033601, 0.6065273, 0.49467945, 0.60975593, 0.5438702, 0.67580307, 0.5352134, 0.5351189, 0.5164794, 1.2932512, 0.530841, 0.5613107, 0.5747229, 0.6153363, 0.69288534, 0.73454154, 0.7963729, 0.9340939, 0.9061446, 0.8644048, 0.6714076, 1.5606565, 0.941973, 1.2875243, 0.7941743, 1.2103845, 0.8305099, 0.8761752, 1.0048068, 0.88290966, 1.115034, 1.2115251, 0.8221712, 1.1686953, 0.7790844, 1.1449019, 0.8508323, 1.0258093, 0.7735584, 1.1074013, 0.92094, 1.2690783, 0.71658856, 0.8557813, 1.2300011, 0.84578, 1.1981679, 0.82651323, 0.98635924, 0.8231096, 1.1058893, 0.7501948, 1.2815992, 1.2641573, 1.4082575, 0.8120696, 1.6744521, 0.80710745, 0.69401944, 0.69107604, 0.65535337, 0.6175843, 0.5751827, 0.56600857, 0.7434592, 1.0150554, 0.5291063, 0.85518825, 0.505835, 0.66027874, 0.47185394, 0.66268295, 0.4784247, 0.7141076, 0.47908252, 1.1147851, 0.7437393, 0.47818726, 0.7420926, 0.5304161, 0.7554003, 0.5219802, 0.53414726, 0.6579319, 0.5813466, 0.6648667, 0.6812192, 0.55679446, 0.7858459, 0.61234736, 0.90832764, 0.66216594, 1.1510803, 0.79021436, 1.2437972, 0.80525696, 1.3260329, 0.98816884, 0.81196576, 1.4024754, 1.5411903, 0.80517477, 1.4454, 0.80882, 0.8256909, 1.5140216, 0.7821746, 1.5480615, 1.9271337, 0.6575707, 0.8532314, 1.6728225, 0.77384084, 1.6450547, 0.76620847, 1.6568415, 0.82671773, 1.687027, 0.74413717, 1.6513188, 1.0541382, 0.84480643, 1.7517663, 0.8235319, 1.9427128, 0.8360591, 1.7816162, 0.86821485, 2.060267, 0.8635903, 2.2101786, 0.6593719, 0.8574193, 1.9723985, 0.8379703, 2.0249712, 2.4563153, 0.84171796, 0.79054374, 0.8600884, 0.7776823, 0.7621567, 0.93064773, 0.8283876, 1.9580004, 0.812206, 0.81479645, 2.1649315, 0.8364219, 2.316044, 2.1345913, 0.8233733, 0.81206775, 0.76781774, 2.121104, 0.84292144, 2.2646177, 0.82188, 2.131322, 0.8642273, 1.861799, 0.8568418, 1.8881639, 0.83086586, 0.9205483, 1.970709, 0.79460216, 2.0112295, 0.80412513, 2.049962, 0.8499535, 2.0870154, 0.8504202, 2.0556993, 0.81836253, 0.8803396, 2.0543869, 0.858088, 2.0506258, 0.8747589, 2.100355, 0.8565494, 2.09331, 0.8427079, 2.056882, 0.8498192, 0.9262274, 2.1112003, 0.83304644, 2.0908165, 2.040586, 0.857286, 0.8692205, 2.2731776, 0.86447513, 2.128486, 2.1762793, 0.8642776, 0.8500794, 0.8641033, 2.233551, 0.8403846, 2.1779752, 0.90956044, 2.026657, 0.8853182, 2.120083, 0.863556, 0.78225803, 2.0102499, 0.88370526, 2.2035468, 1.0071713, 2.0621948, 0.9749452, 2.4540544, 1.0041325, 2.1910758, 1.065562, 0.7987671, 0.87140316, 2.013938, 1.4004277, 2.0645647, 2.075469, 2.3838005, 2.2514946, 2.1641004, 2.0505753, 1.9139413, 1.9874606, 0.77282906, 2.109186, 2.0854957, 1.8753357, 1.7928915, 1.7345796, 1.7518744, 1.6468326, 1.5872114, 1.5071647, 1.3260947, 0.9851184, 1.2711598, 1.318278, 1.3737893, 1.3719155, 1.2650447, 1.3725559, 1.2251494, 1.1758108, 1.1470869, 1.3664687, 0.7876287, 1.083378, 1.2213445, 1.0640783, 1.0879697, 1.0987216, 0.97382206, 1.0592632, 0.82981306, 0.96179694, 0.734828, 1.0239046, 0.9823249, 0.6462918, 0.9674748, 0.7021813, 0.8785777, 0.58806145, 0.85024166, 0.54667455, 0.90781134, 0.5862899, 0.74760556, 1.0105321, 0.5388808, 0.94989955, 0.54140675, 0.9843829, 0.6095975, 0.9945696, 0.566234, 1.1322937, 0.56915283, 0.7976001, 1.144848, 0.6582654, 1.1481693, 0.59746593, 1.1596093, 0.607268, 1.2570902, 0.6575186, 1.2653844, 0.697754, 0.78676885, 1.3094696, 0.7030861, 1.4959838, 0.644682, 1.5395366, 0.6112347, 1.5463458, 0.66552866, 1.6228333, 0.6426631, 0.8210946, 1.6252048, 0.6754187, 1.6055412, 0.65801823, 1.6412569, 0.6313117, 1.6608609, 0.6654809, 1.7387371, 0.7852914, 0.77620924, 1.7288685, 0.7358352, 1.8349786, 0.8045137, 1.8474768, 0.9606919, 1.800747, 0.9127569, 1.8255085, 0.9544598, 0.78778416, 0.7507276, 1.9080701, 0.99197274, 1.7925915, 0.9965617, 2.0022306, 0.9459298, 2.0140297, 1.2003335, 1.9634508, 1.2305028, 0.7562148, 2.047152, 1.1401912, 2.1424859, 1.1956407, 2.0622795, 1.2163255, 2.3645637, 1.242086, 2.0788229, 2.1836028, 0.7408715, 1.2368637, 2.4816005, 1.1639409, 1.816778, 1.1369102, 1.8137153, 1.0794735, 1.0440029, 1.8701278, 1.0478073, 0.8082941, 2.061497, 1.0462133, 1.8948138, 1.0157764, 1.925389, 1.0242473, 1.752244, 0.96857065, 1.7757725, 1.0453942, 0.821699, 1.7160088, 1.1472667, 1.7679403, 1.0169966, 1.8738892, 0.9996787, 1.7641501, 1.0335757, 1.9098867, 1.0650661, 0.82312196, 2.1222188, 1.023088, 2.0958006, 1.0295632, 2.1084456, 1.0063224, 2.094128, 1.0279547, 1.7152883, 1.0434023, 0.89243513, 1.7265257, 1.9207932, 1.0695702, 1.0721875, 1.9427251, 1.0860884, 2.1265209, 1.1401149, 1.89675, 1.0597306, 0.8805764, 2.113135, 1.0011426, 1.8924992, 1.0614065, 2.0055299, 1.0790135, 1.9516157, 1.061102, 1.9096705, 1.1258191, 0.8361913, 2.1671834, 1.1426846, 1.9847586, 1.1051928, 2.1721215, 1.1573689, 2.0931742, 1.2000002, 1.9756588, 1.1483175, 0.88544285, 1.830621, 1.1493261, 2.1833203, 1.1331874, 2.1941862, 1.1457887, 1.8628683, 1.1246079, 1.8812563, 1.170191, 1.9793794, 0.93301183, 1.897062, 1.9294311, 1.1434639, 1.9567504, 1.2074342, 2.0224605, 1.2914606, 1.1606067, 2.049404, 1.1814524, 0.9600321, 2.2552886, 2.0775847, 1.2377504, 1.1739173, 1.9979347, 1.1811588, 1.9068525, 2.0620503, 1.1048108, 1.0937648, 1.0108091, 2.0243282, 1.9853, 1.0033097, 1.0370371, 1.845263, 1.1892483, 1.9102588, 1.2155662, 1.872905, 1.110211, 0.97017956, 2.0007405, 1.2090455, 2.035563, 1.1449649, 1.989371, 1.9575002, 1.9707087, 2.0586767, 2.08827, 1.9990865, 1.032305, 2.263075, 2.1818182, 2.2418792, 2.2138243, 2.0329192, 2.1401227, 2.0908825, 2.0675392, 2.0170765, 2.0022795, 1.0486584, 2.0249882, 2.0085256, 2.1459048, 1.9680924, 2.029544, 2.0367246, 1.9700058, 2.057259, 2.0589771, 1.961165, 1.0986198, 2.0480652, 2.006017, 1.8032862, 1.8312852, 1.8443644, 1.7192636, 1.8231567, 1.8473811, 1.5873004, 1.5752752, 1.1057483, 1.5548626, 1.5550172, 1.4985814, 1.4883099, 1.5067742, 1.6006765, 1.6826665, 1.5778236, 1.6332556, 1.5509131, 1.0966171, 1.5348473, 1.466666, 1.5425999, 1.5747477, 1.451374, 1.383931, 1.3206353, 1.3948854, 1.2403187, 1.2286447, 1.1369318, 1.2792226, 2.1752489, 1.2132244, 2.0621722, 1.1922047, 1.9412369, 1.204619, 1.7641343, 1.2913885, 1.6415199, 0.9006636, 1.1490208, 1.2106327, 1.5913944, 1.1947829, 1.3919072, 1.167765, 1.2538917, 1.0923692, 1.2209177, 1.1110005, 1.1239532, 1.2611048, 1.1016407, 0.9392632, 1.0645386, 1.0204505, 0.9999467, 0.90096873, 1.0374386, 0.870784, 1.0378878]
        #noline_loss = [1.379164, 0.64775914, 1.3417615, 4.9101253, 4.7664804, 4.4638643, 4.2100244, 4.414504, 6.0034943, 10.0810995, 10.029552, 5.4012647, 10.424898, 1.2859616, 9.871092, 4.5671453, 10.315784, 10.878724, 10.694081, 11.084484, 11.3124485, 12.030845, 5.02777, 11.97454, 1.3141966, 4.8961096, 12.157801, 5.248364, 11.955091, 4.83218, 11.880718, 6.11392, 11.568265, 5.761218, 11.322871, 1.3174758, 5.8576784, 10.825482, 6.7406235, 10.273292, 10.673338, 9.240569, 9.239459, 8.262513, 7.6423316, 7.125893, 1.2763736, 6.7391434, 6.3563695, 6.1523423, 6.012737, 5.1900887, 5.482439, 5.144725, 4.90246, 4.5377293, 4.437884, 1.1734804, 4.329805, 4.254024, 4.3072324, 4.425372, 4.4310555, 4.4531693, 4.581457, 4.701758, 4.8671503, 5.025923, 1.1742289, 5.117206, 5.174338, 5.29183, 5.0534906, 5.226385, 5.2628484, 5.1334114, 5.021128, 5.048851, 4.9785533, 1.3036572, 5.0476775, 4.8973536, 4.851448, 4.8674054, 4.899365, 4.912694, 4.9718313, 5.0084376, 5.0491223, 5.041843, 1.2540756, 4.9960036, 5.0356483, 5.0646057, 4.9474463, 5.0110345, 4.9539557, 4.9923368, 5.072187, 5.0532146, 5.0351663, 1.2030369, 5.088299, 5.0283246, 5.098251, 5.108157, 5.12263, 5.128315, 5.1929617, 5.234412, 5.247928, 5.2174544, 3.6779025, 1.1258131, 5.282349, 5.0871124, 5.3055987, 5.2209325, 5.22499, 5.116683, 5.130188, 5.2961607, 5.143526, 5.087479, 1.1905003, 5.311313, 5.113325, 5.1839757, 5.3608513, 5.2018743, 5.326948, 5.175318, 4.9768715, 5.180049, 5.2036943, 1.1304194, 5.2425632, 5.576514, 5.0886054, 5.268594, 4.9525275, 5.068535, 5.279517, 4.9518538, 6.722457, 4.892946, 1.1406168, 6.8162737, 4.7192245, 4.881198, 4.9315157, 4.9794602, 4.9219294, 4.537878, 4.5225363, 4.6821074, 4.6396933, 1.2538567, 4.7013183, 4.6195498, 4.693279, 4.6720467, 4.6708813, 4.7426953, 5.0012207, 4.745612, 4.8417015, 5.1033397, 1.2821702, 5.190149, 4.8214784, 5.083667, 4.8526626, 5.3945804, 4.9291277, 5.265763, 5.1573896, 4.7933927, 5.080822, 1.1973337, 5.046548, 5.112001, 4.9530544, 4.974281, 4.982852, 4.5670056, 4.748274, 4.8111005, 4.7143216, 4.678874, 1.1897984, 4.4346128, 4.3082094, 4.293495, 4.1119127, 3.8524776, 3.8081508, 4.003519, 3.9734044, 3.840597, 3.736528, 1.0911514, 3.6480875, 3.7363994, 3.5206265, 3.2482884, 3.024782, 2.7412367, 2.7519464, 2.814065, 3.7843366, 3.77317, 1.0282735, 3.834039, 3.6123135, 3.6539812, 4.0491185, 3.9590545, 3.9615095, 3.9207556, 3.930594, 3.7727804, 3.8507023, 4.972323, 1.0116061, 4.060838, 4.0744195, 4.153784, 4.260429, 4.162053, 4.0861325, 4.23429, 3.9950893, 3.9909885, 4.0399384, 0.98221356, 3.9614353, 4.1831717, 4.051021, 3.9733534, 3.804828, 3.8687088, 4.0696354, 3.9920397, 4.102005, 4.1007648, 0.9659832, 4.148145, 4.2255, 4.299805, 4.2644234, 4.3397145, 4.3792367, 4.4062057, 4.5031576, 4.6890035, 4.650916, 1.0320953, 4.7101603, 4.640361, 4.8409076, 4.7457786, 4.9404583, 4.9096494, 5.000093, 5.0042033, 5.1464157, 5.1916265, 1.2840066, 5.2076607, 5.1197877, 5.116444, 5.123144, 5.118279, 5.077558, 5.1149335, 5.124285, 5.177471, 5.158133, 1.529228, 5.1235795, 5.196534, 5.180963, 5.0805144, 5.1686006, 5.2090216, 5.046668, 5.157177, 5.214348, 5.2999897, 1.6252197, 5.3453183, 5.3379893, 5.351318, 5.353169, 5.405328, 5.4415603, 5.5021696, 5.4545226, 5.5080943, 4.9670243, 1.7886283, 5.6760755, 5.7647386, 5.6406193, 5.8354955, 5.8633223, 5.895316, 5.963326, 6.038063, 6.023847, 6.069978, 1.0053993, 6.0059366, 6.0594, 6.142705, 6.164653, 6.150599, 6.187579, 6.209066, 6.2970943, 6.261782, 6.322736, 1.0227587, 6.5139694, 6.588146, 6.5698256, 6.5958977, 6.404993, 6.669505, 6.6684747, 6.7393527, 6.7484574, 6.7278886, 0.6570909, 0.8805344, 6.6418266, 6.7776084, 6.7105727, 6.7072487, 6.7224054, 6.834923, 6.793119, 6.8175135, 6.787206, 6.779457, 2.0079665, 6.735751, 6.589194, 6.809354, 6.7961307, 6.779944, 6.800625, 6.8185725, 6.7963533, 6.550046, 6.7992983, 2.4189444, 6.8095794, 6.792476, 6.840311, 6.95827, 6.8334327, 6.873217, 6.849225, 6.9700913, 6.945909, 6.9711776, 3.3108182, 6.968893, 6.795589, 7.02888, 6.998665, 6.8080106, 7.158932, 7.0316753, 7.097985, 7.0453167, 7.0989428, 3.3528938, 7.1093483, 7.111658, 7.1255608, 7.206514, 7.181017, 7.198429, 7.2444396, 7.316403, 7.0395155, 7.332963, 3.6234305, 7.2572603, 7.3200135, 7.2497315, 7.355371, 7.301611, 7.294648, 7.2780924, 7.280601, 7.277409, 7.304515, 3.9521768, 7.250771, 7.2387295, 7.2093563, 7.2716193, 7.161416, 7.1031504, 7.121967, 7.1274505, 7.070367, 7.055622, 2.8397038, 7.08185, 7.124754, 7.1229277, 7.1661115, 7.091841, 7.0656314, 7.0760384, 7.1432037, 7.11874, 6.9423037, 2.1635158, 7.113695, 7.1712384, 7.1226997, 7.2832985, 7.1652265, 7.35728, 7.1344748, 7.3491244, 7.1169944, 7.1461887, 2.4223943, 7.1397834, 7.168665, 7.0599756, 7.092245, 6.9426208, 6.993233, 6.919306, 6.793997, 6.3504815, 6.2627354, 5.746078, 1.9810226, 6.02623, 5.8538365, 5.776943, 5.530649, 5.395414, 5.393508, 5.469608, 5.311428, 5.0880437, 5.0451937, 1.8243034, 4.7873473, 4.570678, 4.3845186, 4.096441, 3.8405976, 3.974465, 3.5178597, 3.0415318, 2.8075624, 2.6015558, 1.3939501, 2.0395904, 1.6805034, 2.9029946, 0.9049603, 5.0620337, 1.1969769, 5.3060684, 5.273647, 5.196715, 5.6483426, 1.4311204, 6.667158, 1.570299, 1.7756697, 7.024079, 5.5905023, 2.2770846, 3.33383, 3.541521, 3.6280951, 3.7045593, 1.2821997, 3.7977324, 3.826181, 3.5861702, 3.5127237, 3.4370363, 3.3945494, 3.3584914, 3.2501705, 3.319481, 3.4678004, 1.1304998, 3.362238, 3.3884916, 3.3958573, 3.569314, 7.6796308, 3.5423987, 3.4536042, 3.3966818, 3.4715228, 2.6927872, 0.9993853, 3.4916813, 3.0084572, 3.6024003, 3.5422716, 2.674268, 3.5819798, 5.884299, 3.545869, 4.4619746, 3.6350508, 1.0953243, 2.6527767, 3.5464902, 1.9845264, 1.3387616, 3.6638288, 3.5229974, 1.2706406, 3.4967952, 1.1003399, 3.3841417, 1.0387706, 1.3130121, 3.3130243, 1.2990199, 3.1207027, 1.1019228, 2.9842505, 1.1202645, 1.0481726, 2.9535477, 2.9565237, 1.037617, 1.205523, 2.9828894, 1.1566625, 3.0951269, 1.1369925, 3.1671963, 1.1812195, 3.177233, 1.1737531, 3.0244184, 6.1155567, 0.95964634, 1.1864495, 1.0625598, 0.98413324, 1.1319376, 1.3050126, 1.3023111, 3.1577454, 1.5280628, 1.5729411, 1.3879163, 1.003341, 2.0741098, 1.4075723, 1.1792676, 1.2633928, 1.2905618, 1.0753803, 0.99021906, 0.9374028, 1.0655652, 1.118465, 1.0380722, 1.2389845, 0.76175, 1.0820792, 3.7615054, 3.802969, 0.91943604, 4.0836945, 1.0963029, 1.1591985, 1.1690178, 1.0741675, 1.094195, 4.622142, 0.79775697, 4.998198, 5.374561, 1.1783819, 5.484442, 1.0732439, 5.9540067, 1.0776573, 1.1219091, 6.226625, 7.487713, 8.805012, 10.857228, 10.432449, 2.585559, 2.8815901, 2.9952214, 2.8613591, 2.607643, 1.073593, 2.8876972, 2.6605265, 2.642603, 2.8720326, 2.454018, 2.3589501, 2.1214607, 2.1426587, 2.2496986, 2.3235784, 1.1723822, 2.181881, 2.006181, 3.6245437, 3.63582, 3.6188853, 3.5249069, 3.352914, 3.4317234, 3.3768044, 3.2606237, 1.1499025, 3.078337, 2.85246, 2.6510067, 2.5908244, 2.535471, 2.6052792, 2.6438272, 2.6064067, 2.6758564, 2.668451, 1.0951556, 2.7627428, 2.728688, 2.7419183, 2.797144, 2.848068, 2.863926, 2.9261575, 2.9536753, 3.033501, 3.0325673, 1.1351613, 3.238915, 3.2319183, 3.370019, 3.2405622, 3.3270235, 3.3205345, 3.3681765, 3.404551, 3.4689353, 3.5170283, 6.0496716, 1.107946, 3.5005999, 3.4433575, 3.4884326, 3.514992, 3.9010444, 3.5805933, 3.5358562, 3.5867498, 3.5751362, 3.6760027, 1.0534426, 3.6291864, 3.693714, 3.6244504, 3.742375, 3.6931083, 3.6960478, 3.5885305, 3.844324, 3.7762656, 3.706763, 1.0535369, 3.5857513, 3.6013894, 3.7234411, 3.6276982, 3.7259192, 3.7940018, 3.7519374, 3.6947489, 3.6291003, 3.799365, 1.0022991, 3.7208989, 3.805362, 3.831186, 3.8864474, 3.6620028, 3.95591, 3.8085883, 3.9086356, 3.64574, 3.8537018, 0.94095004, 3.7027977, 3.8947153, 3.8840995, 3.6889892, 3.986217, 3.8185737, 3.870505, 4.0245075, 4.105253, 4.0615172, 0.9068966, 3.9305098, 4.1281743, 4.034035, 4.303629, 4.230232, 4.2664895, 4.0405006, 4.23206, 4.41625, 4.2555375, 0.8921769, 4.36133, 4.1931477, 4.308924, 4.3052764, 4.320404, 4.372498, 4.55659, 4.3121586, 4.241451, 4.1873593, 0.84788275, 4.1614523, 4.2514043, 4.2512197, 4.256922, 4.2436147, 4.272861, 4.26218, 4.1420536, 4.097688, 4.1001134, 0.8846, 4.216137, 3.8807805, 3.8600616, 3.8414984, 4.059388, 3.9317093, 4.016901, 3.9127276, 3.954016, 4.111426, 0.9329278, 4.174136, 4.293953, 4.2917, 4.271813, 6.9074235, 4.24212, 4.25752, 6.8888903, 4.2497263, 4.2629333, 4.66547, 1.0316203, 4.3957524, 4.332322, 4.2507257, 4.3730407, 4.2753773, 7.0952563, 3.6458352, 4.3763876, 3.3359437, 6.9217486, 1.171363, 4.4031687, 3.0188046, 4.486945, 4.0493326, 4.405557, 5.0523467, 4.7197723, 6.9003115, 3.8196623, 4.624153, 1.2260085, 5.9207907, 4.5213814, 2.8302007, 4.9239354, 2.7164094, 4.572397, 2.9235542, 5.160445, 4.602916, 5.9377203, 1.3043046, 2.9406662, 4.654827, 5.9155955, 3.1631305, 4.8662553, 2.9890208, 4.8650613, 2.1251583, 4.8889866, 4.9727697, 1.1712269, 2.36079, 5.6741066, 2.165614, 5.067485, 2.1245408, 5.057967, 2.393396, 5.4537745, 3.874723, 2.299228, 1.1730281, 5.368915, 4.4448643, 1.924123, 5.39566, 2.6765134, 5.6093073, 2.2391026, 4.370018, 5.6500025, 5.539173, 1.1377132, 4.9727783, 5.7209144, 4.25865, 6.0625315, 3.7507758, 5.6171145, 3.205116, 5.611303, 4.0035944, 5.5196934, 1.132474, 4.1540008, 5.7831345, 4.1051416, 5.5998287, 4.1449494, 5.570796, 4.824359, 5.115757, 4.9279094, 5.115837, 1.1840274, 4.9341784, 4.5288267, 4.8633137, 4.9427056, 5.116444, 4.4625716, 4.746837, 4.275482, 4.789212, 4.054961, 1.2339758, 4.731941, 4.3884306, 4.6788855, 4.49412, 3.9890118, 3.8768504, 4.4047475, 3.7610996, 4.8092833, 3.3960977, 1.57542, 1.1999708, 4.743869, 3.4060583, 4.2111783, 3.4005113, 4.347954, 3.3764188, 3.6538422, 3.2884512, 3.4465938, 3.1943142, 1.2255185, 3.2739117, 3.2321005, 4.064195, 3.31839, 4.512263, 3.421822, 4.0994906, 3.5209026, 4.0685406, 3.7177677, 1.1425892, 4.2617507, 4.355211, 4.7100515, 3.9603822, 3.7538464, 6.5018134, 4.1771007, 4.0966525, 6.088739, 3.2159433, 1.0777863, 3.4942992, 4.892122, 3.8281522, 3.4216597, 3.2821767, 3.6319666, 3.944603, 3.3677924, 2.9826756, 2.7247672, 1.2226412, 2.4033732, 2.3840625, 2.0851905, 1.7662601, 1.1955004, 1.2415674, 0.94793177, 1.2973188, 0.8473568, 0.9870215, 1.2549789, 2.1964376, 2.381419, 2.4001162, 2.2639534, 0.94678944, 2.2341568, 2.3461263, 0.9246297, 0.9858532, 2.3546789, 1.2910303, 0.95207673, 2.3289542, 0.9867547, 2.3480897, 2.2005622, 2.3242817, 0.8899288, 2.272728, 0.90709364, 2.250097, 1.3383259, 0.88986903, 2.233204, 0.875574, 2.2208889, 2.1853585, 0.9098949, 2.1704423, 0.95461196, 2.18049, 0.9955543, 1.3601606, 2.1484418, 1.0231786, 2.1655872, 1.0142502, 2.1590834, 0.98346865, 2.1469796, 0.94305164, 2.1110039, 0.9120032, 1.3282365, 1.9897202, 0.91487694, 2.085427, 0.9575864, 2.0821936, 0.97201496, 2.0569048, 1.01835, 2.0489626, 2.128169, 1.6852083, 1.1909, 1.0698637, 1.0424628, 2.0767393, 2.11892, 1.0204507, 0.9785838, 2.115618, 0.937414, 2.1046991, 0.9611991, 1.3180991, 2.125094, 0.9922985, 2.0638921, 0.8634584, 2.008282, 0.8284824, 2.0463738, 0.8469408, 1.9821905, 0.86152667, 1.3230251, 1.9644104, 0.79802257, 1.9549917, 0.7999069, 2.0013154, 0.8089142, 2.01133, 0.8389893, 2.0226707, 0.8510154, 1.3514495, 1.9536362, 0.89033824, 2.104342, 0.8302876, 1.992359, 0.84332174, 2.4121501, 0.84370357, 2.3169127, 0.9654017, 1.3452265, 2.4659295, 0.97060335, 2.555465, 1.0044632, 2.6490297, 0.98172, 3.1750023, 1.0432587, 2.7773252, 0.9901436, 1.3651543, 2.6989856, 0.9708775, 2.8451533, 1.0342599, 2.8309531, 1.1573434, 3.1459644, 1.1562059, 3.0388885, 1.031976, 1.3552437, 3.2027967, 0.98224807, 3.2118335, 1.0490233, 3.6341627, 1.0444592, 3.7550952, 0.93645173, 3.5962849, 0.89527595, 1.3586739, 3.6696568, 0.9372033, 3.7791066, 0.9218017, 3.913798, 0.9044046, 3.6472049, 0.91067815, 4.626734, 0.92084503, 1.3318572, 4.651491, 1.0900093, 3.9526098, 1.1380444, 4.0120277, 4.959712, 5.158723, 4.6190457, 5.5399756, 5.759945, 1.3265778, 4.9638586, 5.8652472, 6.1231947, 5.725153, 4.6020045, 4.645149, 5.5849423, 1.2467718, 4.6169705, 1.278186, 1.4098581, 1.8805374, 1.3542769, 4.677634, 1.1938714, 1.2348473, 1.3717964, 1.3847083, 1.395461, 1.3831059, 1.3500015, 1.3326575, 1.3268398, 1.3227695, 1.2849524, 1.8560115, 1.2476591, 1.2570634, 1.1999784, 1.2165796, 1.1864487, 1.1527488, 1.138159, 1.0718751, 1.0970446, 1.0800483, 1.8182606, 1.0552528, 0.9931659, 1.0261467, 0.9695245, 0.97090566, 0.9507373, 0.91914654, 0.9617044, 1.0025289, 0.9513339, 1.8701394, 1.0190564, 0.7789086, 0.7457367, 0.6676843, 0.6237731, 0.6254242, 0.7355079, 0.67201567, 0.69408983, 0.61749405, 1.8683277, 1.2725481, 1.3044019, 1.4264016, 1.5380026, 1.5411066, 1.501323, 1.4757875, 1.6766938, 1.5648134, 1.6413052, 1.8818554, 1.6698592, 1.73934, 1.7990805, 1.9651178, 2.0387042, 1.966556, 2.2594903, 2.2055824, 2.079498, 2.2454436, 1.7856952, 2.0971737, 1.8890315, 2.194285, 2.115554, 2.0415637, 2.1395261, 2.1625655, 2.3928254, 2.1078374, 2.1737063, 1.9203749, 2.1333878, 2.1713133, 2.1918032, 2.148474, 2.110594, 2.1793003, 2.1060183, 2.1270998, 2.1084373, 2.1015384, 1.8658787, 2.1095092, 2.2858522, 2.2219844, 2.182811, 2.3303587, 2.1328557, 2.3965576, 2.2828355, 1.9915346, 2.1672137, 1.9132501, 1.9147978, 1.8626462, 1.8911421, 1.7644998, 1.7175589, 1.7190769, 1.7164584, 1.6947769, 1.5767452, 1.706526, 1.474343, 1.8565525, 1.5457706, 1.4629779, 1.3615732, 1.3808985, 1.4208096, 1.3807294, 1.2744638, 1.2517486, 1.2588242, 1.1645662, 1.9524955, 1.3382109, 1.0681785, 1.6233742, 0.99104977, 1.7708302, 1.0533506, 1.8538545, 1.310489, 1.5829225, 1.272374, 2.1849666, 1.8564903, 0.78103274, 2.0929868, 2.0426776, 2.0131752, 2.182818, 2.168699, 2.3033693, 2.2963867, 2.3065653, 2.154163, 2.358603, 2.3763685, 2.4073184, 2.4336019, 2.4511867, 2.483282, 2.461018, 2.4982235, 2.4848077, 2.5785637, 2.0943851, 2.5618033, 2.52826, 2.5015988, 2.5115066, 2.4614296, 2.51218, 2.4494674, 2.4058788, 2.2833998, 2.2506244, 2.0871048, 2.093061, 1.8100855, 1.6967561, 1.5768368, 1.4736433, 1.471251, 1.3550212, 1.4937071, 1.4252518, 1.5218517, 2.0871806, 1.4305714, 1.5054543, 1.4416795, 1.3729093, 1.2065415, 1.1606627, 1.5218203, 1.6417367, 1.6601279, 1.6838964, 2.0051177, 1.6862049, 1.4609767, 1.5084691, 1.453648, 1.2031752, 1.0648135, 1.1630019, 0.95994866, 1.0145952, 0.8549895, 2.025725, 2.129408, 0.88932365, 0.94546175, 2.0199234, 2.3206909, 3.1159909, 3.5233877, 3.2547653, 2.796896, 2.0816407, 2.0997314, 1.6130509, 2.1511838, 2.3936253, 2.6963143, 2.583269, 2.3022223, 1.8467253, 2.4366112, 2.6384032, 2.711955, 1.558912, 2.3202114, 2.475243, 2.5309215, 2.5664747, 2.0107932, 2.1556673, 2.335351, 2.5533876, 2.687956, 2.8985863, 2.9327223, 2.4289813, 2.8566043, 2.7883186, 2.681464, 2.5484211, 2.5164397, 2.408796, 2.3223069, 2.1247048, 1.9979845, 1.9127856, 2.53612, 1.9206475, 1.741613, 1.7005334, 1.6269065, 1.561564, 1.484043, 1.504611, 1.5464267, 1.639387, 1.747118, 2.384449, 1.5772482, 1.577722, 1.5594298, 1.838858, 1.9993571, 2.0609424, 2.0217903, 2.1083546, 2.2511482, 2.1672695, 2.5026891, 2.2605572, 2.1646566, 2.1733165, 2.1175816, 2.1156836, 2.1354325, 2.1812716, 2.037996, 2.1110194, 2.163756, 2.3985996, 2.1836617, 2.2449906, 2.286733, 2.284403, 2.4614747, 2.4701974, 2.3966272, 2.3815048, 2.545803, 2.4290328, 2.3839955, 2.431661, 2.4423792, 2.4641604, 2.452717, 2.468691, 2.4796731, 2.5593219, 2.5545404, 2.5833893, 2.543402, 2.2677774, 2.5422907, 2.4898653, 2.4765399, 2.415419, 2.3922708, 2.362439, 2.3343868, 2.2772176, 2.2412758, 2.1414447, 2.1831279, 2.199418, 2.1483855, 2.201675, 2.2408907, 2.3631434, 2.4293003, 2.5678697, 2.6836348, 5.155804, 6.2795253, 2.2161367, 7.9917593, 8.429954, 7.7032313, 7.6743155, 8.170709, 7.9757123, 7.0640507, 6.2106524, 5.47434, 5.9685655, 1.7735628, 2.0482635, 7.721525, 8.558301, 8.654221, 8.801667, 8.79027, 8.658567, 8.518524, 8.328642, 7.9056463, 7.24802, 2.0900025, 6.160594, 5.0350833, 3.9550602, 4.090211, 3.9287086, 4.07117, 3.5552197, 3.4027905, 5.513432, 5.2673206, 2.060936, 5.2796884, 5.4046655, 6.0484753, 5.9755754, 5.720319, 5.787922, 5.918473, 5.914404, 5.62061, 6.1589494, 1.8812325, 6.358809, 5.871143, 5.7927036, 6.187734, 6.812914, 6.7887216, 6.540836, 6.8123283, 6.967613, 7.521024, 1.7811551, 7.7364287, 7.651989, 8.044943, 8.160509, 8.517094, 7.9871297, 8.385259, 7.8435764, 8.070583, 7.9533634, 1.9090388, 7.53164, 7.8023343, 7.258415, 7.072748, 7.001466, 6.896203, 6.7474694, 6.6179233, 6.754543, 6.362356, 2.003385, 6.3859515, 6.4808793, 6.71889, 6.6831613, 6.638773, 6.532848, 6.2762203, 6.388907, 6.121168, 5.9790416, 1.9662101, 5.9361057, 6.192448, 5.9530435, 5.8888893, 5.9423027, 5.8919797, 5.904088, 6.142886, 5.791343, 5.7690334, 1.752754, 6.343214, 5.927099, 5.9102526, 5.8626475, 6.049196, 6.080396, 6.055787, 6.050322, 6.1307135, 6.231326, 1.6992755, 6.156064, 6.1618233, 6.2596345, 6.2894716, 6.381941, 6.3471107, 6.5225306, 6.5657973, 6.614443, 6.686286, 1.8958162, 1.6076777, 6.6723814, 6.7767344, 6.9519305, 7.2766476, 6.9155693, 7.1259527, 7.021141, 7.1493983, 7.2112017, 7.1809015, 1.6849133, 7.469122, 7.3217607, 7.536149, 7.500189, 7.6795206, 7.598707, 7.819268, 7.753621, 7.768056, 8.01442, 1.7127491, 7.984247, 8.0550785, 8.11765, 8.173901, 8.215702, 8.236287, 8.310742, 8.374172, 8.446546, 8.402364, 1.612709, 8.577628, 8.29139, 8.408563, 8.369886, 8.397087, 8.380842, 8.379775, 8.324441, 8.3333235, 8.4605, 1.5198753, 8.5271, 8.430873, 8.457589, 8.486333, 8.409853, 8.455012, 8.655816, 8.58006, 8.490375, 8.408713, 1.5293349, 8.498413, 8.397491, 8.498318, 8.408144, 8.463224, 8.403464, 8.450766, 8.316982, 8.431144, 8.358473, 1.505471, 8.42593, 8.364465, 8.468003, 8.371839, 8.45551, 8.39494, 8.364468, 8.28481, 8.280201, 8.334006, 1.5741955, 8.352163, 8.348495, 8.449505, 8.396185, 8.374654, 8.408079, 8.4055395, 8.356823, 8.369417, 8.324999, 1.5227625, 8.320844, 8.311063, 8.22222, 8.204172, 8.100131, 8.049147, 8.07043, 7.992178, 8.03113, 8.206408, 1.509497, 7.6613054, 7.5061545, 7.7193794, 7.286268, 7.0676284, 6.8394766, 6.565672, 6.2017093, 5.7372484, 5.6855893, 2.0174522, 1.4406462, 5.2424746, 4.7024407, 4.6194468, 3.9932933, 4.1573415, 4.7855005, 5.1651335, 5.6160307, 5.3669887, 5.696159, 1.4852858, 5.538687, 5.793811, 5.5718527, 5.2061477, 5.2044644, 5.4001966, 5.3991976, 5.3897552, 5.433515, 5.187853, 1.6249259, 5.177125, 4.821611, 4.442905, 4.4060974, 5.266726, 4.475666, 6.0566893, 7.269965, 8.61507, 9.619465, 1.5902553, 10.236073, 10.83543, 11.218211, 11.243454, 11.241394, 11.254599, 11.306493, 11.371976, 11.383549, 11.409743, 1.5270104, 11.370665, 11.33555, 11.327843, 11.283856, 11.335864, 11.215164, 11.315452, 11.251296, 11.142881, 11.063075, 1.5187516, 10.665755, 10.524619, 10.137694, 9.689718, 9.314396, 8.340043, 7.4468555, 7.281886, 6.64684, 6.0570636, 1.3897717, 5.8193893, 5.7490335, 5.7118983, 6.1335235, 6.3432274, 6.5236607, 6.766954, 6.2386775, 6.0341554, 5.493253, 1.5831707, 4.988975, 4.3600245, 4.6480165, 2.3491774, 2.3328803, 2.350573, 2.3354957, 2.2220786, 2.2027874, 2.2509, 1.6092926, 2.1554465, 2.0395584, 1.9095435, 2.2829247, 7.6267533, 8.53271, 9.912293, 10.593939, 11.619665, 7.652373, 1.6075078, 11.939026, 12.061074, 7.2442575, 12.027477, 6.770174, 12.080783, 7.7291803, 12.272715, 12.2808485, 12.521535, 2.3456314, 1.4861062, 8.665596, 12.446775, 8.582218, 11.188596, 7.872434, 11.81358, 9.361185, 11.640903, 9.989331, 9.801624, 1.386216, 9.186852, 8.0067425, 7.4182324, 6.4983554, 6.5375967, 4.318077, 7.690562]
        plot_loss_distribution(SHOW_MAX_NUM,line_loss,noline_loss)
        
        