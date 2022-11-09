#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:46:08 2022

@author: ali
"""

import os
import pathlib
'''
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
'''
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
def get_interpreter(w,tflite=False,edgetpu=True):
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

def detect_image(w, im, interpreter=None, tflite=False,edgetpu=True, save_image=True, cnt=1):
    SHOW_LOG=False
    INFER=False
    ONLY_DETECT_ONE_IMAGE=True
    if interpreter is None:
        print('interpreter is None, get interpreter now')
        interpreter = get_interpreter(w,tflite,edgetpu)
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
    os.makedirs('./runs/detect/ori_images',exist_ok=True)
    os.makedirs('./runs/detect/gen_images',exist_ok=True)
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
        if save_image:
        #cv2.imshow('ori_image',im)
            filename = 'ori_image_' + str(cnt) + '.jpg'
            file_path = os.path.join('./runs/detect/ori_images', filename)
            cv2.imwrite(file_path,im)
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
    if SHOW_LOG:
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
        if SHOW_LOG:
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
            if save_image:
                #cv2.imshow('out_image',gen_img)
                filename = 'out_image_' + str(cnt) + '.jpg'
                file_path = os.path.join('./runs/detect/gen_images/',filename)
                cv2.imwrite(file_path,gen_img)
                #cv2.waitKey(10)
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
    
    if SHOW_LOG:
        print('input image : {}'.format(input_img))
        print('input image : {}'.format(input_img.shape))
        print('gen_img : {}'.format(gen_img))
        print('gen_img : {}'.format(gen_img.shape))
    latent_i = y[1]
    latent_o = y[2]
    if SHOW_LOG:
        print('latent_i : {}'.format(latent_i))
        print('latent_o : {}'.format(latent_o))
    _g_loss = g_loss(input_img/255.0, gen_img/255.0, latent_i, latent_o)
    #_g_loss = 888
    if SHOW_LOG:
        print('g_loss : {}'.format(_g_loss))
    #print(y)
    return _g_loss, gen_img

def g_loss(input_img, gen_img, latent_i, latent_o):
    
    def l1_loss(A,B):
        return np.mean((abs(A-B)).flatten())
    def l2_loss(A,B):
        return np.mean(np.sqrt((A-B)*(A-B)).flatten())
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
    err_g_enc = l_enc(latent_i,latent_o)
    g_loss = err_g_con * 50 + \
             err_g_enc * 1
    return g_loss


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
        
        g_loss,fake_img = detect_image(w, images, interpreter, tflite=True,edgetpu=False, save_image=True, cnt=1)
        
        
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
    print('positive_loss len: {}'.format(len(positive_loss)))
    print('defeat_loss len: {}'.format(len(defeat_loss)))
    #print(positive_loss)
    #print(defeat_loss)
    # Plot a simple line chart
    #plt2.plot(x, y)
    # Plot another line on the same chart/graph
    #plt2.plot(x, z)
    plt2.scatter(x,y,s=1)
    plt2.scatter(x,z,s=1) 
    os.makedirs('./runs/detect',exist_ok=True)
    file_path = os.path.join('./runs/detect','loss_distribution_100.jpg')
    plt2.savefig(file_path)
    plt2.show()

#https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib
def plot_two_loss_histogram(normal_list, abnormal_list, name):
    import numpy
    from matplotlib import pyplot
    bins = numpy.linspace(0, 8, 100)
    pyplot.hist(normal_list, bins, alpha=0.5, label='normal')
    pyplot.hist(abnormal_list, bins, alpha=0.5, label='abnormal')
    pyplot.legend(loc='upper right')
    os.makedirs('./runs/detect',exist_ok=True)
    filename = str(name) + '.jpg'
    file_path = os.path.join('./runs/detect',filename)
    pyplot.savefig(file_path)
    pyplot.show()

def infer_python(img_dir,interpreter,SHOW_MAX_NUM,save_image=False):
    import glob
    image_list = glob.glob(os.path.join(img_dir,'*.jpg'))
    loss_list = []
    cnt = 0
    for image_path in image_list:
        print(image_path)
        cnt+=1
        
        if cnt<=SHOW_MAX_NUM:
            loss,gen_img = detect_image(w, image_path, interpreter=interpreter, tflite=True,edgetpu=False, save_image=save_image, cnt=cnt)
            print('{} loss: {}'.format(cnt,loss))
            loss_list.append(loss)
    
    
    return loss_list

if __name__=="__main__":
    PYCORAL = False
    DETECT = False
    DETECT_IMAGE = False
    INFER = True
    if DETECT:
        w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        get_interpreter(w,tflite=False,edgetpu=True)
    if PYCORAL:
        Pycoral_Edgetpu()
        
    if DETECT_IMAGE:
        save_image = True
        im = r'/home/ali/Desktop/factory_data/crops_1cls/line/ori_video_ver2121.jpg'
        #im = r'/home/ali/Desktop/factory_data/crops_2cls_small/noline/ori_video_ver244.jpg'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        #w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104.tflite'
        w = r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        loss, gen_image = detect_image(w, im, tflite=False,edgetpu=True, save_image=True)
        
        
    if INFER:
        #import tensorflow as tf
        test_data_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_line/line'
        abnormal_test_data_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_noline/noline'
        (img_height, img_width) = (64,64)
        batch_size_ = 1
        shuffle = False
        SHOW_MAX_NUM = 500
        save_image=True
        w = r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-20221109.tflite'
        interpreter = get_interpreter(w,tflite=True,edgetpu=False)
        line_loss = infer_python(test_data_dir,interpreter,SHOW_MAX_NUM,save_image=save_image)
        
        noline_loss = infer_python(abnormal_test_data_dir,interpreter,SHOW_MAX_NUM)
        plot_loss_distribution(SHOW_MAX_NUM,line_loss,noline_loss)
        plot_two_loss_histogram(line_loss,noline_loss,'line_noline')
        #=================================================
        #if plt have QT error try
        #pip uninstall opencv-python
        #pip install opencv-python-headless
        #=================================================
        #for loss in loss_list:
            #print(loss)
        '''
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
        
        w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        
        SHOW_MAX_NUM = 1800
        
        show_img = False
        
        line_data_type = 'normal'
        noline_data_type = 'abnormal'
        
        line_loss = infer(test_dataset, w, SHOW_MAX_NUM, show_img, line_data_type,tflite=False,edgetpu=True)
        
        noline_loss = infer(test_dataset_abnormal, w, SHOW_MAX_NUM, show_img, noline_data_type,tflite=False,edgetpu=True)
        
        plot_loss_distribution(SHOW_MAX_NUM,line_loss,noline_loss)
        '''