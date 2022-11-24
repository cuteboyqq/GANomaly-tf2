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

def Pycoral_Edgetpu(w):
    # Specify the TensorFlow model, labels, and image
    #script_dir = pathlib.Path(__file__).parent.absolute()
    script_dir = r'/home/ali/Desktop/GANomaly-tf2/export_model'
    model_file = os.path.join(script_dir, 'G-uint8-20221104_edgetpu.tflite')
    label_file = os.path.join(script_dir, 'imagenet_labels.txt')
    image_file = os.path.join(script_dir, 'parrot.jpg')

    # Initialize the TF interpreter
    print('Start interpreter')
    interpreter = edgetpu.make_interpreter(w)
    print('End interpreter')

    print('Start allocate_tensors')
    interpreter.allocate_tensors()
    print('End allocate_tensors')

    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs 
    print('input details : \n{}'.format(input_details))
    print('output details : \n{}'.format(output_details))
    
    
    return interpreter
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

def detect_image(w, im, interpreter=None, tflite=False,edgetpu=True, save_image=True, cnt=1, name='normal',isize=32):
    SHOW_LOG=False
    INFER=False
    ONLY_DETECT_ONE_IMAGE=True
    USE_PIL = False
    
    USE_OPENCV = True
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
        if USE_PIL:
            im_p = Image.open(im)
            im_p = im_p.convert('RGB')
            im_p = im_p.resize((isize,isize),resample=2) #bicubic
            im = np.asarray(im_p)
            input_img = im
        if USE_OPENCV:
            im_o = cv2.imread(im)
            im_ori = cv2.resize(im_o, (isize, isize)) #lininear
            #im = cv2.cvtColor(im_ori, cv2.COLOR_BGR2RGB)
            im = im_ori.transpose((2,0,1))[::-1] #HWC to CHW , BGR to RGB
            im = np.ascontiguousarray(im)
            im = np.transpose(im,[1,2,0])
            input_img = im
        '''
        diff = np.sum(im_p[:, :, 0] - im_o[:, :, 0])
        
        print('image p.shape = {}'.format(im_p.shape))
        print('image o.shape = {}'.format(im_o.shape))
        
        print('image p.10 element = {}'.format(im_p[0:10, 0, 0]))
        print('image o.10 element = {}'.format(im_o[0:10, 0, 0]))
        
        print('image diff = {}'.format(diff))
        
        
        input()
        '''
        #input_img = im
        '''
        if save_image:
        #cv2.imshow('ori_image',im)
            filename = 'ori_image_' + str(cnt) + '.jpg'
            file_path = os.path.join(save_ori_image_dir, filename)
            cv2.imwrite(file_path,im_ori)
            '''
            #cv2.waitKey(10)
                      
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    
    #im = im[np.newaxis, ...].astype(np.float32)
    im = np.expand_dims(im, axis=0).astype(np.float32)
    im = im/255.0
    if SHOW_LOG:
        print('im : {}'.format(im.shape))
    
    #im = im/255.0
    #im = tf.expand_dims(im, axis=0)
    #im = im.numpy()
    
    #print('im:{}'.format(im.shape))
    #print('im: {}'.format(im))
    input = input_details[0]
    #int8 = input['dtype'] == np.int8
    int8 = input['dtype'] == np.int8  # is TFLite quantized uint8 model (np.uint8)
    #int32 = input['dtype'] == np.int32  # is TFLite quantized uint8 model (np.uint8)
    #print('input[dtype] : {}'.format(input['dtype']))
    if int8:
        #print('is TFLite quantized uint8 model')
        scale, zero_point = input['quantization']
        #im = (im / scale + zero_point).astype(np.uint8)  # de-scale
        im = (im / scale + zero_point).astype(np.int8)  # de-scale
        #im = im.astype(np.int8)
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
        if x.shape[1]==isize:
            #print('get out images')
            
            scale, zero_point = output['quantization']
            
            x = (x.astype(np.float32)-zero_point) * scale  # re-scale
            #x = x.astype(np.float32)
            #x = tf.squeeze(x)
            #x = x.numpy()
            
            
            gen_img = x*255
            
            gen_img_for_loss = np.squeeze(gen_img)
            gen_img = cv2.cvtColor(gen_img_for_loss, cv2.COLOR_RGB2BGR)
            #print('after squeeze & numpy x : {}'.format(x))
            '''
            if save_image:
                #cv2.imshow('out_image',gen_img)
                filename = 'out_image_' + str(cnt) + '.jpg'
                file_path = os.path.join(save_gen_image_dir,filename)
                cv2.imwrite(file_path,gen_img)
                '''
                #cv2.waitKey(10)
            #gen_img = renormalize(gen_img)
            #gen_img = tf.transpose(gen_img, perm=[0,1,2])
            #plt.imshow(gen_img)
            #plt.show()
        else:
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
    latent_i = y[0]
    latent_o = y[2]
    if SHOW_LOG:
        print('latent_i : {}'.format(latent_i))
        print('latent_o : {}'.format(latent_o))
    _g_loss = g_loss(input_img/255.0, gen_img_for_loss/255.0, latent_i, latent_o)
    
    
    _g_loss_str = str(int(_g_loss))
    if save_image:
        #cv2.imshow('ori_image',im)
        
        save_ori_image_dir = os.path.join('./runs/detect',name,'ori_images')
        save_gen_image_dir = os.path.join('./runs/detect',name,'gen_images')
        os.makedirs(save_ori_image_dir,exist_ok=True)
        os.makedirs(save_gen_image_dir,exist_ok=True)
        
        filename = 'ori_image_' + str(cnt) + '.jpg'
        file_path = os.path.join(save_ori_image_dir, _g_loss_str, filename)
        file_dir = os.path.join(save_ori_image_dir, _g_loss_str)
        os.makedirs(file_dir,exist_ok=True)
        cv2.imwrite(file_path,im_ori)
        
        filename = 'out_image_' + str(cnt) + '.jpg'
        file_path = os.path.join(save_gen_image_dir, _g_loss_str, filename)
        file_dir = os.path.join(save_gen_image_dir, _g_loss_str)
        os.makedirs(file_dir,exist_ok=True)
        cv2.imwrite(file_path,gen_img)
        
        
    
    #_g_loss = 888
    if SHOW_LOG:
        print('g_loss : {}'.format(_g_loss))
    #print(y)
    return _g_loss, gen_img

def g_loss(input_img, gen_img, latent_i, latent_o):
    #from tensorflow.keras import losses as losses
    def l1_loss(A,B):
        return np.mean(np.abs(A-B))
    def l2_loss(A,B):
        return np.mean((A-B)*(A-B))
    #import tensorflow as tf
    # tf loss
    #l2_loss = tf.keras.losses.MeanSquaredError()
    #l1_loss = tf.keras.losses.MeanAbsoluteError()
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

def plot_loss_distribution(SHOW_MAX_NUM,positive_loss,defeat_loss,name):
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
    filename = str(name) + '.jpg'
    file_path = os.path.join('./runs/detect',filename)
    plt2.savefig(file_path)
    plt2.show()

#https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib
def plot_two_loss_histogram(normal_list, abnormal_list, name):
    import numpy
    from matplotlib import pyplot
    bins = numpy.linspace(0, 10, 100)
    pyplot.hist(normal_list, bins, alpha=0.5, label='normal')
    pyplot.hist(abnormal_list, bins, alpha=0.5, label='abnormal')
    pyplot.legend(loc='upper right')
    os.makedirs('./runs/detect',exist_ok=True)
    filename = str(name) + '.jpg'
    file_path = os.path.join('./runs/detect',filename)
    pyplot.savefig(file_path)
    pyplot.show()

def Analysis_two_list(normal_list, abnormal_list, name):
    import math
    import numpy
    normal_count_list = [0]*13
    abnormal_count_list = [0]*13
    for i in range(len(normal_list)):
        normal_count_list[int(normal_list[i])]+=1
    print('normal_count_list')
    for i in range(len(normal_count_list)):
        print('{}: {}'.format(i,normal_count_list[i]))
    
    for i in range(len(abnormal_list)):
        abnormal_count_list[int(abnormal_list[i])]+=1
    print('abnormal_count_list')
    for i in range(len(abnormal_count_list)):
        print('{}: {}'.format(i,abnormal_count_list[i]))
    
    overlap_normal_count = 0
    overlap_abnormal_count = 0
    overlap_count = 0
    for i in range(len(normal_count_list)):
        if normal_count_list[i]!=0 and abnormal_count_list[i]!=0:
            overlap_normal_count += normal_count_list[i]
            overlap_abnormal_count += abnormal_count_list[i]
            overlap_count += min(normal_count_list[i],abnormal_count_list[i])
    print('overlap_normal_count: {}'.format(overlap_normal_count))
    print('overlap_abnormal_count: {}'.format(overlap_abnormal_count))
    print('overlap_count: {}'.format(overlap_count))
    
    from matplotlib import pyplot
    bins = numpy.linspace(0, 13, 100)
    pyplot.hist(normal_list, bins, alpha=0.5, label='normal')
    pyplot.hist(abnormal_list, bins, alpha=0.5, label='abnormal')
    pyplot.legend(loc='upper right')
    os.makedirs('./runs/detect',exist_ok=True)
    filename = str(name) + '.jpg'
    file_path = os.path.join('./runs/detect',filename)
    pyplot.savefig(file_path)
    pyplot.show()
       

def infer_python(img_dir,interpreter,SHOW_MAX_NUM,save_image=False,name='normal',isize=32):
    import glob
    image_list = glob.glob(os.path.join(img_dir,'*.jpg'))
    loss_list = []
    cnt = 0
    for image_path in image_list:
        print(image_path)
        cnt+=1
        
        if cnt<=SHOW_MAX_NUM:
            loss,gen_img = detect_image(w, image_path, interpreter=interpreter, tflite=False,edgetpu=True, save_image=save_image, cnt=cnt, name=name,isize=isize)
            print('{} loss: {}'.format(cnt,loss))
            loss_list.append(loss)
    
    
    return loss_list


def Analysis_two_list_UserDefineLossTH(normal_list, abnormal_list, name, user_loss_list=None):
    show_log = False
    import math
    import numpy
    normal_count_list = [0]*len(user_loss_list)
    abnormal_count_list = [0]*len(user_loss_list)

    user_loss_list = sorted(user_loss_list)

    if show_log:
        print('normal_list : {}'.format(normal_list))
        print('abnormal_list : {}'.format(abnormal_list))
        print('user_loss_list : {}'.format(user_loss_list))

    for i in range(len(user_loss_list)):
        for j in range(len(normal_list)):
            if (i+1) < len(user_loss_list):
                if normal_list[j] >= user_loss_list[i] and  normal_list[j] < user_loss_list[i+1]:
                    normal_count_list[i]+=1
            else:
                if normal_list[j] >= user_loss_list[i]:
                    normal_count_list[i]+=1

    for i in range(len(user_loss_list)):
        for j in range(len(abnormal_list)):
            if (i+1) < len(user_loss_list):
                if abnormal_list[j] >= user_loss_list[i] and  abnormal_list[j] < user_loss_list[i+1]:
                    abnormal_count_list[i]+=1
            else:
                if abnormal_list[j] >= user_loss_list[i]:
                    abnormal_count_list[i]+=1
            
    normal_acc,abnormal_acc = Get_lossTH_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list, user_loss_list)

    print('user_loss_list: {}'.format(user_loss_list))

    print('normal_count_list:') 
    for i in range(len(user_loss_list)):
        print('{} : {}'.format(user_loss_list[i], normal_count_list[i]))
        
    print('abnormal_count_list:')
    for i in range(len(user_loss_list)):
        print('{} : {}'.format(user_loss_list[i], abnormal_count_list[i]))
        
        
    #print('normal_count_list: {}'.format(normal_count_list))
    #print('abnormal_count_list: {}'.format(abnormal_count_list))

    return normal_count_list,abnormal_count_list,normal_acc,abnormal_acc

def Get_lossTH_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list, user_loss_list):
    normal_acc_list,abnormal_acc_list=[0.0]*len(user_loss_list),[0.0]*len(user_loss_list)

    for i in range(len(user_loss_list)):
        normal_acc,abnormal_acc = Analysis_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list,user_loss_list[i],user_loss_list)
              
        normal_acc_list[i] = normal_acc
        abnormal_acc_list[i] = abnormal_acc
        
    for i in range(len(user_loss_list)):
        print('loss {} ,normal acc: {} ,abnormal acc{}'.format(user_loss_list[i],normal_acc_list[i],abnormal_acc_list[i]))
        
    return normal_acc,abnormal_acc

def Analysis_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list,loss_th=3.0, user_loss_list=None):
    show_log = False
    normal_correct_cnt = 0
    total_normal_cnt = 0
    for i in range(len(normal_count_list)):
        total_normal_cnt+=normal_count_list[i]
        if user_loss_list[i] < loss_th:
            normal_correct_cnt+=normal_count_list[i]
    if show_log:
        print('normal_correct_cnt: {}'.format(normal_correct_cnt))
        print('total_normal_cnt: {}'.format(total_normal_cnt))
    if total_normal_cnt == 0:
        normal_acc = 0.0
    else:
        normal_acc = float(normal_correct_cnt/total_normal_cnt)

    total_abnormal_cnt = 0
    abnormal_correct_cnt = 0
    for i in range(len(abnormal_count_list)):
        total_abnormal_cnt+=abnormal_count_list[i]
        if user_loss_list[i] >= loss_th:
            abnormal_correct_cnt+=abnormal_count_list[i]
    if show_log:
        print('abnormal_correct_cnt : {}'.format(abnormal_correct_cnt))
        print('total_abnormal_cnt: {}'.format(total_abnormal_cnt))
    if total_abnormal_cnt==0:
        abnormal_acc = 0
    else:
        abnormal_acc = float(abnormal_correct_cnt / total_abnormal_cnt)


    return normal_acc,abnormal_acc

if __name__=="__main__":
    PYCORAL = False
    DETECT = False
    DETECT_IMAGE = False
    INFER = True
    Analysis = True
    if DETECT:
        w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        get_interpreter(w,tflite=False,edgetpu=True)
    if PYCORAL:
        Pycoral_Edgetpu()
        
    if DETECT_IMAGE:
        save_image = True
        #im = r'/home/ali/Desktop/factory_data/crops_2cls_small/line/ori_video_ver21913.jpg'
        im = r'/home/ali/Desktop/factory_data/crops_2cls_small/line/ori_video_ver22856.jpg'
        #im = r'/home/ali/Desktop/factory_data/crops_2cls_small/noline/ori_video_ver244.jpg'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        #w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104.tflite'
        w = r'/home/ali/Desktop/GANomaly-tf2/export_model/64-nz100-ndf16-ngf16/64-nz100-ndf16-ngf16-20221115-G-int8_edgetpu.tflite'
        #w = r'/home/ali/Desktop/GANomaly-tf2/export_model/32nz100-20221111-G-int8.tflite'
        interpreter = Pycoral_Edgetpu(w)
        loss, gen_image = detect_image(w, im, interpreter=interpreter, tflite=False,edgetpu=True, save_image=True, cnt=1, name='normal',isize=64)
        
        
    if INFER:
        #import tensorflow as tf
        test_data_dir = r'/home/ali/Desktop/factory_data/crops_line/line'
        abnormal_test_data_dir = r'/home/ali/Desktop/factory_data/defect_aug/noline'
        (img_height, img_width) = (32,32)
        isize=32
        batch_size_ = 1
        shuffle = False
        SHOW_MAX_NUM = 7800
        save_image=True
        w = r'/home/ali/Desktop/GANomaly-tf2/export_model/32-nz100-ndf64-ngf64/ckpt-32-nz100-ndf64-ngf64-20221124-G-int8_edgetpu.tflite'
        interpreter = get_interpreter(w,tflite=False,edgetpu=True)
        line_loss = infer_python(test_data_dir,interpreter,SHOW_MAX_NUM,save_image=save_image, name='normal-20221124', isize=isize)
        
        noline_loss = [5]*7800
        #noline_loss = infer_python(abnormal_test_data_dir,interpreter,SHOW_MAX_NUM, save_image=save_image, name='abnormal-20221124',isize=isize)
        plot_loss_distribution(SHOW_MAX_NUM,line_loss,noline_loss,'loss_di_int8_32nz100-20221124')
        plot_two_loss_histogram(line_loss,noline_loss,'line_noline_int8_32nz100-20221124')
        Analysis = False
        if Analysis:
            #Analysis_two_list(line_loss, noline_loss, 'count-histogram-20221123')
            user_loss_list = [0,0.25,0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.75,2.0,3.0,4.0,5.0,6.0]
            #user_loss_list = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]
            print('len(user_loss_list) : {}'.format(len(user_loss_list)))
            Analysis_two_list_UserDefineLossTH(line_loss, noline_loss, 'count-histogram-20221124', user_loss_list)
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
        
    