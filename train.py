import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
import os
import time
import numpy as np
import cv2
from model import GANomaly

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "buffer size for pseudo shuffle")
flags.DEFINE_integer("batch_size", 32, "batch_size")
flags.DEFINE_integer("isize", 32, "input size")
flags.DEFINE_string("ckpt_dir", 'ckpt', "checkpoint folder")
flags.DEFINE_integer("nz", 100, "latent dims")
flags.DEFINE_integer("nc", 3, "input channels")
flags.DEFINE_integer("ndf", 64, "number of discriminator's filters")
flags.DEFINE_integer("ngf", 64, "number of generator's filters")
flags.DEFINE_integer("extralayers", 0, "extralayers for both G and D")
flags.DEFINE_list("encdims", None, "Layer dimensions of the encoder and in reverse of the decoder."
                                   "If given, dense encoder and decoders are used.")
flags.DEFINE_integer("niter",100,"number of training epochs")
flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("w_adv", 1., "Adversarial loss weight")
flags.DEFINE_float("w_con", 50., "Reconstruction loss weight")
flags.DEFINE_float("w_enc", 1., "Encoder loss weight")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_string("dataset", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_line', "name of dataset")
#flags.DEFINE_string("dataset", 'cifar10', "name of dataset")
flags.DEFINE_string("dataset_test", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_2cls_6000', "name of dataset")
flags.DEFINE_string("dataset_infer", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_line_6000', "name of dataset")
flags.DEFINE_string("dataset_infer_abnormal", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/defect_aug', "name of dataset")
DATASETS = ['mnist', 'cifar10']
'''
flags.register_validator('dataset',
                         lambda name: name in DATASETS,
                         message='--dataset must be {}'.format(DATASETS))
'''
flags.DEFINE_integer("anomaly", 5, "the anomaly idx")
flags.mark_flag_as_required('anomaly')
flags.mark_flag_as_required('isize')
flags.mark_flag_as_required('nc')

def batch_resize(imgs, size: tuple):
    img_out = np.empty((imgs.shape[0], ) + size)
    for i in range(imgs.shape[0]):
        img_out[i] = cv2.resize(imgs[i], size, interpolation=cv2.INTER_CUBIC)
    return img_out

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def main(_):
    show_loss_histogram = False
    show_loss_distribution = False
    infer_one_image = False
    infer_images = True
    show_img = False
    TRAIN = False
    opt = FLAGS
    # logging
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    if FLAGS.log_dir:
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
    
    # dataset
    if opt.dataset=='mnist':
        data_train, data_test = tf.keras.datasets.mnist.load_data()
        x_train, y_train = data_train
        x_test, y_test = data_test
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.reshape([-1,])
        y_test = y_test.reshape([-1,])
        # resize to (32, 32)
        if opt.dataset=='mnist':
            x_train = batch_resize(x_train, (32, 32))[..., None]
            x_test = batch_resize(x_test, (32, 32))[..., None]
        # normalization
        mean = x_train.mean()
        stddev = x_train.std()
        x_train = (x_train - mean) / stddev
        x_test = (x_test - mean) / stddev
        logging.info('{}, {}'.format(x_train.shape, x_test.shape))
        # define abnoraml data and normal
        # training data only contains normal
        x_train = x_train[y_train != opt.anomaly, ...]
        y_train = y_train[y_train != opt.anomaly, ...]
        y_test = (y_test == opt.anomaly).astype(np.float32)
        # tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
            opt.batch_size, drop_remainder=True)
        test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)
        test_dataset = test_dataset.shuffle(buffer_size=len(y_test))
    elif opt.dataset=='cifar10':
        data_train, data_test = tf.keras.datasets.cifar10.load_data()
        x_train, y_train = data_train
        x_test, y_test = data_test
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.reshape([-1,])
        y_test = y_test.reshape([-1,])
        # resize to (32, 32)
        if opt.dataset=='mnist':
            x_train = batch_resize(x_train, (32, 32))[..., None]
            x_test = batch_resize(x_test, (32, 32))[..., None]
        # normalization
        mean = x_train.mean()
        stddev = x_train.std()
        x_train = (x_train - mean) / stddev
        x_test = (x_test - mean) / stddev
        logging.info('{}, {}'.format(x_train.shape, x_test.shape))
        # define abnoraml data and normal
        # training data only contains normal
        x_train = x_train[y_train != opt.anomaly, ...]
        y_train = y_train[y_train != opt.anomaly, ...]
        y_test = (y_test == opt.anomaly).astype(np.float32)
        # tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(
            opt.batch_size, drop_remainder=True)
        test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)
        test_dataset = test_dataset.shuffle(buffer_size=len(y_test))
    else:
        '''
        Use Custom dataaset
        tf.keras.utils.image_dataset_from_directory(
                                        directory,
                                        labels='inferred',
                                        label_mode='int',
                                        class_names=None,
                                        color_mode='rgb',
                                        batch_size=32,
                                        image_size=(256, 256),
                                        shuffle=True,
                                        seed=None,
                                        validation_split=None,
                                        subset=None,
                                        interpolation='bilinear',
                                        follow_links=False,
                                        crop_to_aspect_ratio=False,
                                        **kwargs
                                    )

        '''
        #raise NotImplementError
        train_data_dir = opt.dataset
        img_height = opt.isize
        img_width = opt.isize
        batch_size = opt.batch_size
        train_dataset = tf.keras.utils.image_dataset_from_directory(
          train_data_dir,
          #validation_split=0.1,
          #subset="training",
          #seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size
          )
        
        train_dataset = train_dataset.map(process)
        
        if TRAIN == False:
            if show_img==True:
                batch_size_=opt.batch_size
                shuffle=True
            else:
                batch_size_=1
                shuffle=False
        else:
            batch_size_=opt.batch_size
            shuffle=True
            
        val_data_dir = opt.dataset_test
        print(val_data_dir)
        print(batch_size_)
        test_dataset = tf.keras.utils.image_dataset_from_directory(
          val_data_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          #seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
          #interpolation='nearest')
        
        test_dataset = test_dataset.map(process)
        
        infer_data_dir = opt.dataset_infer
        print(infer_data_dir)
        print(batch_size_)
        infer_dataset = tf.keras.utils.image_dataset_from_directory(
          infer_data_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          #seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
          #interpolation='nearest')
        
        infer_dataset = infer_dataset.map(process)
        
        infer_data_abnormal_dir = opt.dataset_infer_abnormal
        print(infer_data_abnormal_dir)
        print(batch_size_)
        infer_dataset_abnormal = tf.keras.utils.image_dataset_from_directory(
          infer_data_abnormal_dir,
          #validation_split=0.1,
          #subset="validation",
          shuffle=shuffle,
          #seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size_)
          #interpolation='nearest')
    
        infer_dataset_abnormal = infer_dataset_abnormal.map(process)
    
    ganomaly = GANomaly(opt,
                        train_dataset,
                        valid_dataset=None,
                        test_dataset=test_dataset)
    if TRAIN:
        # training
        ganomaly.fit(opt.niter)
    
        # evaluating
        ganomaly.evaluate_best(test_dataset)
    else:
        if show_img:
            SHOW_MAX_NUM = 10
        else:
            SHOW_MAX_NUM = 6000
            
        if show_loss_histogram or show_loss_distribution or show_img:
            ANDY_CODE = False
            #input()
            if ANDY_CODE:
                image_path = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/test20221117/line/ori_video_ver21189.jpg'
                #image_path = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops/noline/ori_video_ver235.jpg'
                image = cv2.imread(image_path)
                
                image = cv2.resize(image,(opt.isize,opt.isize))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                #image = tf.expand_dims(image, axis=0)
                
                image = image/255.0
                image = image[np.newaxis, ...].astype(np.float32)
                
                #print('img_tf.shape = {}'.format(image_dbg.shape))
                #print('img_op.shape = {}'.format(image.shape))
                
                #diff = image_dbg[0:10, 0, 0] - image[0:10, 0, 0]
                #diff = np.sum(image_dbg - image)
                #print('diff = {}'.format(diff))
    
                ganomaly.infer_dbg(infer_dataset,image,1,show_img,'normal')
                input()
            else:
                positive_loss = ganomaly.infer(infer_dataset,SHOW_MAX_NUM,show_img,'normal')
                defeat_loss = ganomaly.infer(infer_dataset_abnormal,SHOW_MAX_NUM,show_img,'abnormal')
        
        if show_loss_distribution:
            name = str(opt.isize) + 'nz' + str(opt.nz) + '_loss' + str(SHOW_MAX_NUM) + '-tf' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            ganomaly.plot_loss_distribution( SHOW_MAX_NUM,positive_loss,defeat_loss,name)
        if show_loss_histogram:
            positive_name = 'positive' + str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-tf' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            negative_name = 'negative' + str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-tf' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            ganomaly.plot_loss_histogram(positive_loss,positive_name)
            ganomaly.plot_loss_histogram(defeat_loss,negative_name)
            two_name =  str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-tf' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            ganomaly.plot_two_loss_histogram(positive_loss,defeat_loss,two_name)
            
            analysis_name = str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-tf' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf) + 'analysis'
            normal_count_list,abnormal_count_list,normal_acc,abnormal_acc = ganomaly.Analysis_two_list(positive_loss, defeat_loss, analysis_name)
        
        if infer_one_image:
            save_img = True
            show_log = True
            image_path = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_line/line/ori_video_ver22911.jpg'
            #image_path = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops/noline/ori_video_ver235.jpg'
            image = cv2.imread(image_path)
            
            image = cv2.resize(image,(opt.isize,opt.isize))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #image = tf.expand_dims(image, axis=0)
            
            image = image/255.0
            image = image[np.newaxis, ...].astype(np.float32)
            print('input image shape for inference : {}'.format(image.shape))
            print('Start inference one image : \n {}'.format(str(image_path)))
            g_loss,out_image = ganomaly.infer_cropimage(image, save_img=save_img, show_log=show_log, name='factory_data_20221115', cnt=1)
            print('finish inference one image')
        if infer_images:
            img_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/crops_line_6000/line'
            normal_name =  str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-opencv-normal' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            loss_normal_list = ganomaly.infer_python(img_dir,SHOW_MAX_NUM,save_image=True,name=normal_name,isize=opt.isize)
            
            img_dir = r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/factory_data/defect_aug/noline'
            abnormal_name =  str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-opencv-abnormal' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            loss_abnormal_list = ganomaly.infer_python(img_dir,SHOW_MAX_NUM,save_image=True,name=abnormal_name,isize=opt.isize)
            
            print('loss_normal_list: {}'.format(loss_normal_list))
            print('loss_abnormal_list: {}'.format(loss_abnormal_list))
            loss_name =  str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-opencv-loss' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            ganomaly.plot_loss_distribution( SHOW_MAX_NUM,loss_normal_list,loss_abnormal_list,loss_name)
            print('Start plot_two_loss_histogram')
            hist_name =  str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-opencv-histogram' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            ganomaly.plot_two_loss_histogram(loss_normal_list,loss_abnormal_list,hist_name)
            print('Finish plot_two_loss_histogram')
            analysis_name =  str(opt.isize) + 'nz' + str(opt.nz) + '-' + str(SHOW_MAX_NUM) + '-opencv-analysis' + '-ndf' + str(opt.ndf) + '-ngf' + str(opt.ngf)
            ganomaly.Analysis_two_list(loss_normal_list, loss_abnormal_list, analysis_name)
            
    #print(loss_list)
    #print(loss_abnormal_list)
if __name__ == '__main__':
    app.run(main)
