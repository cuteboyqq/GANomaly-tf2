import time
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import metrics
from absl import logging
import matplotlib.pyplot as plt
import os
class Conv_BN_Act(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 ks,
                 act_type,
                 is_bn=True,
                 padding='same',
                 strides=1,
                 conv_tran=False):
        super(Conv_BN_Act, self).__init__()
        if conv_tran:
            self.conv = layers.Conv2DTranspose(filters,
                                               ks,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=False)
        else:
            self.conv = layers.Conv2D(filters,
                                      ks,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=False)

        self.is_bn = is_bn
        if is_bn:
            self.bn = layers.BatchNormalization(epsilon=1e-05, momentum=0.9)

        if act_type == 'LeakyReLU':
            self.act = layers.LeakyReLU(alpha=0.2)
            self.erase_act = False
        elif act_type == 'ReLU':
            self.act = layers.ReLU()
            self.erase_act = False
        elif act_type == 'Tanh':
            self.act = layers.Activation(tf.tanh)
            self.erase_act = False
        elif act_type == '':
            self.erase_act = True
        else:
            raise ValueError

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.is_bn else x
        x = x if self.erase_act else self.act(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """ DCGAN ENCODER NETWORK
    """
    def __init__(self,
                 isize,
                 nz,
                 nc,
                 ndf,
                 n_extra_layers=0,
                 output_features=False):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ndf(int): num of discriminator(Encoder) filters
        """
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.in_block = Conv_BN_Act(filters=ndf,
                                    ks=4,
                                    act_type='LeakyReLU',
                                    is_bn=False,
                                    strides=2)
        csize, cndf = isize / 2, ndf

        self.extra_blocks = []
        for t in range(n_extra_layers):
            extra = Conv_BN_Act(filters=cndf, ks=3, act_type='LeakyReLU')
            self.extra_blocks.append(extra)

        self.body_blocks = []
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            body = Conv_BN_Act(filters=out_feat,
                               ks=4,
                               act_type='LeakyReLU',
                               strides=2)
            self.body_blocks.append(body)
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        self.output_features = output_features
        self.out_conv = layers.Conv2D(filters=nz,
                                      kernel_size=4,
                                      padding='valid')

    def call(self, x):
        x = self.in_block(x)
        for block in self.extra_blocks:
            x = block(x)
        for block in self.body_blocks:
            x = block(x)
        last_features = x
        out = self.out_conv(last_features)
        if self.output_features:
            return out, last_features
        else:
            return out


class DenseEncoder(tf.keras.layers.Layer):
    def __init__(self, layer_dims, out_size=None, output_features=False, hidden_activation="selu", p_dropout=.2):
        """
        Params:
            layer_dims(Tuple[int]): dense layer dimensions
            out_size(int): overwrite the output size of the last layer; use layer_dims[-1] if None
            output_features(bool): use intermediate activation
            hidden_activation(Union[str,tf.keras.layers.Activation]): activation of the hidden layers
            p_dropout(float): dropout between the hidden layers
        """
        super(DenseEncoder, self).__init__()

        # Config
        self.output_features = output_features

        # Layers
        self.in_block = tf.keras.layers.Dense(layer_dims[0], activation=hidden_activation)
        self.body_blocks = []
        self.body_blocks.append(tf.keras.layers.Dropout(p_dropout))
        for cur_dim in layer_dims[1:-1]:
            self.body_blocks.append(tf.keras.layers.Dense(cur_dim, activation=hidden_activation))
            self.body_blocks.append(tf.keras.layers.Dropout(p_dropout))

        # Override the output dimension if given
        if out_size is not None:
            self.out_act = tf.keras.layers.Dense(out_size)
        else:
            self.out_act = tf.keras.layers.Dense(layer_dims[-1])

    def call(self, x):
        x = self.in_block(x)
        for block in self.body_blocks:
            x = block(x)
        last_features = x
        out = self.out_act(last_features)
        if self.output_features:
            return out, last_features
        else:
            return out


class Decoder(tf.keras.layers.Layer):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ngf(int): num of Generator(Decoder) filters
        """
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        self.in_block = Conv_BN_Act(filters=cngf,
                                    ks=4,
                                    act_type='ReLU',
                                    padding='valid',
                                    conv_tran=True)

        csize, _ = 4, cngf
        self.body_blocks = []
        while csize < isize // 2:
            body = Conv_BN_Act(filters=cngf // 2,
                               ks=4,
                               act_type='ReLU',
                               strides=2,
                               conv_tran=True)
            self.body_blocks.append(body)
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        self.extra_blocks = []
        for t in range(n_extra_layers):
            extra = Conv_BN_Act(filters=cngf,
                                ks=3,
                                act_type='ReLU',
                                conv_tran=True)
            self.extra_blocks.append(extra)

        self.out_block = Conv_BN_Act(filters=nc,
                                     ks=4,
                                     act_type='Tanh',
                                     strides=2,
                                     is_bn=False,
                                     conv_tran=True)

    def call(self, x):
        x = self.in_block(x)
        for block in self.body_blocks:
            x = block(x)
        for block in self.extra_blocks:
            x = block(x)
        x = self.out_block(x)
        return x


class DenseDecoder(tf.keras.layers.Layer):
    def __init__(self, isize, layer_dims, hidden_activation="selu", p_dropout=.2):
        """
        Params:
            isize(int): input size
            layer_dims(Tuple[int]): dense layer dimensions
            hidden_activation(Union[str,tf.keras.layers.Activation]): activation of the hidden layers
            p_dropout(float): dropout between the hidden layers
        """
        super(DenseDecoder, self).__init__()

        # Layers
        self.in_block = tf.keras.layers.Dense(layer_dims[0], activation=hidden_activation)
        self.body_blocks = []
        self.body_blocks.append(tf.keras.layers.Dropout(p_dropout))
        for cur_dim in layer_dims[1:]:
            self.body_blocks.append(tf.keras.layers.Dense(cur_dim, activation=hidden_activation))
            self.body_blocks.append(tf.keras.layers.Dropout(p_dropout))

        self.out_block = tf.keras.layers.Dense(isize, activation="tanh")

    def call(self, x):
        x = self.in_block(x)
        for block in self.body_blocks:
            x = block(x)
        x = self.out_block(x)
        return x


class NetG(tf.keras.Model):
    def __init__(self, opt):
        super(NetG, self).__init__()

        # Use the dense encoder-decoder pair when the dimensions are given
        if opt.encdims:
            self.encoder1 = DenseEncoder(opt.encdims)
            self.decoder = DenseDecoder(opt.isize, tuple(reversed(opt.encdims[:-1])))
            self.encoder2 = DenseEncoder(opt.encdims)
        else:
            self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.extralayers)
            self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.extralayers)
            self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.extralayers)

    def call(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        latent_o = self.encoder2(gen_img)
        return latent_i, gen_img, latent_o

    def num_params(self):
        return sum(
            [np.prod(var.shape.as_list()) for var in self.trainable_variables])


class NetD(tf.keras.Model):
    """ DISCRIMINATOR NETWORK
    """
    def __init__(self, opt):
        super(NetD, self).__init__()

        # Use the dense encoder when the dimensions are given
        if opt.encdims:
            self.encoder = DenseEncoder(opt.encdims, out_size=1, output_features=True)
        else:
            self.encoder = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.extralayers, output_features=True)

        self.sigmoid = layers.Activation(tf.sigmoid)

    def call(self, x):
        output, last_features = self.encoder(x)
        output = self.sigmoid(output)
        return output, last_features


class GANRunner:
    def __init__(self,
                 G,
                 D,
                 best_state_key,
                 best_state_policy,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None,
                 save_path='ckpt/'):
        self.G = G
        self.D = D
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.num_ele_train = self._get_num_element(self.train_dataset)
        self.best_state_key = best_state_key
        self.best_state_policy = best_state_policy
        self.best_state = 1e-9 if self.best_state_policy == max else 1e9
        self.save_path = save_path

    def train_step(self, x, y):
        raise NotImplementedError

    def validate_step(self, x, y):
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def _get_num_element(self, dataset):
        num_elements = 0
        for _ in dataset:
            num_elements += 1
        return num_elements

    def fit(self, num_epoch, best_state_ths=None):
        self.best_state = self.best_state_policy(
            self.best_state,
            best_state_ths) if best_state_ths is not None else self.best_state
        for epoch in range(num_epoch):
            start_time = time.time()
            # train one epoch
            G_losses = []
            D_losses = []
            with tqdm(total=self.num_ele_train, leave=False) as pbar:
                for step, (x_batch_train,
                           y_batch_train) in enumerate(self.train_dataset):
                    loss = self.train_step(x_batch_train, y_batch_train)
                    G_losses.append(loss[0].numpy())
                    D_losses.append(loss[1].numpy())
                    pbar.update(1)
                G_losses = np.array(G_losses).mean()
                D_losses = np.array(D_losses).mean()
                speed = step * len(x_batch_train) / (time.time() - start_time)
                logging.info(
                    'epoch: {}, G_losses: {:.4f}, D_losses: {:.4f}, samples/sec: {:.4f}'
                    .format(epoch, G_losses, D_losses, speed))

            # validate one epoch
            if self.valid_dataset is not None:
                G_losses = []
                D_losses = []
                for step, (x_batch_train,
                           y_batch_train) in enumerate(self.valid_dataset):
                    loss = self.validate_step(x_batch_train, y_batch_train)
                    G_losses.append(loss[0].numpy())
                    D_losses.append(loss[1].numpy())
                G_losses = np.array(G_losses).mean()
                D_losses = np.array(D_losses).mean()
                logging.info(
                    '\t Validating: G_losses: {}, D_losses: {}'.format(
                        G_losses, D_losses))

            # evaluate on test_dataset
            if self.test_dataset is not None:
                dict_ = self.evaluate(self.test_dataset)
                log_str = '\t Testing:'
                for k, v in dict_.items():
                    log_str = log_str + '   {}: {:.4f}'.format(k, v)
                state_value = dict_[self.best_state_key]
                self.best_state = self.best_state_policy(
                    self.best_state, state_value)
                if self.best_state == state_value:
                    log_str = '*** ' + log_str + ' ***'
                    self.save_best()
                logging.info(log_str)

    def save(self, path):
        #self.G.save_weights(self.save_path + 'G')
        #self.D.save_weights(self.save_path + 'D')
        #tf.saved_model.save(model, "saved_model_keras_dir")
        self.G.save(self.save_path + 'G')
        self.D.save(self.save_path + 'D')

    def load(self, path):
        #self.G.load_weights(self.save_path + 'G')
        #self.D.load_weights(self.save_path + 'D')
        self.G = tf.keras.models.load_model(self.save_path + 'G')
        self.D = tf.keras.models.load_model(self.save_path + 'D')

    def save_best(self):
        self.save(self.save_path + 'best') 

    def load_best(self):
        self.load(self.save_path + 'best')


class GANomaly(GANRunner):
    def __init__(self,
                 opt,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None):
        self.opt = opt
        self.G = NetG(self.opt)
        self.D = NetD(self.opt)
        super(GANomaly, self).__init__(self.G,
                                       self.D,
                                       best_state_key='roc_auc',
                                       best_state_policy=max,
                                       train_dataset=train_dataset,
                                       valid_dataset=valid_dataset,
                                       test_dataset=test_dataset)
        self.D(tf.keras.Input(shape=[opt.isize] if opt.encdims else [opt.isize, opt.isize, opt.nc]))
        self.D_init_w_path = '/tmp/D_init'
        self.D.save_weights(self.D_init_w_path)

        # label
        self.real_label = tf.ones([
            self.opt.batch_size,
        ], dtype=tf.float32)
        self.fake_label = tf.zeros([
            self.opt.batch_size,
        ], dtype=tf.float32)

        # loss
        l2_loss = tf.keras.losses.MeanSquaredError()
        l1_loss = tf.keras.losses.MeanAbsoluteError()
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        # optimizer
        self.d_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)

        # adversarial loss (use feature matching)
        self.l_adv = l2_loss
        # contextual loss
        self.l_con = l1_loss
        # Encoder loss
        self.l_enc = l2_loss
        # discriminator loss
        self.l_bce = bce_loss
    
        self.show_max_num = 5
        
    def renormalize(self, tensor):
        minFrom= tf.math.reduce_min(tensor)
        maxFrom= tf.math.reduce_max(tensor)
        minTo = 0
        maxTo = 1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
    
    
    def infer_cropimage(self, image, save_img=False, show_log=False, name='factory_data', cnt=1, load_model=True):
        if load_model:
            self.load_best()
        def renormalize(tensor,minto, maxto):
                minFrom= tf.math.reduce_min(tensor)
                maxFrom= tf.math.reduce_max(tensor)
                #minFrom= tensor.min() #tf.reduce_min
                #maxFrom= tensor.max()
                minTo = minto
                maxTo = maxto
                return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
            
        import cv2
        abnormal = 0
        self.input = image
        self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
        
        #self.gen_img = renormalize(self.gen_img,0,1)
        
        self.pred_real, self.feat_real = self.D(self.input)
        self.pred_fake, self.feat_fake = self.D(self.gen_img)
        g_loss = self.g_loss()
        
        print('{} loss: {}'.format(cnt,g_loss))
        if g_loss>2.0:
            abnormal=1
            print('abnoraml')
        else:
            abnormal=0
            print('normal')
            
        g_loss_str = str(int(g_loss))
        
        if save_img:
            save_ori_image_dir = os.path.join('./runs/detect',name,'ori_images',g_loss_str)
            save_gen_image_dir = os.path.join('./runs/detect',name,'gen_images',g_loss_str)

            os.makedirs(save_ori_image_dir,exist_ok=True)
            os.makedirs(save_gen_image_dir,exist_ok=True)
            
            #ori_image = tf.squeeze(self.input)
            #ori_image = renormalize(ori_image,0,255)
            ori_image = np.squeeze(image)
            ori_image = ori_image*255
            ori_image = cv2.cvtColor(ori_image,cv2.COLOR_RGB2BGR)
            #ori_image = ori_image.cpu().numpy()
            filename = 'ori_image_' + str(cnt) + '.jpg'
            file_path = os.path.join(save_ori_image_dir, filename)
            cv2.imwrite(file_path, ori_image)
            #cv2.imshow('ori_img',ori_image)
            #cv2.waitKey(10)
            out_image = tf.squeeze(self.gen_img)  
            #out_image = renormalize(out_image,0,255)
            #out_image = renormalize(out_image,0,255)
            out_image = out_image.numpy()
            out_image = out_image*255
            out_image = cv2.cvtColor(out_image,cv2.COLOR_RGB2BGR)  
            
            #out_image = out_image.cpu().numpy()
            #out_image = np.squeeze(out_image)
            #out_image = renormalize(out_image)
            
            filename = 'out_image_' + str(cnt) + '.jpg'
            file_path = os.path.join(save_gen_image_dir,filename)
            cv2.imwrite(file_path, out_image)
            #cv2.imshow('gen_img',out_image)
            #cv2.waitKey(10)
        if show_log:
            print('ori image : {}'.format(ori_image.shape))
            print('ori image : {}'.format(ori_image))
            print('for infer : {}'.format(self.input.shape))
            print('for infer : {}'.format(self.input))
            print('out image : {}'.format(out_image.shape))
            print('out image : {}'.format(out_image))
            print('lentent_i : {}'.format(self.latent_i.shape))
            print('lentent_i : {}'.format(self.latent_i))
            print('lentent_o : {}'.format(self.latent_o.shape))
            print('lentent_o : {}'.format(self.latent_o))
            
        return g_loss,out_image
        
    def infer_python(self, img_dir,SHOW_MAX_NUM,save_image=False,name='normal',isize=64):
        import glob
        import cv2
        import numpy as np
        #import torchvision
        #import torch
        #import imageio as iio
        from PIL import Image
        image_list = glob.glob(os.path.join(img_dir,'*.jpg'))
        loss_list = []
        self.load_best()
        cnt = 0
        USE_PIL = False
        USE_OPENCV = True
        for image_path in image_list:
            #print(image_path)
            
            #image = torchvision.io.read_image(image_path)
            if USE_PIL:
                image = Image.open(image_path)
                image = image.convert('RGB')
                image = image.resize((isize,isize))
                image = np.asarray(image)
            cnt+=1
            if USE_OPENCV:
                image = cv2.imread(image_path)
                image = cv2.resize(image,(isize,isize))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                
                
            #image = tf.expand_dims(image, axis=0)
            
            image = image/255.0
           
            #tf.convert_to_tensor(image)
            #image = tf.convert_to_tensor(image)

            #tf.expand_dims(image,axis=0)
            image = image[np.newaxis, ...].astype(np.float32)
            if cnt<=SHOW_MAX_NUM:
                loss,gen_img = self.infer_cropimage(image, save_img=save_image, show_log=False, name=name, cnt=cnt, load_model=False)
                #loss,gen_img = detect_image(w, image_path, interpreter=interpreter, tflite=False,edgetpu=True, save_image=save_image, cnt=cnt, name=name,isize=isize)
                #print('{} loss: {}'.format(cnt,loss))
                loss_list.append(loss.numpy())
        
        
        return loss_list
    
    def infer(self, test_dataset,SHOW_MAX_NUM,show_img,data_type):
        show_num = 0
        self.load_best()
        
        
        loss_list = []
        dataiter = iter(test_dataset)
        #for step, (images, y_batch_train) in enumerate(test_dataset):
        cnt=1
        os.makedirs('./runs/detect',exist_ok=True)
        
        image_dbg = None
        
        while(show_num < SHOW_MAX_NUM):
            images, labels = dataiter.next()
            #image_dbg = images
            #latent_i, fake_img, latent_o = self.G(images)
            self.input = images
            
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss = self.g_loss()
            #g_loss = 0.0
            #print("input")
            #print(self.input)
            #print("gen_img")
            #print(self.gen_img)
            images = self.renormalize(self.input)
            fake_img = self.renormalize(self.gen_img)
            #fake_img = self.gen_img
            images = images.cpu().numpy()
            fake_img = fake_img.cpu().numpy()
            #fake_img = self.gen_img
            #print(fake_img.shape)
            #print(images.shape)
            if show_img:
                plt = self.plot_images(images,fake_img)
                if data_type=='normal':
                    file_name = 'infer_normal' + str(cnt) + '.jpg'
                else:
                    file_name = 'infer_abnormal' + str(cnt) + '.jpg'
                file_path = os.path.join('./runs/detect',file_name)
                plt.savefig(file_path)
                cnt+=1
            if data_type=='normal':
                print('{} normal: {}'.format(show_num,g_loss.numpy()))
            else:
                print('{} abnormal: {}'.format(show_num,g_loss.numpy()))
            loss_list.append(float(g_loss))
            show_num+=1
            #if show_num%20==0:
                #print(show_num)
        return loss_list
    
    def infer_dbg(self, test_dataset, image_opencv, SHOW_MAX_NUM,show_img,data_type):
        

        show_num = 0
        self.load_best()
                
        loss_list = []


        for step, (images, y_batch_train) in enumerate(test_dataset):
 
            #latent_i, fake_img, latent_o = self.G(images)
            self.input = images
            
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss = self.g_loss()
            
            ######
            self.input = image_opencv
            
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss_opencv = self.g_loss()
            
            
            print('tf loss = {}'.format(g_loss))
            print('opencv loss = {}'.format(g_loss_opencv))
            
          

    def plot_images(self,images,outputs):
        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=15, sharex=True, sharey=True, figsize=(25,4))
        # input images on top row, reconstructions on bottom
        for images2, row in zip([images,outputs], axes):     
            for img, ax in zip(images2, row):
                #img = img[:,:,::-1].transpose((2,1,0))
                #print(img)
                ax.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        return plt
    
    def plot_loss_distribution(self, SHOW_MAX_NUM,positive_loss,defeat_loss,name):
        # Importing packages
        import matplotlib.pyplot as plt2
        # Define data values
        x = [i for i in range(SHOW_MAX_NUM)]
        y = positive_loss
        z = defeat_loss
        #print(x)
        #print(positive_loss)
        #print(defeat_loss)
        print('positive_loss: {}'.format(len(positive_loss)))
        print('defeat_loss: {}'.format(len(defeat_loss)))
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
        
    def plot_loss_histogram(self,loss_list, name):
        from matplotlib import pyplot
        import numpy
        bins = numpy.linspace(0, 6, 100)
        pyplot.hist(loss_list, bins=bins, alpha=0.5, label=name)
        os.makedirs('./runs/detect',exist_ok=True)
        filename = str(name) + '.jpg'
        file_path = os.path.join('./runs/detect',filename)
        plt.savefig(file_path)
        plt.show()
    
    #https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib
    def plot_two_loss_histogram(self,normal_list, abnormal_list, name):
        import numpy
        from matplotlib import pyplot
        bins = numpy.linspace(0, 6, 100)
        pyplot.hist(normal_list, bins, alpha=0.5, label='normal')
        pyplot.hist(abnormal_list, bins, alpha=0.5, label='abnormal')
        pyplot.legend(loc='upper right')
        os.makedirs('./runs/detect',exist_ok=True)
        filename = str(name) + '.jpg'
        file_path = os.path.join('./runs/detect',filename)
        plt.savefig(file_path)
        pyplot.show()
        
    
    def Analysis_two_list(self, normal_list, abnormal_list, name):
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
        
        
        normal_acc,abnormal_acc = self.Get_lossTH_Accuracy(normal_count_list,abnormal_count_list)
        
        
        return normal_count_list,abnormal_count_list,normal_acc,abnormal_acc
    
    def Analysis_Accuracy(self, normal_count_list,abnormal_count_list,loss_th=3.0):
        show_log = False
        normal_correct_cnt = 0
        total_normal_cnt = 0
        for i in range(len(normal_count_list)):
            total_normal_cnt+=normal_count_list[i]
            if i < loss_th:
                normal_correct_cnt+=normal_count_list[i]
        if show_log:
            print('normal_correct_cnt: {}'.format(normal_correct_cnt))
            print('total_normal_cnt: {}'.format(total_normal_cnt))
        normal_acc = float(normal_correct_cnt/total_normal_cnt)
        
        total_abnormal_cnt = 0
        abnormal_correct_cnt = 0
        for i in range(len(abnormal_count_list)):
            total_abnormal_cnt+=abnormal_count_list[i]
            if i >= loss_th:
                abnormal_correct_cnt+=abnormal_count_list[i]
        if show_log:
            print('abnormal_correct_cnt : {}'.format(abnormal_correct_cnt))
            print('total_abnormal_cnt: {}'.format(total_abnormal_cnt))
        abnormal_acc = float(abnormal_correct_cnt / total_abnormal_cnt)
        
        
        return normal_acc,abnormal_acc
    
    
    def Get_lossTH_Accuracy(self, normal_count_list,abnormal_count_list):
        normal_acc_list,abnormal_acc_list=[0.0]*10,[0.0]*10
        
        for i in range(len(normal_acc_list)):
            normal_acc,abnormal_acc = self.Analysis_Accuracy(normal_count_list,abnormal_count_list,i)
                  
            normal_acc_list[i] = normal_acc
            abnormal_acc_list[i] = abnormal_acc
            
        for i in range(len(normal_acc_list)):
            print('loss {} ,normal acc: {} ,abnormal acc{}'.format(i,normal_acc_list[i],abnormal_acc_list[i]))
            
        return normal_acc,abnormal_acc
                
        
        
    
    def _evaluate(self, test_dataset):
        an_scores = []
        gt_labels = []
        for step, (x_batch_train, y_batch_train) in enumerate(test_dataset):
            latent_i, gen_img, latent_o = self.G(x_batch_train)
            latent_i, gen_img, latent_o = latent_i.numpy(), gen_img.numpy(
            ), latent_o.numpy()
            error = np.mean((latent_i - latent_o)**2, axis=-1)
            an_scores.append(error)
            gt_labels.append(y_batch_train)
        an_scores = np.concatenate(an_scores, axis=0).reshape([-1])
        gt_labels = np.concatenate(gt_labels, axis=0).reshape([-1])
        return an_scores, gt_labels

    def evaluate(self, test_dataset):
        ret_dict = {}
        an_scores, gt_labels = self._evaluate(test_dataset)
        # normed to [0,1)
        an_scores = (an_scores - np.amin(an_scores)) / (np.amax(an_scores) -
                                                        np.amin(an_scores))
        # AUC
        auc_dict = metrics.roc_auc(gt_labels, an_scores)
        ret_dict.update(auc_dict)
        # Average Precision
        p_r_dict = metrics.pre_rec_curve(gt_labels, an_scores)
        ret_dict.update(p_r_dict)
        return ret_dict

    def evaluate_best(self, test_dataset):
        self.load_best()
        an_scores, gt_labels = self._evaluate(test_dataset)
        # AUC
        _ = metrics.roc_auc(gt_labels, an_scores, show=True)
        # Average Precision
        _ = metrics.pre_rec_curve(gt_labels, an_scores, show=True)

    @tf.function
    def _train_step_autograph(self, x):
        """ Autograph enabled by tf.function could speedup more than 6x than eager mode.
        """
        self.input = x
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            self.latent_i, self.gen_img, self.latent_o = self.G(self.input)
            self.pred_real, self.feat_real = self.D(self.input)
            self.pred_fake, self.feat_fake = self.D(self.gen_img)
            g_loss = self.g_loss()
            d_loss = self.d_loss()

        g_grads = g_tape.gradient(g_loss, self.G.trainable_weights)
        d_grads = d_tape.gradient(d_loss, self.D.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads,
                                             self.G.trainable_weights))
        self.d_optimizer.apply_gradients(zip(d_grads,
                                             self.D.trainable_weights))
        return g_loss, d_loss

    def train_step(self, x, y):
        g_loss, d_loss = self._train_step_autograph(x)
        if d_loss < 1e-5:
            st = time.time()
            self.D.load_weights(self.D_init_w_path)
            logging.info('re-init D, cost: {:.4f} secs'.format(time.time() -
                                                               st))

        return g_loss, d_loss

    def validate_step(self, x, y):
        pass

    def g_loss(self):
        self.err_g_adv = self.l_adv(self.feat_real, self.feat_fake)
        self.err_g_con = self.l_con(self.input, self.gen_img)
        self.err_g_enc = self.l_enc(self.latent_i, self.latent_o)
        g_loss= self.err_g_adv * self.opt.w_adv + \
                self.err_g_con * self.opt.w_con + \
                self.err_g_enc * self.opt.w_enc
        return g_loss

    def d_loss(self):
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)
        d_loss = (self.err_d_real + self.err_d_fake) * 0.5
        return d_loss
