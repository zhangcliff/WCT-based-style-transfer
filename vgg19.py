import os
import tensorflow as tf
import numpy as np
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def encoder(self,inputs,target_layer):
        layer_num =dict(zip(['relu1','relu2','relu3','relu4','relu5'],range(1,6)))[target_layer]
        encode = inputs
        encoder_arg={
                '1':[('conv1_1',64),
                     ('conv1_2',64),
                     ('pool1',64)],
                '2':[('conv2_1',128),
                     ('conv2_2',128),
                     ('pool2',128)],
                '3':[('conv3_1',256),
                     ('conv3_2',256),
                     ('conv3_3',256),
                     ('conv3_4',256),
                     ('pool3',256)],                    
                '4':[('conv4_1',512),
                     ('conv4_2',512),
                     ('conv4_3',512),
                     ('conv4_4',512),
                     ('pool4',512)],
                '5':[('conv5_1',512),
                     ('conv5_2',512),
                     ('conv5_3',512),
                     ('conv5_4',512),]}  
                
        for d in range(1,layer_num+1):
            for layer in encoder_arg[str(d)]:                
                if 'conv' in layer[0] :
                    encode =self.conv_layer(encode,layer[0])
                if 'pool' in layer[0] and d <layer_num :
                    encode = self.max_pool(encode,layer[0])               
        return encode
    
    def encoder_all_layer(self, inputs):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """


        encode = inputs
        conv1_1 = self.conv_layer(encode, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")



        
        return conv1_2 , conv2_2, conv3_4, conv4_4

            
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            filt_size=3
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
