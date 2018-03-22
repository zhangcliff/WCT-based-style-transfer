import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Decoder:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        

    
    def decoder(self,encode,target_layer) :
        layer_num = dict(zip(['relu1','relu2','relu3','relu4','relu5'],range(1,6)))[target_layer]
        var_list=[]
        decode_arg={
                '5':[('upsample',14,28),
                     ('dconv5_1',512,512),
                     ('dconv5_2',512,512),
                     ('dconv5_3',512,512),
                     ('dconv5_4',512,512)],
                
                '4':[('upsample',28,56),
                     ('dconv4_1',512,256),
                     ('dconv4_2',256,256),
                     ('dconv4_3',256,256),
                     ('dconv4_4',256,256)],

                '3':[('upsample',56,112),
                     ('dconv3_1',256,128),
                     ('dconv3_2',128,128),
                     ('dconv3_3',128,128),
                     ('dconv3_4',128,128)],

                '2':[('upsample',112,224),
                     ('dconv2_1',128,64),
                     ('dconv2_2',64,64)],
                '1':[('dconv1_1',64,64),
                     ('output',64,3)]} 
        decode = encode
        for d in reversed(range(1,layer_num+1)):
            for layer in decode_arg[str(d)]:
                if 'up' in layer[0]:
                    decode = self.upsample(decode,layer[1])
                if 'dconv' in layer[0] :
                    decode ,var_list= self.conv_layer(decode,layer[1],layer[2],layer[0]+'_'+target_layer,var_list)
                if 'out' in layer[0] :
                    decode, var_list = self.output_layer(decode,layer[1],layer[2],layer[0]+'_'+target_layer,var_list)
                    
        return decode , var_list
        #self.data_dict = None

    def upsample(self,bottom,height):
        height=height
        width=height
        
        new_height=height*2
        new_width = width*2
        return tf.image.resize_images(bottom, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def output_layer(self, bottom, in_channels, out_channels, name,var_list):
        with tf.variable_scope(name):
            filt_size = 9
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            
            var_list.append(filt)
            var_list.append(conv_biases)
            return bias,var_list
    
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name,var_list,trainable=True):
        filt_size = 3
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)
            
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)  
            
            var_list.append(filt)
            var_list.append(conv_biases)
            return relu,var_list

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            print ('resore %s weight'%(name))
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
