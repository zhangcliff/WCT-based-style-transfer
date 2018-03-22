#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:45:07 2018

@author: zwx
"""
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from skimage.io import imsave,imshow,imread
from keras.preprocessing import image
from vgg19 import Vgg19
from Decoder import  Decoder
from wct import wct_tf,Adain

class WTC_Model:
    def __init__(self, target_layer=None, pretrained_path=None, max_iterator=None, checkpoint_path=None, tfrecord_path=None, batch_size=None):
        self.pretrained_path = pretrained_path
        self.target_layer = target_layer
        self.encoder = Vgg19(self.pretrained_path)
        self.max_iterator = max_iterator
        self.checkpoint_path = checkpoint_path
        self.tfrecord_path = tfrecord_path
        self.batch_size = batch_size
        
    def encoder_decoder(self,inputs):
        encoded = self.encoder.encoder(inputs,self.target_layer)
        model=Decoder()
        decoded,_ = model.decoder(encoded,self.target_layer)
        decoded_encoded= self.encoder.encoder(decoded,self.target_layer)
        
        return encoded,decoded,decoded_encoded
    
    
    def train(self):
        inputs = tf.placeholder('float',[None,224,224,3])
        outputs = tf.placeholder('float',[None,224,224,3])
        
        encoded,decoded,decoded_encoded = self.encoder_decoder(inputs)
        
        pixel_loss = tf.losses.mean_squared_error(decoded,outputs)
        feature_loss = tf.losses.mean_squared_error(decoded_encoded,encoded)
        loss = pixel_loss+ feature_loss
        opt= tf.train.AdamOptimizer(0.0001).minimize(loss)
        
        tfrecords_filename =  self.tfrecord_path
        filename_queue = tf.train.string_input_producer([tfrecords_filename],num_epochs=100)

        reader = tf.TFRecordReader()  
        _, serialized_example = reader.read(filename_queue)

        feature2 = {  
                    'image_raw': tf.FixedLenFeature([], tf.string)} 
        features = tf.parse_single_example(serialized_example, features=feature2)  
        image = tf.decode_raw(features['image_raw'], tf.uint8) 
        image = tf.reshape(image,[224,224,3])   
        images = tf.train.shuffle_batch([image], batch_size=self.batch_size, capacity=30, min_after_dequeue=10)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config)as sess  :
             tf.global_variables_initializer().run()
             tf.local_variables_initializer().run()
        

        
             coord = tf.train.Coordinator()  
             threads = tf.train.start_queue_runners(coord=coord)  
  
             saver = tf.train.Saver()
             

             for i in range (self.max_iterator):
                 batch_x=sess.run(images)
                 feed_dict = {inputs:batch_x, outputs : batch_x}
            
                 _,p_loss,f_loss,reconstruct_imgs=sess.run([opt,pixel_loss,feature_loss,decoded],feed_dict=feed_dict)
            
                 print('step %d |  pixel_loss is %f   | feature_loss is %f  |'%(i,p_loss,f_loss))
            
                 if i % 5 ==0:
                    result_img = np.clip(reconstruct_imgs[0],0,255).astype(np.uint8)
                    imsave('result.jpg',result_img)
                
             saver.save(sess,self.checkpoint_path)
             coord.request_stop()  
             coord.join(threads)

class WCT_test_single_layer:
    def __init__(self,target_layer,content_path,style_path,alpha,pretrained_vgg,output_path,decoder_weights) :
        self.target_layer = target_layer
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.alpha = alpha
        self.encoder = Vgg19(pretrained_vgg)
        self.decoder = Decoder()  
        self.decoder_weights = decoder_weights
    def test(self):
        content = tf.placeholder('float',[1,304,304,3])
        style = tf.placeholder('float',[1,304,304,3])
        
        content_encode = self.encoder.encoder(content,self.target_layer)
        style_encode = self.encoder.encoder(style,self.target_layer)
        
        blended = wct_tf(content_encode,style_encode,self.alpha)
        #blended = Adain(content_encode,style_encode)
        
        stylized = self.decoder.decoder(blended,self.target_layer)
        saver = tf.train.Saver()
        
        with tf.Session()as sess:
             tf.global_variables_initializer().run()
             tf.local_variables_initializer().run()
             saver.restore(sess,self.decoder_weights)
             img_c = image.load_img(self.content_path,target_size=(304,304,3))
             img_c = image.img_to_array(img_c)
             img_c = np.expand_dims(img_c,axis=0)
             
             img_s = image.load_img(self.style_path,target_size = (304,304,3))
             img_s = image.img_to_array(img_s)
             img_s = np.expand_dims(img_s,axis=0)    
             
             feed_dict = {content : img_c , style : img_s}
             
             result,e = sess.run([stylized,content_encode],feed_dict= feed_dict)
             result = result[0]
             result = np.clip(result,0,255)/255.
             #print(e)
             imsave(self.output_path,result)
             
class WCT_test_all_layer:
    def __init__(self,content_path,style_path,alpha,pretrained_vgg,output_path) :
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.alpha = alpha
        self.encoder = Vgg19(pretrained_vgg)
        self.decoder = Decoder()  
        self.decoder_weights = ['models/decoder_1.ckpt','models/decoder_2.ckpt','models/decoder_3.ckpt','models/decoder_4.ckpt']
    def test(self):
        content = tf.placeholder('float',[1,304,304,3])
        style = tf.placeholder('float',[1,304,304,3])
        
        content_encode_4 = self.encoder.encoder(content,'relu4')
        style_encode_4 = self.encoder.encoder(style,'relu4')
        blended_4 = wct_tf(content_encode_4,style_encode_4,self.alpha)
        stylized_4 ,var_list4= self.decoder.decoder(blended_4,'relu4')
        
        content_encode_3 = self.encoder.encoder(stylized_4,'relu3')
        style_encode_3 = self.encoder.encoder(style,'relu3')
        blended_3 = wct_tf(content_encode_3,style_encode_3,self.alpha)
        stylized_3 ,var_list3= self.decoder.decoder(blended_3,'relu3')
        
        content_encode_2 = self.encoder.encoder(stylized_3,'relu2')
        style_encode_2 = self.encoder.encoder(style,'relu2')
        blended_2 = wct_tf(content_encode_2,style_encode_2,self.alpha)
        stylized_2 ,var_list2= self.decoder.decoder(blended_2,'relu2')        
        
        content_encode_1 = self.encoder.encoder(stylized_2,'relu1')
        style_encode_1 = self.encoder.encoder(style,'relu1')
        blended_1 = wct_tf(content_encode_1,style_encode_1,self.alpha)
        stylized_1,var_list1 = self.decoder.decoder(blended_1,'relu1')
        saver1 = tf.train.Saver(var_list1)
        saver2 = tf.train.Saver(var_list2)
        saver3 = tf.train.Saver(var_list3)
        saver4 = tf.train.Saver(var_list4)
        
        with tf.Session()as sess:
             tf.global_variables_initializer().run()
             tf.local_variables_initializer().run()
             saver1.restore(sess,self.decoder_weights[0])
             saver2.restore(sess,self.decoder_weights[1])
             saver3.restore(sess,self.decoder_weights[2])
             saver4.restore(sess,self.decoder_weights[3])
             img_c = image.load_img(self.content_path,target_size=(304,304,3))
             img_c = image.img_to_array(img_c)
             img_c = np.expand_dims(img_c,axis=0)
             
             img_s = image.load_img(self.style_path,target_size = (304,304,3))
             img_s = image.img_to_array(img_s)
             img_s = np.expand_dims(img_s,axis=0)    
             
             feed_dict = {content : img_c , style : img_s}
             
             result = sess.run(stylized_1,feed_dict= feed_dict)
             result = result[0]
             result = np.clip(result,0,255)/255.
             
             imsave(self.output_path,result) 
        



       
             

    
        
