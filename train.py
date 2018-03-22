#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:08:02 2018

@author: zwx
"""

from argparse import ArgumentParser
from model import WTC_Model


parser = ArgumentParser()

parser.add_argument('--target_layer', type=str,
                        dest='target_layer', help='target_layer(such as relu5)',
                        metavar='target_layer', required=True)
    
parser.add_argument('--pretrained_path',type=str,
                        dest='pretrained_path',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
parser.add_argument('--max_iterator',type=int,
                        dest='max_iterator',help='the max iterator',
                        metavar='MAX',required = True)

parser.add_argument('--checkpoint_path',type=str,
                        dest='checkpoint_path',help='checkpoint path',
                        metavar='CheckPoint',required = True)
    
parser.add_argument('--tfrecord_path',type=str,
                        dest='tfrecord_path',help='tfrecord path',
                        metavar='Tfrecord',required = True)
    
parser.add_argument('--batch_size',type=int,
                        dest='batch_size',help='batch_size',
                        metavar='Batch_size',required = True)
    


def main():

    opts = parser.parse_args()
    
    model = WTC_Model(target_layer = opts.target_layer,
                      pretrained_path = opts.pretrained_path,
                      max_iterator = opts.max_iterator,
                      checkpoint_path = opts.checkpoint_path,
                      tfrecord_path = opts.tfrecord_path,
                      batch_size = opts.batch_size)
    
    model.train()
    
if __name__=='__main__' :
    main()