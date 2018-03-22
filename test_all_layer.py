#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:29:40 2018

@author: zwx
"""

from argparse import ArgumentParser
from model import WCT_test_all_layer


parser = ArgumentParser()


    
parser.add_argument('--pretrained_vgg',type=str,
                        dest='pretrained_vgg',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
parser.add_argument('--content_path',type=str,
                        dest='content_path',help='the content path',
                        metavar='Content',required = True)

parser.add_argument('--style_path',type=str,
                        dest='style_path',help='style path',
                        metavar='Style',required = True)
    
parser.add_argument('--output_path',type=str,
                        dest='output_path',help='output_path',
                        metavar='Output',required = True)
    
parser.add_argument('--alpha',type=float,
                        dest='alpha',help='the blended weight',
                        metavar='ALpha',required = True)

def main():
    opts = parser.parse_args()
    
    model = WCT_test_all_layer(
                     pretrained_vgg = opts.pretrained_vgg,
                     content_path = opts.content_path,
                     style_path = opts.style_path,
                     output_path = opts.output_path,
                     alpha = opts.alpha,
                     )
    model.test()
    
if __name__ == '__main__' :
    main()