# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:19:59 2018

@author: gxjco
"""

import argparse
import os
from model import graph2graph
import tensorflow as tf




parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
parser.add_argument('--Ds', type=int, default=3494,help='The State Dimention')
parser.add_argument('--Ds_label', type=int, default=54,help='The State Dimention')
parser.add_argument('--No', type=int, default=374, help='The Number of nuerons')
parser.add_argument('--Dr', type=int, default=1,help='The Relationship Dimension')
parser.add_argument('--De_o', type=int, default=10,help='The Effect Dimension on node')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint',help='models are saved here')
parser.add_argument('--Type', dest='Type', default='test',help='train or test')
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    tf.reset_default_graph() 
    with tf.Session() as sess:
        model = graph2graph(sess, Ds=args.Ds,No=args.No,Nr=int(args.No*(args.No-1)), Ds_label=args.Ds_label,Dr=args.Dr,De_o=args.De_o,
                        checkpoint_dir=args.checkpoint_dir,epoch=args.epoch)

        model.train(args)

if __name__ == '__main__':
      tf.app.run()