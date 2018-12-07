# -*- coding: utf-8 -*-
"""
AIT 690 | Patent Classificaiton Prediction Project | Due 11/28/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
This is the main function for training and tesing the P-GNN model.
This code is modified based on the model "Interaction Network": https://github.com/jaesik817/Interaction-networks_tensorflow
**********************
To directly use the classifier, use the command:
    python GNN.py
"""

import argparse
import os
from model import graph2graph
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', type=int, default=300, help='number of training epochs')
parser.add_argument('--Ds', type=int, default=100,help='The feature Dimention')
parser.add_argument('--Ds_label', type=int, default=57,help='The State Dimention')
parser.add_argument('--Dr', type=int, default=1,help='The Relationship Dimension')
parser.add_argument('--De_o', type=int, default=1,help='The Effect Dimension on node')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint',help='models are saved here')
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = graph2graph(sess, Ds=args.Ds, Ds_label=args.Ds_label,Dr=args.Dr,De_o=args.De_o,
                        checkpoint_dir=args.checkpoint_dir,epoch=args.epoch)
        model.train(args)


if __name__ == '__main__':
      tf.app.run()
