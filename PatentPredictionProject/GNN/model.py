# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:27:44 2018

@author: gxjco
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from utils import read_data

class graph2graph(object):
    def __init__(self, sess,Ds,No,Nr,Ds_label, Dr,De_o,checkpoint_dir,epoch):
        self.sess = sess
        self.Ds = Ds
        self.Ds_label = Ds_label
        self.No = No
        self.Nr =Nr
        self.Dr = Dr
        self.De_o=De_o
        self.epoch=epoch
        self.checkpoint_dir = checkpoint_dir
        self.build_model()
        
    def variable_summaries(self,var,idx):
         """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
         with tf.name_scope('summaries_'+str(idx)):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            
    def build_model(self):
       self.O = tf.placeholder(tf.float32, [self.Ds,self.No], name="O")
       self.O_target=tf.placeholder(tf.float32, [self.Ds_label,self.No], name="O_target")
       # Relation Matrics R=<Rr,Rs,Ra>
       self.Rr = tf.placeholder(tf.float32, [self.No,self.Nr], name="Rr")
       self.Rs = tf.placeholder(tf.float32, [self.No,self.Nr], name="Rs")
       self.Ra = tf.placeholder(tf.float32, [self.Dr,self.Nr], name="Ra")
       # External Effects 
       # marshalling function, m(G)=B, G=<O,R>  
       self.B=self.m(self.O,self.Rr,self.Rs,self.Ra)
       # updating the node state
       self.E_O=self.phi_E_O(self.B)  
       self.C_O=self.a_O(self.E_O,self.Rr)
       self.O_,self.O_logits=self.phi_U_O(self.C_O)

        # loss  
       self.loss_train=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.O_logits[:250],labels=tf.transpose(self.O_target)[:250]))
       self.loss_test=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.O_logits[250:],labels=tf.transpose(self.O_target)[250:]))
       #self.loss_edge_mse=tf.reduce_mean(tf.reduce_mean(tf.square(self.Ra_-self.Ra_target),[1,2]))
       #self.loss_map=tf.reduce_mean(tf.reduce_mean(tf.square(self.O_-self.O_map),[1,2]))
  
       self.loss_E_O = 0.001*tf.nn.l2_loss(self.E_O) #regulization
       
       params_list=tf.global_variables()
       for i in range(len(params_list)):
            self.variable_summaries(params_list[i],i)
       self.loss_para=0
       for i in params_list:
          self.loss_para+=0.001*tf.nn.l2_loss(i); 
          
       tf.summary.scalar('node_mse',self.loss_train)
       #tf.summary.scalar('edge_mse',self.loss_edge_mse)
       #tf.summary.scalar('map_mse',self.loss_map) 
       
       t_vars = tf.trainable_variables()
       self.vars = [var for var in t_vars]
       
       self.saver = tf.train.Saver()


    def m(self,O,Rr,Rs,Ra):
      #return tf.concat([(tf.matmul(O,Rr)-tf.matmul(O,Rs)),Ra],1);
     return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],0);

    def phi_E_O(self,B):
     with tf.variable_scope("phi_E_O") as scope:  
      h_size=100;
      B_trans=tf.transpose(B,[1,0]);
      B_trans=tf.reshape(B_trans,[-1,(2*self.Ds+self.Dr)]);
  
      w1 = tf.Variable(tf.truncated_normal([(2*self.Ds+self.Dr), h_size], stddev=0.1), name="r_w1o", dtype=tf.float32);
      b1 = tf.Variable(tf.zeros([h_size]), name="r_b1o", dtype=tf.float32);
      h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
      w5 = tf.Variable(tf.truncated_normal([h_size, self.De_o], stddev=0.1), name="r_w5o", dtype=tf.float32);
      b5 = tf.Variable(tf.zeros([self.De_o]), name="r_b5o", dtype=tf.float32);
      h5 = tf.matmul(h1, w5) + b5;
  
      h5_trans=tf.reshape(h5,[self.Nr,self.De_o]);
      h5_trans=tf.transpose(h5_trans,[1,0]);
      return(h5_trans);
  
    def a_O(self,E,Rr):
       E_bar=tf.matmul(E,tf.transpose(Rr,[1,0]));
       return (tf.concat([self.O,E_bar],0)); #self.O

    def phi_U_O(self,C):
     with tf.variable_scope("phi_U_O") as scope:          
       h_size=100;
       C_trans=tf.transpose(C,[1,0]);
       C_trans=tf.reshape(C_trans,[-1,(self.Ds+self.De_o)]);
       w1 = tf.Variable(tf.truncated_normal([(self.Ds+self.De_o), h_size], stddev=0.1), name="o_w1o", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="o_b1o", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
       w2 = tf.Variable(tf.truncated_normal([h_size, self.Ds_label], stddev=0.1), name="o_w2o", dtype=tf.float32);
       b2 = tf.Variable(tf.zeros([self.Ds_label]), name="o_b2o", dtype=tf.float32);
       h2 = tf.matmul(h1, w2) + b2
       h2_trans_logits=tf.reshape(h2,[self.No,self.Ds_label]);
       h2_trans=tf.nn.softmax(h2_trans_logits);
       return h2_trans,h2_trans_logits
       
    def acc(self,node,label):
        score=0
        for i in range(node.shape[1]):
         if np.argmax(node[:,i])==np.argmax(label[:,i]):
             score+=1
        return score/node.shape[1]
    
    
    def train(self, args):
        optimizer = tf.train.AdamOptimizer(0.02);
        trainer=optimizer.minimize(self.loss_train+self.loss_para)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        #read data
        features,label,Ra_data,Rr_data,Rs_data=read_data(self)
     
        max_epoches=self.epoch
        counter=1
        for i in range(max_epoches):
          batch_O=features;
          batch_O_target=label;
          batch_Ra=Ra_data
          tr_loss,te_loss,node,label,_=self.sess.run([self.loss_train,self.loss_test,self.O_,self.O_target,trainer],feed_dict={self.O:batch_O,self.O_target:batch_O_target,self.Ra:batch_Ra,self.Rr:Rr_data,self.Rs:Rs_data});
          node=np.transpose(node)
          #tr_loss_edge+=tr_loss_part_edge
            #tr_loss_map+=tr_loss_part_map
          print("Epoch "+str(i+1)+" train loss: "+str(tr_loss)+str(i+1)+" train acc: "+str(self.acc(node[:,:250],label[:,:250]))+" test loss: "+str(te_loss)+" test acc: "+str(self.acc(node[:,250:],label[:,250:])));
          #print("Epoch ");
          #print("Epoch ");
          #print("Epoch ");
          counter+=1
          self.save(args.checkpoint_dir, counter)
    
    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        

