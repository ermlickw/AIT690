# -*- coding: utf-8 -*-
"""
AIT 690 | Patent Classificaiton Prediction Project | Due 11/28/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
This is the compoments functions of the P-GNN.
"""
import numpy as np
from scipy.spatial.distance import cosine
import random
import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.spatial.distance import cosine
import random


def acc(node,label): #calculate the accuracy of results
        score=0
        for i in range(node.shape[1]):
         if np.argmax(node[:,i])==np.argmax(label[:,i]):
             score+=1
        return score/node.shape[1]

def read_data():
    "read in the dataset and represent the dataset with a graph output"
    train=np.load('train-D.npy')
    test=np.load('test-D.npy')
    train_label=np.load('train_label-D.npy')
    test_label=np.load('test_label-D.npy')
    features=np.concatenate((train,test),axis=0)
    train_len=train.shape[0]
    test_len=test.shape[0]

    labels=np.concatenate((train_label,test_label),axis=0)
    labels=list(labels)
    label_list=list(set(labels))  #convert label to hot vector for each sample

    num_class=len(label_list)
    label=np.zeros((features.shape[0],num_class))
    for i in range(features.shape[0]):
        j=label_list.index(labels[i])
        label[i][j]=1

    #s=np.load('order40.npy')
    f=features #[s]
    l=label #[s]

    f_train=f[:train_len]

    adj_train = np.zeros((train_len,train_len))   #compute adjacent matrix
    for i in range(f_train.shape[0]):
        for j in range(i+1,f_train.shape[0]):
            if cosine(f_train[i],f_train[j])>=1.1:
                adj_train[i][j]=1


    Rr_data_train=np.zeros((1352,98423),dtype=float);  #transform relation format
    Rs_data_train=np.zeros((1352,98423),dtype=float);
    Ra_data_train=np.zeros((1,98423),dtype=float);

    cnt=0
    for i in range(1352):
       for j in range(i+1,1352):
           if adj_train[i,j]==1:
             Rr_data_train[i,cnt]=1.0;
             Rs_data_train[j,cnt]=1.0;
             Ra_data_train[0,cnt]=adj_train[i,j]
             cnt+=1;

    return np.transpose(f),np.transpose(l),Ra_data_train,Rr_data_train,Rs_data_train,train_len,test_len

class graph2graph(object):
    "build a class of model"
    def __init__(self, sess,Ds,Ds_label, Dr,De_o,checkpoint_dir,epoch):
        self.sess = sess
        self.Ds = Ds
        self.Ds_label = Ds_label
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

    def build_model(self): #building the model with compuataions steps
       self.No=1352
       self.Nr=98423
       self.O = tf.placeholder(tf.float32, [self.Ds,1710], name="O")
       self.O_target=tf.placeholder(tf.float32, [self.Ds_label,1710], name="O_target")
       # Relation Matrics R=<Rr,Rs,Ra>
       self.Rr = tf.placeholder(tf.float32, [self.No,self.Nr], name="Rr")
       self.Rs = tf.placeholder(tf.float32, [self.No,self.Nr], name="Rs")
       self.Ra = tf.placeholder(tf.float32, [self.Dr,self.Nr], name="Ra")
       self.index=tf.placeholder(tf.int32, name="batch_index")
       # External Effects
       # marshalling function
       self.B=self.m(self.O[:,:self.No],self.Rr,self.Rs,self.Ra)
       # updating the node state
       self.E_O=self.phi_E_O(self.B)
       self.C_O=self.a_O(self.E_O,self.Rr)
       self.O_p,self.O_logits=self.phi_U_O(self.C_O,1710)
       self.mini_batch=100
        # loss
       self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.O_logits[:,self.index*self.mini_batch:(self.index*self.mini_batch+self.mini_batch)],labels=self.O_target[:,self.index*self.mini_batch:(self.index*self.mini_batch+self.mini_batch)],dim=0))
       #self.loss_te=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.O_logits_te,labels=self.O_target_te,dim=0))

       #self.loss_E_O_tr = 0.01*tf.nn.l2_loss(self.E_O_tr) #regulization

       params_list=tf.global_variables()
       for i in range(len(params_list)):
            self.variable_summaries(params_list[i],i)
       self.loss_para=0
       for i in params_list:
          self.loss_para+=0.001*tf.nn.l2_loss(i);

       tf.summary.scalar('node_mse',self.loss)

       t_vars = tf.trainable_variables()
       self.vars = [var for var in t_vars]

       self.saver = tf.train.Saver()


    def m(self,O,Rr,Rs,Ra):
     return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],0);

    def phi_E_O(self,B):
     with tf.variable_scope("phi_E_O") as scope:
      h_size=5;
      B_trans=tf.transpose(B,[1,0]);
      B_trans=tf.reshape(B_trans,[-1,(2*self.Ds+self.Dr)]);

      w1 = tf.Variable(tf.truncated_normal([(2*self.Ds+self.Dr), h_size], stddev=0.1), name="r_w1o", dtype=tf.float32);
      b1 = tf.Variable(tf.zeros([h_size]), name="r_b1o", dtype=tf.float32);
      h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);

      w5 = tf.Variable(tf.truncated_normal([h_size, self.De_o], stddev=0.1), name="r_w5o", dtype=tf.float32);
      b5 = tf.Variable(tf.zeros([self.De_o]), name="r_b5o", dtype=tf.float32);
      h5 = tf.matmul(h1, w5) + b5;

      h5_trans=tf.reshape(h5,[-1,self.De_o]);
      h5_trans=tf.transpose(h5_trans,[1,0]);
      return(h5_trans);

    def a_O(self,E,Rr):
       E_bar=tf.matmul(E,tf.transpose(Rr,[1,0]));
       return self.O  #(tf.concat([self.O,0.01*E_bar],0));

    def phi_U_O(self,C,no):
     with tf.variable_scope("phi_U_O") as scope:
       h_size=50;
       C_trans=tf.transpose(C,[1,0]);
       C_trans=tf.reshape(C_trans,[-1,(self.Ds)]);
       w1 = tf.Variable(tf.truncated_normal([(self.Ds), h_size], stddev=0.1), name="o_w1o", dtype=tf.float32);
       b1 = tf.Variable(tf.zeros([h_size]), name="o_b1o", dtype=tf.float32);
       h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
       #h1=tf.nn.dropout(h1, 0.5)

       w3 = tf.Variable(tf.truncated_normal([h_size, self.Ds_label], stddev=0.1), name="o_w2o", dtype=tf.float32);
       b3 = tf.Variable(tf.zeros([self.Ds_label]), name="o_b2o", dtype=tf.float32);
       h3 = tf.matmul(h1, w3) + b3
       #h3=tf.nn.dropout(h3, 0.5)

       h3_trans_logits=tf.transpose(h3,[1,0]);
       h3_trans=tf.nn.softmax(h3_trans_logits,dim=0);
       return h3_trans,h3_trans_logits



    def p_r_f(self,node,label):
        "this fucntion is used to compute the precision, recall, and f1 score"
        new_node=np.zeros(node.shape[1])
        new_label=np.zeros(node.shape[1])
        for i in range(node.shape[1]):
            new_node[i]=np.argmax(node[:,i])
            new_label[i]=np.argmax(label[:,i])
        recall=metrics.recall_score(new_label, new_node, average='macro')
        precision=metrics.precision_score(new_label, new_node, average='macro')
        f1=metrics.f1_score(new_label, new_node, average='macro')
        confusion_matrix(new_label, new_node)
        return precision,recall,f1

    def train(self, args):
        "This function used for training the model"
        optimizer = tf.train.AdamOptimizer(0.001);
        trainer=optimizer.minimize(self.loss+self.loss_para)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        #read data
        #features,features_test,label_train,label_test,Ra_data_train,Ra_data_test,Rr_data_train,Rr_data_test,Rs_data_train,Rs_data_test=read_data()
        features,label,Ra_data,Rr_data,Rs_data,train_len,test_len=read_data()
        max_epoches=self.epoch
        counter=1
        for i in range(max_epoches):
          tr_loss_node=0
          for j in range(int(train_len/self.mini_batch)):
             O=features
             O_target=label
             Ra=Ra_data
             tr_loss,node,label,_=self.sess.run([self.loss,self.O_p,self.O_target,trainer],feed_dict={self.O:O,self.O_target:O_target,self.Ra:Ra,self.Rr:Rr_data,self.Rs:Rs_data,self.index:j});
             tr_loss_node+=tr_loss
          print("Epoch "+str(i+1)+" train loss: "+str(tr_loss/(train_len/self.mini_batch))+" train acc: "+str(acc(node[:,:train_len],label[:,:train_len]))+" test acc: "+str(acc(node[:,train_len:],label[:,train_len:])));
          counter+=1
          #self.save(args.checkpoint_dir, counter)
        p,r,f1=self.p_r_f(node[:,train_len:],label[:,train_len:])
        print(" test precision: "+str(p))
        print(" test recall: "+str(r))
        print(" test f1: "+str(f1))
        np.save('predict.npy',node)
        np.save('label.npy',label)
