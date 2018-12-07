# AIT 690 | P-GNN based Patent Classification
### Billy Ermlick
### Xiaojie Guo
### Nidhi Mehrotra

This project propose a novel patent graphical neural network (P-GNN) approach for the task of automated patent classification. Two experiments are performed on the benchmarked WIPO-alpha dataset. 


Data processing:

The file factory.py is used to process the raw dataset and embed the patent documents into feature vectors. The output of this file is train-D.npy, test-D.npy, train_label-D.npy and test_label-D.npy, as well as the saved 
classifiers and confusion matrix plots. 


Experiment 1

It utilizes a stacked TF-IDF feature matrix on the extracted title, abstract, claims, and description. Grid search tuning is performed to achieve results comparable to others in the field. 
To uss the models in experiment 1:


Experiment 2 

It implements the P-GNN model on a subset of the dataset, outperforming previous methods conducted by others in the patent classification field. Improvements in the current P-GNN model to handle larger datasets shows promising results for future classification tasks. 

To use the P-GNN model in experiment 1,use the command:
       
       python GNN.py 
       


