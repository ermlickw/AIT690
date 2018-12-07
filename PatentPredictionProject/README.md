# AIT 690 | P-GNN based Patent Classification
### Billy Ermlick
### Xiaojie Guo
### Nidhi Mehrotra

## 
This project proposes a novel patent graphical neural network (P-GNN) approach for the task of automated patent classification. Two experiments are performed on the benchmarked WIPO-alpha dataset. Experiment 1 utilizes the entire data set to make predictions at the Subclass level. Experiment 2 utilizes Section D of the dataset to make predictions at the maingroup level.


## Data Processing:

The file factory.py is used to process the raw dataset and embed the patent documents into feature vectors. The output of this file is train-D.npy, test-D.npy, train_label-D.npy and test_label-D.npy, as well as the saved classifiers and confusion matrix plots. 


## Experiment 1

It utilizes a stacked TF-IDF feature matrix on the extracted title, abstract, claims, and description. Grid search tuning is performed to achieve results comparable to others in the field. 

To run the models in experiment 1,<br>
1)Comment lines 315 and 316 <br>
2)Change line 320 to:
              
    combineddf['mainclass'] = combineddf['mainclass'].apply(lambda x: (x[:4]).strip())
       

Then use the command:
       
     python factory.py 
       
       
## Experiment 2 

It implements the P-GNN model on a subset of the dataset, outperforming previous methods conducted by others in the patent classification field. Improvements in the current P-GNN model to handle larger datasets shows promising results for future classification tasks. 

To use the P-GNN model in experiment 2, use the command:
       
       python GNN.py 
       
To run the baseline models in experiment 2, use the command:
       
       python factory.py 

## Dataset
The parsed data is available for download via: https://drive.google.com/drive/folders/1gOBlngdaolH7OUROw3pgA02R1vEtHzM5?usp=sharing

The offical dataset is available via: https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset/wipo-alpha-readme.html
