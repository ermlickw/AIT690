# AIT 690 | P-GNN based Patent Classification
### Billy Ermlick
### Xiaojie Guo
### Nidhi Mehrotra

## 
This project proposes a novel patent graphical neural network (P-GNN) approach for the task of automated patent classification. Two experiments are performed on the benchmarked WIPO-alpha dataset. Experiment 1 utilizes the entire data set to make predictions at the Subclass level. Experiment 2 utilizes Section D of the dataset to make predictions at the maingroup level.

**It is recommended that you run Experiment 2 first.**

## Data Extraction:

The file parser.py dictates how the training and text set data were extracted from the given XML files and into a CSV format. Downloads of the CSV files are given below as they are too large to store in GitHUB.

## Data Processing:

The file factory.py is used to process the raw dataset and embed the patent documents into feature vectors. The output of this file is train-D.npy, test-D.npy, train_label-D.npy and test_label-D.npy, as well as the saved classifiers and confusion matrix plots. 


## Experiment 1

It utilizes a stacked TF-IDF feature matrix on the extracted title, abstract, claims, and description. Grid search tuning is performed to achieve results comparable to others in the field. 

To run the models in experiment 1,<br>
1)Download WIPO-alpha-train.csv and WIPO-alpha-test.csv from link below or use parser.py on web provided dataset<br>
2)Change lines 322 and 323 to "False" <br>
3)Comment lines 332 and 333 <br>
4)Change line 337 and 338 to: <br>
              
    combineddf['mainclass'] = combineddf['mainclass'].apply(lambda x: (x[:4]).strip()) #change class to Subclass for ExP1
    labels = list(set(testdf['mainclass'].apply(lambda x: (x[:4]).strip())))  #change class to Subclass for Exp1
       

Then use the command:
       
     python factory.py 
       
       
## Experiment 2 

It implements the P-GNN model on a subset of the dataset, outperforming previous methods conducted by others in the patent classification field. Improvements in the current P-GNN model to handle larger datasets shows promising results for future classification tasks. 

#### If you have not run experiment 1:<br>

To use the P-GNN model in experiment 2, use the command:
       
       python GNN.py 
       
To run the baseline models in experiment 2, use the command:
       
       python factory.py 

#### If you have run experiment 1: <br>
1)Uncomment lines 332 and 333 <br>
2)Change line 337 and 338 to: <br>
              
    combineddf['mainclass'] = combineddf['mainclass'].apply(lambda x: (x[:6]).strip()) #change class to Main group for ExP2
    labels = list(set(testdf['mainclass'].apply(lambda x: (x[:6]).strip())))  #change class to Main group for Exp2
    
3) Run as stated above
## Dataset
The parsed data is available for download via: https://drive.google.com/drive/folders/1gOBlngdaolH7OUROw3pgA02R1vEtHzM5?usp=sharing

The offical dataset is available via: https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset/wipo-alpha-readme.html
