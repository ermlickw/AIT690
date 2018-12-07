'''
AIT 690 | Patent Classificaiton Prediction Project | Due 11/28/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
This script was used to parse the data from the downloaded xml content into CSV files
for the training and testing data.
'''

import sys
import os
import dataset
from bs4 import BeautifulSoup
from datafreeze import freeze

db = dataset.connect()  # create table
table = db['PATENT_DATA']
properties = dict()

i=0
j=0
root = os.path.realpath("PatentData") +"\Test" #find directory where data is stored
for path, subdirs,files in os.walk(root): #walk it
    for name in files:
        j=j+1
        filename = os.path.join(path,name)
        if filename.endswith(".xml"):
            print(filename)
            contents = open(filename,"r").read()
            soup = BeautifulSoup(contents)

            wiponumber = soup.find_all('record')
            for c in wiponumber:
                properties['wipenumber'] = c.attrs[u'pn']

            mainclassification = soup.find_all('ipcs')
            for c in mainclassification:
                properties['mainclass'] = c.attrs[u'mc']

            subclasses = soup.find_all('ipc')
            subcls = []
            for c in subclasses:
                subcls.append(c.attrs[u'ic'])
            properties['subclasses'] = '"'+"--//--".join(subcls)+'"'
            properties['title'] = '"'+ soup.find('ti').get_text().replace('\n',"").replace('"',"").replace("'","") +'"'
            properties['abstract'] = '"'+ soup.find('ab').get_text().replace('\n',"").replace('"',"").replace("'","") +'"'
            properties['claims'] = '"'+ soup.find('cl').get_text().replace('\n',"").replace('"',"").replace("'","")+'"'
            properties['description'] = '"'+ soup.find('txt').get_text().replace('\n',"").replace('"',"").replace("'","")+'"'

            #find records in xml and store to table form and then insert into table
            # print(properties)
            # print(table)
            table.insert(properties)
            i=i+1

print(i)
print(j)


freeze(table, format='csv', filename='WIPO-alpha-test.csv') #save table as csv
