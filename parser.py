import sys
import os
import dataset
from bs4 import BeautifulSoup
from datafreeze import freeze
db = dataset.connect()  # SQL database URL can be stored here <------
table = db['PATENT_DATA']
properties = dict()

# print(os.getcwd())
root = os.path.realpath("PatentData") +"\Test"
for path, subdirs,files in os.walk(root):
    for name in files:
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
            # print(properties)
            # print(table)
            table.insert(properties)


freeze(table, format='csv', filename='WIPO-alpha-test.csv')
