#  Code to extract patent data from a multi xml file ipgXXXXXX.xml
# import os
# import sys
from xml.dom.minidom import parse
import xml.dom.minidom
import re
# import time
import dataset
import os

db = dataset.connect()  # SQL database URL can be stored here <------
table = db['PATENT_DATA']
import os

directory = os.path.join("c:\\","Users\Billy\workspace\parserfinal")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".xml"):
            i=0
            f=open(file, 'r')
            inputtext = f.read().replace('\n', '')
            start = re.escape("<?")
            end = re.escape("/us-patent-grant>")
            matchedtext = re.findall(r'(?<={}).*?(?={})'.format(start, end), inputtext)
            n = len(matchedtext)
            for p in range(0, len(matchedtext)):
                matchedtext[p] = "<?" + matchedtext[p] + "/us-patent-grant>"
            properties = dict()
            for p in matchedtext:
                try:
                    # time.sleep(0.05)
                    cleanmatchedtext = matchedtext[i-2].replace("&", "and")
                    xmldoc = xml.dom.minidom.parseString(cleanmatchedtext)
                    grantdata = xmldoc.childNodes[1]
                    bibdata = grantdata.childNodes[0]
                    appref = bibdata.childNodes[1].attributes['appl-type'].value
                    if appref == "utility":

                            abstractstart = xmldoc.getElementsByTagName("abstract")
                            abstract_text = abstractstart[0].getElementsByTagName("p")
                            patent_title = xmldoc.getElementsByTagName("invention-title")
                            patentgrant = xmldoc.getElementsByTagName("us-patent-grant")
                            granted_date = patentgrant[0].attributes['date-produced'].value
                            doc_number = xmldoc.getElementsByTagName('doc-number')
                            claim = xmldoc.getElementsByTagName("claim-text")
                            claimtext = claim[0].getElementsByTagName("claim-text")
                            firstline = xmldoc.getElementsByTagName("claim-text")
                            apple = xmldoc.getElementsByTagName("department")
                            deptval = apple[0].childNodes[0].nodeValue
                            ClaimText = firstline[0].childNodes[0].nodeValue
                            for claimtext in claimtext:
                                finalclaimtext = claimtext.childNodes[0].nodeValue
                                ClaimText = ClaimText + finalclaimtext
                            properties['docno'] = doc_number[1].firstChild.nodeValue
                            properties['date'] = granted_date
                            properties['artunit'] = deptval
                            properties['title'] = patent_title[0].firstChild.nodeValue
                            properties['abstract'] = abstract_text[0].firstChild.nodeValue
                            properties['claims'] = ClaimText
                            print(properties)
                            print(table)
                            table.insert(properties)

                            '''doc.append(doc_number[0].firstChild.nodeValue)
                            date.append(granted_date)
                            artunit.append(deptval)
                            title.append(patent_title[0].firstChild.nodeValue)
                            abstracttext.append(abstract_text[0].firstChild.nodeValue)
                            firstclaim.append(ClaimText)'''
                    else:
                        print("not a utility patent")
                except:
                    pass
                i += 1
                print(i)
                print(n)
                print("\n")

dataset.freeze(table, format='csv', filename='2016Patent_Data.csv')
