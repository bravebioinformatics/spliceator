import os
import shutil

#read the list of lncRNA
with open('lncRNA_list.txt') as f:
    lines = f.readlines()

#search in dat/ and copy the file if it matches
for l in lines:
    #print(l)
    #print('./dat/'+l)
    l = l.replace('\n','')
    shutil.copy('dat/'+l, './raw_seq')
