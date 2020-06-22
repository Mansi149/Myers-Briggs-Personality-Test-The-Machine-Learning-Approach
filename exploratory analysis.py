# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 22:19:20 2020

@author: Mansi
"""

import pandas as pd
import numpy as np

# plotting
import seaborn as sea
import matplotlib.pyplot as mtp

# read data
data = pd.read_csv('mbti_1.csv') 
[p.split('|||') for p in data.head(2).posts.values]

# Distribution of the MBTI personality types
types = data['type'].value_counts()

mtp.figure(figsize=(12,4))
sea.barplot(types.index, types.values, alpha=0.8)
mtp.ylabel('Number of Occurrences', fontsize=12)
mtp.xlabel('Types', fontsize=12)
mtp.show()

# Distribution of the type Indicators
def get_types(row):
    t=row['type']

    I = 0; N = 0
    T = 0; J = 0
    
    if t[0] == 'I': I = 1
    elif t[0] == 'E': I = 0
    else: print('I-E incorrect')
        
    if t[1] == 'N': N = 1
    elif t[1] == 'S': N = 0
    else: print('N-S incorrect')
        
    if t[2] == 'T': T = 1
    elif t[2] == 'F': T = 0
    else: print('T-F incorrect')
        
    if t[3] == 'J': J = 1
    elif t[3] == 'P': J = 0
    else: print('J-P incorrect')
    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J }) 

data = data.join(data.apply (lambda row: get_types (row),axis=1))


print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
print ("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
print ("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
print ("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])

N = 4
but = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

ind = np.arange(N)  
width = 0.4      

p1 = mtp.bar(ind, but, width)
p2 = mtp.bar(ind, top, width, bottom=but)

mtp.ylabel('Count')
mtp.title('Distribution accoss types indicators')
mtp.xticks(ind, ('I/E',  'N/S', 'T/F', 'J/P',))

mtp.show()