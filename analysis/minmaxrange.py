# Author: Morgan Sandler (sandle20@msu.edu)
# finds the range of expressibility for all models
# syntax: e.g., python minmaxrange.py test2 > test2minmax.txt

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import norm
import math
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt
Z = norm.ppf


p = 0.01  # 1% or .01 of the lines works for val and test2.     0.5% or .005 works for test1
if sys.argv[1] == 'test1':
    p=.005
else:
    p=0.01
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
test_results = pd.read_csv('../ExperimentData/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # evector

test2_results = pd.read_csv('../baseline_experiment/ExperimentData/pretrained_ecapa_vox/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # ecapa vox

test3_results = pd.read_csv('../baseline_experiment/finetuned_ecapavox/ExperimentData/125epoch/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # ecapa vox+msp

test4_results = pd.read_csv('../baseline_experiment/ExperimentData/pretrained_ecapa_msp/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # ecapa msp


emotions = ['A', 'S', 'H', 'U', 'F', 'D', 'C', 'N', 'O', 'X']

emo_map = {'A':'Angry', 'S':'Sad', 'H':'Happy', 'U':'Surprise', 'F':'Fear', 'D':'Disgust', 'C':'Contempt', 'N':'Neutral', 'O':'Other', 'X':'No Agreement'}

models = ['evec', 'ecapavox', 'ecapavoxmsp', 'ecapamsp']

for model in models:
    if model == 'ecapavox':
        test_results = test2_results
    elif model =='ecapavoxmsp':
        test_results = test3_results
    elif model == 'ecapamsp':
        test_results = test4_results
    
    print(model)
    gen_mean = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']]['MatchScore'].mean() # mean match score
    gen_max = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']]['MatchScore'].max() # max match score
    gen_min = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']]['MatchScore'].min() # min match score
    print('GENUINE mean', gen_mean, 'max', gen_max,'min', gen_min)
    
    imp_mean = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']]['MatchScore'].mean() # mean match score
    imp_max = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']]['MatchScore'].max() # max match score
    imp_min = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']]['MatchScore'].min() # min match score
    print('IMPOSTOR mean', imp_mean, 'max', imp_max,'min', imp_min)
    