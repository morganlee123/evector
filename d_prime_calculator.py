# Implementation of D-prime from the following post
# https://stats.stackexchange.com/questions/492673/understanding-and-implementing-the-dprime-measure-in-python

# USAGE python d_prime_calculator.py [test1/test2/val] > d_prime_res.txt 

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import norm
import math
import sys
import random
import pandas as pd
Z = norm.ppf


p = 0.01  # 1% or .01 of the lines works for val and test2.     0.5% or .005 works for test1
if sys.argv[1] == 'test1':
    p=.005
else:
    p=0.01
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
test_results = pd.read_csv('./ExperimentData/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # evector

# TODO: Maybe write handling code for test1 case since test1 currently doesnt evaluate on VoxCeleb...
test2_results = pd.read_csv('./baseline_experiment/ExperimentData/pretrained_ecapa_vox/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # ecapa vox

test3_results = pd.read_csv('./baseline_experiment/finetuned_ecapavox/ExperimentData/125epoch/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # ecapa vox+msp

test4_results = pd.read_csv('./baseline_experiment/ExperimentData/pretrained_ecapa_msp/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) # ecapa msp


print('results loaded in')

# Sort results into genuine and impostor lists
genuine1, impostor1 = [], []
for i, row in test_results.iterrows():
    if row['SpkrIDA'] == row['SpkrIDB']:
        # Genuine
        genuine1.append(row)
    else:
        # Impostor
        impostor1.append(row)

print(len(genuine1),'genuine scores,', len(impostor1), 'impostor scores loaded')


genuine2, impostor2 = [], []
for i, row in test2_results.iterrows():
    if row['SpkrIDA'] == row['SpkrIDB']:
        # Genuine
        genuine2.append(row)
    else:
        # Impostor
        impostor2.append(row)

print(len(genuine2),'genuine scores,', len(impostor2), 'impostor scores loaded')

genuine3, impostor3 = [], []
for i, row in test3_results.iterrows():
    if row['SpkrIDA'] == row['SpkrIDB']:
        # Genuine
        genuine3.append(row)
    else:
        # Impostor
        impostor3.append(row)

print(len(genuine3),'genuine scores,', len(impostor3), 'impostor scores loaded')

genuine4, impostor4 = [], []
for i, row in test4_results.iterrows():
    if row['SpkrIDA'] == row['SpkrIDB']:
        # Genuine
        genuine4.append(row)
    else:
        # Impostor
        impostor4.append(row)

print(len(genuine4),'genuine scores,', len(impostor4), 'impostor scores loaded')


genuine1_scores = [i['MatchScore'] for i in genuine1]
impostor1_scores = [i['MatchScore'] for i in impostor1]
total1_scores = np.concatenate([genuine1_scores, impostor1_scores])
ground1_truth = np.concatenate([[1 for i in range(len(genuine1_scores))], [0 for i in range(len(impostor1_scores))]])

genuine2_scores = [i['MatchScore'] for i in genuine2]
impostor2_scores = [i['MatchScore'] for i in impostor2]
total2_scores = np.concatenate([genuine2_scores, impostor2_scores])
ground2_truth = np.concatenate([[1 for i in range(len(genuine2_scores))], [0 for i in range(len(impostor2_scores))]])

genuine3_scores = [i['MatchScore'] for i in genuine3]
impostor3_scores = [i['MatchScore'] for i in impostor3]
total3_scores = np.concatenate([genuine3_scores, impostor3_scores])
ground3_truth = np.concatenate([[1 for i in range(len(genuine3_scores))], [0 for i in range(len(impostor3_scores))]])

genuine4_scores = [i['MatchScore'] for i in genuine4]
impostor4_scores = [i['MatchScore'] for i in impostor4]
total4_scores = np.concatenate([genuine4_scores, impostor4_scores])
ground4_truth = np.concatenate([[1 for i in range(len(genuine4_scores))], [0 for i in range(len(impostor4_scores))]])


dprime_1 = math.sqrt(2) * Z(roc_auc_score(ground1_truth,total1_scores))
dprime_2 = math.sqrt(2) * Z(roc_auc_score(ground2_truth,total2_scores))
dprime_3 = math.sqrt(2) * Z(roc_auc_score(ground3_truth,total3_scores))
dprime_4 = math.sqrt(2) * Z(roc_auc_score(ground4_truth,total4_scores))
print('E-vector d-prime=',dprime_1)
print('ecapa+vox d-prime=',dprime_2)
print('ecapa+vox+msp ft d-prime=',dprime_3)
print('ecapa+msp d-prime=',dprime_4)