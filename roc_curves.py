import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: Compute all the ROC curves for a given MSPPod test set

# SYNTAX. Choose your set between val, test1, or test2
# e.g., python roc_curves.py [val/test1/test2]


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


from sklearn import metrics


ax = plt.gca()
d1 = metrics.RocCurveDisplay.from_predictions(ground1_truth, total1_scores, ax=ax, name='E-Vector w/ MSP-Pod')
d2 = metrics.RocCurveDisplay.from_predictions(ground2_truth, total2_scores, ax=ax, name='ECAPA-TDNN w/ VoxCeleb1+2')
d3 = metrics.RocCurveDisplay.from_predictions(ground3_truth, total3_scores, ax=ax, name='ECAPA-TDNN w/ VoxC1+2 + MSP-Pod')
d4 = metrics.RocCurveDisplay.from_predictions(ground4_truth, total4_scores, ax=ax, name='ECAPA-TDNN w/ MSP-Pod')

plt.xscale("log") # NOTE: This gives semi-log. Feel free to comment this out
plt.ylabel('True Match Rate')
plt.xlabel('False Match Rate')
plt.legend(loc='upper left')
plt.show()
plt.savefig(sys.argv[1]+'_roc.jpg')

print('Done. saved '+sys.argv[1]+'_roc.jpg')