import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: Perform analysis on the computed results from E-Vector.

# SYNTAX. Choose your set between val, test1, or test2
# e.g., python evec_results_analysis.py [val/test1/test2] > [val/test1/test2].txt


p = 0.01  # 1% or .01 of the lines works for val and test2.     0.5% or .005 works for test1
if sys.argv[1] == 'test1':
    p=.005
else:
    p=0.01
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
test_results = pd.read_csv('../ExperimentData/'+sys.argv[1]+'_results.csv',
                            header=0, 
                            skiprows=lambda i: i>0 and random.random() > p) 


print('results loaded in')

# Sort results into genuine and impostor lists
genuine, impostor = [], []
for i, row in test_results.iterrows():
    if row['SpkrIDA'] == row['SpkrIDB']:
        # Genuine
        genuine.append(row)
    else:
        # Impostor
        impostor.append(row)

print(len(genuine),'genuine scores,', len(impostor), 'impostor scores loaded')

# Analysis 1: Plot the Match Scores Distributions. Genuine vs Impostor
bin_size = 30

plt.hist([i['MatchScore'] for i in genuine], label='Genuine', bins=40, color='green', alpha=.5, density=True)
plt.hist([i['MatchScore'] for i in impostor], label='Impostor', bins=60, color='red', alpha=.5, density=True)
plt.legend()
plt.show()
plt.savefig(sys.argv[1]+'_dist.jpg')
# Analysis 2: Average Genuine and Impostor Match Score per Emotion
# TODO: combinations of emotions, and same emotion performance


# FILL THIS




# Analysis 3: Average Genuine and Impostor Match Score per Gender
match_score_genuine_male, match_score_genuine_female = [], []
for row in genuine:
    if row['GenderA'] == row['GenderB'] == 'Male':
        match_score_genuine_male.append(row['MatchScore'])
    elif row['GenderA'] == row['GenderB'] == 'Female':
        match_score_genuine_female.append(row['MatchScore'])
    else:
        print(row['GenderA'])
        break

match_score_impostor_male, match_score_impostor_female, non_gender_match_impostor = [], [], []

for row in impostor:
    # just double check for the same gender
    if row['GenderA'] == row['GenderB'] == 'Male':
        match_score_impostor_male.append(row['MatchScore'])
    elif row['GenderA'] == row['GenderB'] == 'Female':
        match_score_impostor_female.append(row['MatchScore'])
    else:
        non_gender_match_impostor.append(row['MatchScore'])


###### ROC CURVE
genuine_scores = [i['MatchScore'] for i in genuine]
impostor_scores = [i['MatchScore'] for i in impostor]
total_scores = np.concatenate([genuine_scores, impostor_scores])
ground_truth = np.concatenate([[1 for i in range(len(genuine_scores))], [0 for i in range(len(impostor_scores))]])

from sklearn import metrics
from sklearn.metrics import roc_curve
from bisect import bisect

metrics.RocCurveDisplay.from_predictions(ground_truth, total_scores)


fpr, tpr, _ = roc_curve(
            ground_truth,
            total_scores,
            pos_label=None,
            sample_weight=None,
            drop_intermediate=True,
        )
print('TMR@FMR{1%, 10%}', tpr[bisect(fpr, .01)], tpr[bisect(fpr, .1)])
plt.xscale("log") # NOTE: This gives semi-log. Feel free to comment this out
plt.savefig(sys.argv[1]+'_roc.jpg')
###### END ROC CURVE





################################# 

# Get minDCF
from speechbrain.utils.metric_stats import minDCF, EER
import torch


gen = torch.Tensor(genuine_scores) 
imp = torch.Tensor(impostor_scores)

# Get EER
print('Computing EER...')
val_eer, threshold_eer = EER(gen,imp)
print('EER:', val_eer, '. Threshold@', threshold_eer)

# MINDCF params c_miss=1.0, c_fa=1.0, p_target=0.01
#print('Computing MinDCF...')
#val_minDCF, threshold_dcf = minDCF(gen,imp)
#print('MinDCF:', val_minDCF, '. Threshold@', threshold_dcf) # TODO: For some reason this function allocates 300 GB



#################################




print('Average Male Genuine Match Score', np.mean(match_score_genuine_male), '(',len(match_score_genuine_male),') scores')
print('Average Female Genuine Match Score', np.mean(match_score_genuine_female), '(',len(match_score_genuine_female),') scores')


print('Average Male Impostor Match Score', np.mean(match_score_impostor_male), '(',len(match_score_impostor_male),') scores')
print('Average Female Impostor Match Score', np.mean(match_score_impostor_female),  '(',len(match_score_impostor_female),') scores')
print('Average Diff-Gender Impostor Match Score', np.mean(non_gender_match_impostor), '(',len(non_gender_match_impostor),') scores')
