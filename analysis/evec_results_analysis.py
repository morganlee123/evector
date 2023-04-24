import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: Perform analysis on the computed results from E-Vector.

# SYNTAX. Choose your set between val, test1, or test2
# e.g., python evec_results_analysis.py [val/test1/test2] [10token, 20token, 5token] > [val/test1/test2].txt
tokensize = sys.argv[2]

p = 0.01  # 1% or .01 of the lines works for val and test2.     0.5% or .005 works for test1
if sys.argv[1] == 'test1':
    p=.005
else:
    p=0.01
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
test_results = pd.read_csv('../ExperimentData/'+tokensize+'/'+sys.argv[1]+'_results.csv',
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

""" NOTE: I moved this to a separate file called evec_emotion_analysis.py - Morgan


# Analysis 2: Average Genuine and Impostor Match Score per Emotion



emotions = test_results['EmoClassA'].unique()
print(emotions)
emotions = ['A', 'S', 'H', 'U', 'F', 'D', 'C', 'N', 'O', 'X']

emo_map = {'A':'Angry', 'S':'Sad', 'H':'Happy', 'U':'Surprise', 'F':'Fear', 'D':'Disgust', 'C':'Contempt', 'N':'Neutral', 'O':'Other', 'X':'No Agreement'}

# 2.1 - E-Vector Genuine INTRA Emotion
for emo in emotions:
    print('EMOTION ', emo_map[emo])
    gen_mean_intra = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].mean() # mean match score
    gen_mean_intra_val =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValA'].mean() + # mean valence
                            test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValB'].mean()) / 2
    gen_mean_intra_dom =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoDomA'].mean() + # mean dominance
                            test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoDomB'].mean()) / 2
    gen_mean_intra_aro =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActA'].mean() + # mean activation/arousal
                            test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActB'].mean()) / 2

    try:
        gen_max_intra_idx = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].idxmax()
        gen_max_intra = test_results.iloc[gen_max_intra_idx]['MatchScore']
        gen_max_intra_aro = (test_results.iloc[gen_max_intra_idx]['EmoActA'] + test_results.iloc[gen_max_intra_idx]['EmoActB']) / 2
        gen_max_intra_dom = (test_results.iloc[gen_max_intra_idx]['EmoDomA'] + test_results.iloc[gen_max_intra_idx]['EmoDomB']) / 2
        gen_max_intra_val = (test_results.iloc[gen_max_intra_idx]['EmoValA'] + test_results.iloc[gen_max_intra_idx]['EmoValB']) / 2
    except:
        gen_max_intra = 'N/A'
        gen_max_intra_aro = 'N/A'
        gen_max_intra_val = 'N/A'
        gen_max_intra_dom = 'N/A'
    try:
        gen_min_intra_idx = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].idxmin()
        gen_min_intra = test_results.iloc[gen_min_intra_idx]['MatchScore']
        gen_min_intra_aro = (test_results.iloc[gen_min_intra_idx]['EmoActA'] + test_results.iloc[gen_min_intra_idx]['EmoActB']) / 2
        gen_min_intra_dom = (test_results.iloc[gen_min_intra_idx]['EmoDomA'] + test_results.iloc[gen_min_intra_idx]['EmoDomB']) / 2
        gen_min_intra_val = (test_results.iloc[gen_min_intra_idx]['EmoValA'] + test_results.iloc[gen_min_intra_idx]['EmoValB']) / 2
    except:
        gen_min_intra = 'N/A'
        gen_min_intra_aro = 'N/A'
        gen_min_intra_val = 'N/A'
        gen_min_intra_dom = 'N/A'

    print('GENUINE INTRA', emo_map[emo],'\n',
      'Mean Score:', gen_mean_intra, 'A:',gen_mean_intra_aro,' V:',gen_mean_intra_val, 'D:',gen_mean_intra_dom,'\n',
      'Max Score:', gen_max_intra, 'A:',gen_max_intra_aro,' V:',gen_max_intra_val, 'D:',gen_max_intra_dom,'\n',
      'Min Score:', gen_min_intra, 'A:',gen_min_intra_aro,' V:',gen_min_intra_val, 'D:',gen_min_intra_dom,'\n',)

    print('----------------------------------------------------------------------')
# END 2.1

# 2.2 E-Vector Genuine INTER Emotion

# get combos
from itertools import combinations
emo_combos = list(combinations(emotions,2))
for emoA, emoB in emo_combos:
    print('EMOTION ', emo_map[emoA], 'vs EMOTION', emo_map[emoB])
    gen_mean_inter = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].mean() # mean match score
    gen_mean_inter_val =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValA'].mean() + # mean valence
                            test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValB'].mean()) / 2
    gen_mean_inter_dom =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoDomA'].mean() + # mean dominance
                            test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoDomB'].mean()) / 2
    gen_mean_inter_aro =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActA'].mean() + # mean activation/arousal
                            test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActB'].mean()) / 2

    try:
        gen_max_inter_idx = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].idxmax()
        gen_max_inter = test_results.iloc[gen_max_inter_idx]['MatchScore']
        gen_max_inter_aro = (test_results.iloc[gen_max_inter_idx]['EmoActA'] + test_results.iloc[gen_max_inter_idx]['EmoActB']) / 2
        gen_max_inter_dom = (test_results.iloc[gen_max_inter_idx]['EmoDomA'] + test_results.iloc[gen_max_inter_idx]['EmoDomB']) / 2
        gen_max_inter_val = (test_results.iloc[gen_max_inter_idx]['EmoValA'] + test_results.iloc[gen_max_inter_idx]['EmoValB']) / 2
    except:
        gen_max_inter = 'N/A'
        gen_max_inter_aro = 'N/A'
        gen_max_inter_val = 'N/A'
        gen_max_inter_dom = 'N/A'
    try:
        gen_min_inter_idx = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].idxmin()
        gen_min_inter = test_results.iloc[gen_min_inter_idx]['MatchScore']
        gen_min_inter_aro = (test_results.iloc[gen_min_inter_idx]['EmoActA'] + test_results.iloc[gen_min_inter_idx]['EmoActB']) / 2
        gen_min_inter_dom = (test_results.iloc[gen_min_inter_idx]['EmoDomA'] + test_results.iloc[gen_min_inter_idx]['EmoDomB']) / 2
        gen_min_inter_val = (test_results.iloc[gen_min_inter_idx]['EmoValA'] + test_results.iloc[gen_min_inter_idx]['EmoValB']) / 2
    except:
        gen_min_inter = 'N/A'
        gen_min_inter_aro = 'N/A'
        gen_min_inter_val = 'N/A'
        gen_min_inter_dom = 'N/A'

    print('GENUINE INTER', emo_map[emoA],'vs', emo_map[emoB],'\n',
      'Mean Score:', gen_mean_inter, 'A:',gen_mean_inter_aro,' V:',gen_mean_inter_val, 'D:',gen_mean_inter_dom,'\n',
      'Max Score:', gen_max_inter, 'A:',gen_max_inter_aro,' V:',gen_max_inter_val, 'D:',gen_max_inter_dom,'\n',
      'Min Score:', gen_min_inter, 'A:',gen_min_inter_aro,' V:',gen_min_inter_val, 'D:',gen_min_inter_dom,'\n',)

    print('----------------------------------------------------------------------')


# 2.3 - E-Vector Impostor INTRA Emotion
for emo in emotions:
    print('EMOTION ', emo_map[emo])
    imp_mean_intra = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].mean() # mean match score
    imp_mean_intra_val =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValA'].mean() + # mean valence
                            test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValB'].mean()) / 2
    imp_mean_intra_dom =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoDomA'].mean() + # mean dominance
                            test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoDomB'].mean()) / 2
    imp_mean_intra_aro =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActA'].mean() + # mean activation/arousal
                            test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActB'].mean()) / 2

    try:
        imp_max_intra_idx = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].idxmax()
        imp_max_intra = test_results.iloc[imp_max_intra_idx]['MatchScore']
        imp_max_intra_aro = (test_results.iloc[imp_max_intra_idx]['EmoActA'] + test_results.iloc[imp_max_intra_idx]['EmoActB']) / 2
        imp_max_intra_dom = (test_results.iloc[imp_max_intra_idx]['EmoDomA'] + test_results.iloc[imp_max_intra_idx]['EmoDomB']) / 2
        imp_max_intra_val = (test_results.iloc[imp_max_intra_idx]['EmoValA'] + test_results.iloc[imp_max_intra_idx]['EmoValB']) / 2
    except:
        imp_max_intra = 'N/A'
        imp_max_intra_aro = 'N/A'
        imp_max_intra_val = 'N/A'
        imp_max_intra_dom = 'N/A'
    try:
        imp_min_intra_idx = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].idxmin()
        imp_min_intra = test_results.iloc[imp_min_intra_idx]['MatchScore']
        imp_min_intra_aro = (test_results.iloc[imp_min_intra_idx]['EmoActA'] + test_results.iloc[imp_min_intra_idx]['EmoActB']) / 2
        imp_min_intra_dom = (test_results.iloc[imp_min_intra_idx]['EmoDomA'] + test_results.iloc[imp_min_intra_idx]['EmoDomB']) / 2
        imp_min_intra_val = (test_results.iloc[imp_min_intra_idx]['EmoValA'] + test_results.iloc[imp_min_intra_idx]['EmoValB']) / 2
    except:
        imp_min_intra = 'N/A'
        imp_min_intra_aro = 'N/A'
        imp_min_intra_val = 'N/A'
        imp_min_intra_dom = 'N/A'

    print('IMPOSTOR INTRA', emo_map[emo],'\n',
      'Mean Score:', imp_mean_intra, 'A:',imp_mean_intra_aro,' V:',imp_mean_intra_val, 'D:',imp_mean_intra_dom,'\n',
      'Max Score:', imp_max_intra, 'A:',imp_max_intra_aro,' V:',imp_max_intra_val, 'D:',imp_max_intra_dom,'\n',
      'Min Score:', imp_min_intra, 'A:',imp_min_intra_aro,' V:',imp_min_intra_val, 'D:',imp_min_intra_dom,'\n',)

    print('----------------------------------------------------------------------')
# END 2.1

# 2.2 E-Vector Impostor INTER Emotion

# get combos
from itertools import combinations
emo_combos = list(combinations(emotions,2))
for emoA, emoB in emo_combos:
    print('EMOTION ', emo_map[emoA], 'vs EMOTION', emo_map[emoB])
    imp_mean_inter = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].mean() # mean match score
    imp_mean_inter_val =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValA'].mean() + # mean valence
                            test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValB'].mean()) / 2
    imp_mean_inter_dom =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoDomA'].mean() + # mean dominance
                            test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoDomB'].mean()) / 2
    imp_mean_inter_aro =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActA'].mean() + # mean activation/arousal
                            test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActB'].mean()) / 2

    try:
        imp_max_inter_idx = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].idxmax()
        imp_max_inter = test_results.iloc[imp_max_inter_idx]['MatchScore']
        imp_max_inter_aro = (test_results.iloc[imp_max_inter_idx]['EmoActA'] + test_results.iloc[imp_max_inter_idx]['EmoActB']) / 2
        imp_max_inter_dom = (test_results.iloc[imp_max_inter_idx]['EmoDomA'] + test_results.iloc[imp_max_inter_idx]['EmoDomB']) / 2
        imp_max_inter_val = (test_results.iloc[imp_max_inter_idx]['EmoValA'] + test_results.iloc[imp_max_inter_idx]['EmoValB']) / 2
    except:
        imp_max_inter = 'N/A'
        imp_max_inter_aro = 'N/A'
        imp_max_inter_val = 'N/A'
        imp_max_inter_dom = 'N/A'
    try:
        imp_min_inter_idx = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].idxmin()
        imp_min_inter = test_results.iloc[imp_min_inter_idx]['MatchScore']
        imp_min_inter_aro = (test_results.iloc[imp_min_inter_idx]['EmoActA'] + test_results.iloc[imp_min_inter_idx]['EmoActB']) / 2
        imp_min_inter_dom = (test_results.iloc[imp_min_inter_idx]['EmoDomA'] + test_results.iloc[imp_min_inter_idx]['EmoDomB']) / 2
        imp_min_inter_val = (test_results.iloc[imp_min_inter_idx]['EmoValA'] + test_results.iloc[imp_min_inter_idx]['EmoValB']) / 2
    except:
        imp_min_inter = 'N/A'
        imp_min_inter_aro = 'N/A'
        imp_min_inter_val = 'N/A'
        imp_min_inter_dom = 'N/A'

    print('IMPOSTOR INTER', emo_map[emoA],'vs', emo_map[emoB],'\n',
      'Mean Score:', imp_mean_inter, 'A:',imp_mean_inter_aro,' V:',imp_mean_inter_val, 'D:',imp_mean_inter_dom,'\n',
      'Max Score:', imp_max_inter, 'A:',imp_max_inter_aro,' V:',imp_max_inter_val, 'D:',imp_max_inter_dom,'\n',
      'Min Score:', imp_min_inter, 'A:',imp_min_inter_aro,' V:',imp_min_inter_val, 'D:',imp_min_inter_dom,'\n',)

    print('----------------------------------------------------------------------')



"""



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
from speechbrain.utils.metric_stats import EER
from metric import minDCF

import torch


gen = torch.Tensor(genuine_scores) 
imp = torch.Tensor(impostor_scores)

# Get EER
print('Computing EER...')
val_eer, threshold_eer = EER(gen,imp)
print('EER:', val_eer, '. Threshold@', threshold_eer)

# MINDCF params c_miss=10, c_fa=1.0, p_target=0.01
print('Computing MinDCF...')
val_minDCF, threshold_dcf = minDCF(gen,imp,c_miss=10)
print('MinDCF:', val_minDCF, '. Threshold@', threshold_dcf) # TODO: For some reason this function allocates 300 GB


#################################




print('Average Male Genuine Match Score', np.mean(match_score_genuine_male), '(',len(match_score_genuine_male),') scores')
print('Average Female Genuine Match Score', np.mean(match_score_genuine_female), '(',len(match_score_genuine_female),') scores')


print('Average Male Impostor Match Score', np.mean(match_score_impostor_male), '(',len(match_score_impostor_male),') scores')
print('Average Female Impostor Match Score', np.mean(match_score_impostor_female),  '(',len(match_score_impostor_female),') scores')
print('Average Diff-Gender Impostor Match Score', np.mean(non_gender_match_impostor), '(',len(non_gender_match_impostor),') scores')
