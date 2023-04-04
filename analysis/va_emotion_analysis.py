# Implementation of D-prime from the following post
# https://stats.stackexchange.com/questions/492673/understanding-and-implementing-the-dprime-measure-in-python
# Goal: Compute d-prime between arousal/valence distributions for genuine and impostor pairs

# USAGE python va_emotion_analysis.py [test1/test2/val] > d_prime_res.txt 


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

scaling = 60
gen_descale = 0.5

models = ['evec', 'ecapavox', 'ecapavoxmsp', 'ecapamsp']

for model in models:
    if model == 'ecapavox':
        test_results = test2_results
    elif model =='ecapavoxmsp':
        test_results = test3_results
    elif model == 'ecapamsp':
        test_results = test4_results
            
    # gen intra
    for emo in emotions:
        gen_mean_intra = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].mean() # mean match score
        gen_mean_intra_val =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValA'].mean() + # mean valence
                                test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValB'].mean()) / 2
        gen_mean_intra_aro =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActA'].mean() + # mean activation/arousal
                                test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActB'].mean()) / 2
        plt.plot(gen_mean_intra_aro, gen_mean_intra_val, ms=gen_mean_intra * scaling * gen_descale, color='b', marker='o', alpha=0.5)

    # gen inter    
    from itertools import combinations
    emo_combos = list(combinations(emotions,2))
    for emoA, emoB in emo_combos:
        print('EMOTION ', emo_map[emoA], 'vs EMOTION', emo_map[emoB])
        gen_mean_inter = test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].mean() # mean match score
        gen_mean_inter_val =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValA'].mean() + # mean valence
                                test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValB'].mean()) / 2
        gen_mean_inter_aro =  (test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActA'].mean() + # mean activation/arousal
                                test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActB'].mean()) / 2
        plt.plot(gen_mean_inter_aro, gen_mean_inter_val, ms=gen_mean_inter * scaling * gen_descale, color='y', marker='o', alpha=0.5)

    # imp intra
    for emo in emotions:
        imp_mean_intra = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'].mean() # mean match score
        imp_mean_intra_val =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValA'].mean() + # mean valence
                                test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoValB'].mean()) / 2
        imp_mean_intra_aro =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActA'].mean() + # mean activation/arousal
                                test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['EmoActB'].mean()) / 2
        plt.plot(imp_mean_intra_aro, imp_mean_intra_val, ms=imp_mean_intra * scaling, color='orange', marker='o', alpha=0.5)

    # imp inter
    for emoA, emoB in emo_combos:
        imp_mean_inter = test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'].mean() # mean match score
        imp_mean_inter_val =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValA'].mean() + # mean valence
                                test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoValB'].mean()) / 2
        imp_mean_inter_aro =  (test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActA'].mean() + # mean activation/arousal
                                test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['EmoActB'].mean()) / 2
        plt.plot(imp_mean_inter_aro, imp_mean_inter_val, ms=imp_mean_inter * scaling, color='r', marker='o', alpha=0.5)

    plt.title('Match Scores of Intra/Inter-Emotion Pairings in Val-Aro Space')
    plt.xlabel('Arousal')
    plt.ylabel('Valence')
    import matplotlib.patches as mpatches
    gen_intra = mpatches.Patch(color='b', label='Intra-Emotion Genuine Score')
    imp_intra = mpatches.Patch(color='orange', label='Intra-Emotion Impostor Score')
    gen_inter = mpatches.Patch(color='y', label='Inter-Emotion Genuine Score')
    imp_inter = mpatches.Patch(color='r', label='Inter-Emotion Impostor Score')

    plt.legend(handles=[gen_intra,imp_intra, gen_inter, imp_inter], loc='upper left')

    plt.savefig('./'+sys.argv[1]+'_'+model+'_va.png')