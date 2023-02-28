import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from itertools import combinations

# Author: Morgan Sandler (sandle20@msu.edu)
# Purpose: Perform emotion analysis on the computed results from E-Vector.

# SYNTAX. Choose your set between val, test1, or test2
# e.g., python evec_emotion_analysis.py [val/test1/test2] > emotion_analysis_[val/test1/test2].txt


p = 0.01  # 1% or .01 of the lines works for val and test2.     0.5% or .005 works for test1
if sys.argv[1] == 'test1':
    p=.005
else:
    p=0.01
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
test_results = pd.read_csv('../ExperimentData/pretrained_ecapa_'+sys.argv[2]+'/'+sys.argv[1]+'_results.csv',
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


# Analysis 2: Average Genuine and Impostor Match Score per Emotion

# Neutral-Neutral Genuine/Impostor

emotions = ['A', 'S', 'H', 'U', 'F', 'D', 'C', 'N', 'O', 'X']

inter_emotions = list(combinations(emotions, 2))

emo_map = {'A':'Angry', 'S':'Sad', 'H':'Happy', 'U':'Surprise', 'F':'Fear', 'D':'Disgust', 'C':'Contempt', 'N':'Neutral', 'O':'Other', 'X':'No Agreement'}

# GEN INTRA
gen_intra = []
for emo in emotions:
    gen_intra.append( test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'] )

# GEN INTER
gen_inter = []
for emoA, emoB in inter_emotions:
    gen_inter.append( test_results[test_results['SpkrIDA'] == test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'] )

gen_intra.extend(gen_inter)

# IMP INTRA
imp_intra = []
for emo in emotions:
    imp_intra.append( test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emo][test_results['EmoClassB'] == emo]['MatchScore'] )
gen_intra.extend(imp_intra)

# IMP INTER
imp_inter = []
for emoA, emoB in inter_emotions:
    imp_inter.append( test_results[test_results['SpkrIDA'] != test_results['SpkrIDB']][test_results['EmoClassA'] == emoA][test_results['EmoClassB'] == emoB]['MatchScore'] )
gen_intra.extend(imp_inter)

fig = plt.figure(figsize =(10, 30))
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(gen_intra, patch_artist = True,
                notch ='True', vert = 0)
 
import random
colors = ['#005AB5'] * 10
colors.extend(['#c0c066'] * 45)
colors.extend(['#bd0909'] * 10)
colors.extend(['#655925'] * 45)
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
gen_intra_labels = [emo_map[e]+ ' (' + str(len(gen_intra[i]))+')' for i,e in enumerate(emotions)] 
gen_inter_labels = [emo_map[eA]+'-'+emo_map[eB]+' (' + str(len(gen_inter[i]))+')' for i,(eA, eB) in enumerate(inter_emotions)]
imp_intra_labels = [emo_map[e]+ ' (' + str(len(imp_intra[i]))+')' for i,e in enumerate(emotions)] 
imp_inter_labels = [emo_map[eA]+'-'+emo_map[eB]+' (' + str(len(imp_inter[i]))+')' for i,(eA, eB) in enumerate(inter_emotions)]


gen_intra_labels.extend(gen_inter_labels) 
gen_intra_labels.extend(imp_intra_labels) 
gen_intra_labels.extend(imp_inter_labels) # full list now


ax.set_yticklabels(
    gen_intra_labels
)
 
# Adding title
plt.title("Speaker Verification Score Variation Over Different Emotions")
plt.xlabel('Similarity Score (Cosine Similarity)')
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

import matplotlib.patches as mpatches

gen_intra = mpatches.Patch(color='#005AB5', label='Intra-Emotion Genuine Score')
imp_intra = mpatches.Patch(color='#bd0909', label='Intra-Emotion Impostor Score')
gen_inter = mpatches.Patch(color='#c0c066', label='Inter-Emotion Genuine Score')
imp_inter = mpatches.Patch(color='#655925', label='Inter-Emotion Impostor Score')

plt.legend(handles=[gen_intra,imp_intra, gen_inter, imp_inter], loc='lower left')
# show plot
plt.show()
plt.savefig(sys.argv[1]+'_'+sys.argv[2]+'_emo_boxplot.png')


"""
import random
c = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(15)]

plt.bar(scores['comp'], scores['sim'], color=c, edgecolor='black')
plt.xticks(rotation = 45)
plt.show()
"""