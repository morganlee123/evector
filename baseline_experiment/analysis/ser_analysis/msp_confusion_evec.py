# Author: Morgan Sandler (sandle20@msu.edu)
# generate confusio for msp pod using evec embeddings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# nicer visualization of the conf data
conf_mat = [[  68   ,49 , 48  ,27 ,148 ,184  ,21  ,67 ],
 [  53  ,65 , 56  ,35 ,183 ,222  ,28  ,76 ],
  [  62 , 60 , 47  ,33 ,180 ,217 , 23 , 73 ],
   [  16 , 43 , 37 ,  7 , 98 ,103 , 10 , 28 ],
    [ 206 ,252 ,159 ,141,1040,1169 ,125 ,322 ],
     [ 204 ,381 ,260 ,137,1291,1613 ,208 ,452 ],
      [  23 , 37 , 35 , 13 ,115, 168 , 17 , 55 ],
       [  52 ,132,  63 , 32, 293, 284 , 30 , 85 ]]

ax= plt.subplot()
for i, row in enumerate(conf_mat):
    conf_mat[i] = np.array(row) / np.sum(row) * 100
sns.heatmap(conf_mat, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('MSP-Podcast Speech Emotion Recognition (SVM)')

labels = ['Anger','Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.savefig('./evec_ser.png')
