#
# Purpose: Load E-Vector embeddings and learn their emotion classes (provided their labels)
# Author: Morgan Sandler (sandle20@msu.edu)
# Goal: Soft quantification of emotion content in the speaker embeddings a.k.a empirical proof that
# emotion is prevalent in the speaker identity. 
#
# Usage: python evec_ser_experiment.py [test1/test2] <dataset root>
# Example: python evec_ser_experiment.py test2 /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/
# 


import pickle
import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns

test_set = sys.argv[1]

with open('../../ExperimentData/val_speaker_embeddings.pickle', 'rb') as handle:
    embeds = pickle.load(handle)

with open('../../ExperimentData/'+test_set+'_speaker_embeddings.pickle', 'rb') as handle:
    test_embeds = pickle.load(handle)

print(len(embeds), 'train embeddings loaded')
print(len(test_embeds), 'test embeddings loaded')

dataset_root = sys.argv[2] + '/Labels/labels_concensus.csv'
if not dataset_root:
    pass
else:
    # Preprocess all speakers
    msp_information = pd.read_csv(dataset_root)
    # Drop the Other and No agreement emotion labels
    validation_set = msp_information[msp_information['Split_Set'] == 'Validation'][msp_information['SpkrID'] != 'Unknown']
    test1_set = msp_information[msp_information['Split_Set'] == 'Test1'][msp_information['SpkrID'] != 'Unknown']
    test2_set = msp_information[msp_information['Split_Set'] == 'Test2'][msp_information['SpkrID'] != 'Unknown']

    test1_set.set_index('FileName',inplace=True)
    test2_set.set_index('FileName',inplace=True)
    validation_set.set_index('FileName',inplace=True)

    if test_set == 'test1':
        # generate labels in correct order:
        le = preprocessing.LabelEncoder()
        le.fit(test1_set['EmoClass'])
        print(le.classes_)
        X = []
        y = []
        y_readable = []

        for filename in embeds:
            if embeds[filename] is None:
                continue
            curr_label = validation_set.loc[filename]['EmoClass']
            if curr_label == 'O' or curr_label == 'X':
                continue
            encoded_label = le.transform([curr_label])
            X.append(embeds[filename])
            y.append(encoded_label[0])
            y_readable.append(curr_label)
        
        X_test = []
        y_test = []
        y_test_readable = []
        for filename in test_embeds:
            if test_embeds[filename] is None:
                continue
            curr_label = test1_set.loc[filename]['EmoClass']
            if curr_label == 'X' or curr_label == 'O':
                continue
            encoded_label = le.transform([curr_label])
            X_test.append(test_embeds[filename])
            y_test.append(encoded_label[0])
            y_test_readable.append(curr_label)

        # TRAIN BASIC MLP a.k.a transferring knowledge learnt
        model = MLPClassifier(random_state=1,hidden_layer_sizes=(100,100,100,), max_iter=300)
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # random undersample code for balanced train set
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print('class counts', class_counts)
        # end undersample code

        model.fit(X, y)


        print('Beginning 5-fold CV')
        results = cross_val_score(model, X_test, y_test, cv=None, n_jobs=5, scoring="f1_weighted", verbose=1)
        print('All sessions:', results)
        mean_score = results.mean()
        std_score = results.std()
        print('Mean:', mean_score)
        print('Std:', std_score)

        print('generating conf matrix')
        conf_mat = confusion_matrix(y_test, model.predict(X_test))
        print(conf_mat)    
        plt.figure()
        sns.heatmap(conf_mat, annot=True)
        plt.savefig('./'+test_set+'_ser_confusion_matrix.png')


    elif test_set == 'test2':
        # generate labels in correct order:
        le = preprocessing.LabelEncoder()
        le.fit(test2_set['EmoClass'])
        print(le.classes_)
        X = []
        y = []
        y_readable = []

        for filename in embeds:
            if embeds[filename] is None:
                continue
            curr_label = validation_set.loc[filename]['EmoClass']
            if curr_label == 'O' or curr_label == 'X':
                continue
            encoded_label = le.transform([curr_label])
            X.append(embeds[filename])
            y.append(encoded_label[0])
            y_readable.append(curr_label)
        
        X_test = []
        y_test = []
        y_test_readable = []
        for filename in test_embeds:
            if test_embeds[filename] is None:
                continue
            curr_label = test2_set.loc[filename]['EmoClass']
            if curr_label == 'O' or curr_label == 'X':
                continue
            encoded_label = le.transform([curr_label])
            X_test.append(test_embeds[filename])
            y_test.append(encoded_label[0])
            y_test_readable.append(curr_label)

        # TRAIN BASIC MLP a.k.a transferring knowledge learnt
        model = MLPClassifier(random_state=1, hidden_layer_sizes=(100,100,100,), max_iter=300)
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)


        # random undersample code for balanced train set
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print('class counts', class_counts)
        # end undersample code


        model.fit(X, y)

        from sklearn.model_selection import cross_val_score
        print('Beginning 5-fold CV')
        results = cross_val_score(model, X_test, y_test, cv=None, n_jobs=5, scoring="f1_weighted", verbose=1)
        print('All sessions:', results)
        mean_score = results.mean()
        std_score = results.std()
        print('Mean:', mean_score)
        print('Std:', std_score)
    
        print('generating conf matrix')
        conf_mat = confusion_matrix(y_test, model.predict(X_test))
        #print(conf_mat)    

        ax = plt.gca()
        conf_mat = np.array(conf_mat) / sum(conf_mat[0]) * 100 # TODO: This doesn't seem to be working for generating a %
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Speech Emotion Recognition (MSP-Podcast)')
        labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Other', 'Sad', 'Surprise', 'No Agreement']
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        sns.heatmap(conf_mat, annot=True, cmap='Blues', ax=ax)
        plt.savefig('./'+test_set+'_ser_confusion_matrix.png')

    else:
        print('test set does not exist')