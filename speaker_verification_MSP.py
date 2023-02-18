import io
import sys
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from encoder import inference as encoder
from encoder import audio
import librosa
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm
import pickle

#
# Author: Morgan Sandler (sandle20@msu.edu)
# The purpose of this file is to perform a speaker verification for E-Vector over the MSP-Podcast testing sets 
# Note: this file will also perform the VAD/other preprocessing necessary if the get_speaker_embedding flag is set to preprocessing=True (which is default nature)
#

# NOTE: Here is the syntax to run: python speaker_verification_MSP.py <val/test1/test2> <dataset_root_path>
# example: python speaker_verification_MSP.py val /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/

def get_model():
    model_save_path = Path('/research/iprobe-sandle20/Playground/evector/encoder/saved_models/first_backups/first_bak_105000.pt') # NOTE: Add your own path here to your saved model
    module_name = 'model_GST'
    encoder.load_model(model_save_path, module_name=module_name)
    return encoder

def get_tensor(file_path, preprocess=True, sampling_rate=8000, duration=None):
    if(preprocess):
        ref_audio = encoder.preprocess_wav(file_path)
    else:
        ref_audio, sr = librosa.load(file_path, sr=sampling_rate)

    if(duration is not None):
        ref_audio = ref_audio[0:int(duration*sampling_rate)]
    return ref_audio

encoder = get_model()
def get_speaker_embedding(file_path, preprocess=True, sampling_rate=16000, duration=None, normalize=True):
    ref_audio = get_tensor(file_path, preprocess=preprocess, sampling_rate=sampling_rate, duration=duration)
    #print('ref audio', ref_audio.shape)
    embed, partial_embeds, _  = encoder.embed_utterance(ref_audio, return_partials=True)
    if embed is None:
        return None
    #print(embed.shape)
    if(normalize):
        embed = embed / np.linalg.norm(embed)
    return embed

# Main Program

testing_set = sys.argv[1]

dataset_root = sys.argv[2] + '/Labels/labels_concensus.csv'
if not dataset_root:
    pass
else:
    # Preprocess all speakers
    import pandas as pd
    msp_information = pd.read_csv(dataset_root)

    train_set = msp_information[msp_information['Split_Set'] == 'Train'][msp_information['SpkrID'] != 'Unknown'] 
    validation_set = msp_information[msp_information['Split_Set'] == 'Validation'][msp_information['SpkrID'] != 'Unknown']
    test1_set = msp_information[msp_information['Split_Set'] == 'Test1'][msp_information['SpkrID'] != 'Unknown']
    test2_set = msp_information[msp_information['Split_Set'] == 'Test2'][msp_information['SpkrID'] != 'Unknown']

    results = []
    
    if testing_set == 'val':
        print('Validation set selected')
        print(validation_set)
        combos = list(combinations(validation_set['FileName'], 2))
        print(len(combos), 'combinations of files in the VAL set')
        validation_set.set_index('FileName',inplace=True)
        
        speaker_embed_cache = {}

        #print(split_arrays)
        try:
            for sample_a, sample_b in tqdm(combos):
                sample_a_info = validation_set.loc[sample_a]
                sample_b_info = validation_set.loc[sample_b]
                #print(sample_a_info, sample_b_info)

                path_a = Path(sys.argv[2] + 'Audios/' +sample_a)
                path_b = Path(sys.argv[2] + 'Audios/' +sample_b)
                #print(path_a, path_b)
                

                # Simple caching system for computed embeddings
                if sample_a in speaker_embed_cache:
                    #print('using cache')
                    embed1 = speaker_embed_cache[sample_a]
                else:
                    embed1 = get_speaker_embedding(path_a)
                    speaker_embed_cache[sample_a] = embed1

                if sample_b in speaker_embed_cache:
                    #print('using cache')
                    embed2 = speaker_embed_cache[sample_b]
                else:
                    embed2 = get_speaker_embedding(path_b)
                    speaker_embed_cache[sample_b] = embed2

                if embed1 is None or embed2 is None:
                    continue # skip this test

                assert embed1.shape == embed2.shape

                match_score = cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))
                #print('Match Score', match_score)
                
                row = {
                    'FileNameA': sample_a_info.name,
                    'EmoClassA': sample_a_info['EmoClass'],
                    'EmoActA': sample_a_info['EmoAct'],
                    'EmoValA': sample_a_info['EmoVal'],
                    'EmoDomA': sample_a_info['EmoDom'],
                    'SpkrIDA': sample_a_info['SpkrID'],
                    'GenderA': sample_a_info['Gender'],
                    'SplitSetA': sample_a_info['Split_Set'],

                    'FileNameB': sample_b_info.name,
                    'EmoClassB': sample_b_info['EmoClass'],
                    'EmoActB': sample_b_info['EmoAct'],
                    'EmoValB': sample_b_info['EmoVal'],
                    'EmoDomB': sample_b_info['EmoDom'],
                    'SpkrIDB': sample_b_info['SpkrID'],
                    'GenderB': sample_b_info['Gender'],
                    'SplitSetB': sample_b_info['Split_Set'],

                    'MatchScore': match_score[0][0]
                }

                results.append(row)
        except KeyboardInterrupt:
            # save early if keeb interrupt
            final_df = pd.DataFrame(results)
            final_df.to_csv('ExperimentData/val_results.csv')

            with open('ExperimentData/val_speaker_embeddings.pickle', 'wb') as handle:
                pickle.dump(speaker_embed_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Data Saved')
    elif testing_set == 'test1':
        print('Test1 set selected')
        print(test1_set)
        combos = list(combinations(test1_set['FileName'], 2))
        print(len(combos), 'combinations of files in the TEST1 set')
        test1_set.set_index('FileName',inplace=True)
        
        speaker_embed_cache = {}

        #print(split_arrays)
        try:
            for sample_a, sample_b in tqdm(combos):
                sample_a_info = test1_set.loc[sample_a]
                sample_b_info = test1_set.loc[sample_b]
                #print(sample_a_info, sample_b_info)

                path_a = Path(sys.argv[2] + 'Audios/' +sample_a)
                path_b = Path(sys.argv[2] + 'Audios/' +sample_b)
                #print(path_a, path_b)
                

                # Simple caching system for computed embeddings
                if sample_a in speaker_embed_cache:
                    #print('using cache')
                    embed1 = speaker_embed_cache[sample_a]
                else:
                    embed1 = get_speaker_embedding(path_a)
                    speaker_embed_cache[sample_a] = embed1

                if sample_b in speaker_embed_cache:
                    #print('using cache')
                    embed2 = speaker_embed_cache[sample_b]
                else:
                    embed2 = get_speaker_embedding(path_b)
                    speaker_embed_cache[sample_b] = embed2

                if embed1 is None or embed2 is None:
                    continue # skip this test

                assert embed1.shape == embed2.shape

                match_score = cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))
                #print('Match Score', match_score)
                
                row = {
                    'FileNameA': sample_a_info.name,
                    'EmoClassA': sample_a_info['EmoClass'],
                    'EmoActA': sample_a_info['EmoAct'],
                    'EmoValA': sample_a_info['EmoVal'],
                    'EmoDomA': sample_a_info['EmoDom'],
                    'SpkrIDA': sample_a_info['SpkrID'],
                    'GenderA': sample_a_info['Gender'],
                    'SplitSetA': sample_a_info['Split_Set'],

                    'FileNameB': sample_b_info.name,
                    'EmoClassB': sample_b_info['EmoClass'],
                    'EmoActB': sample_b_info['EmoAct'],
                    'EmoValB': sample_b_info['EmoVal'],
                    'EmoDomB': sample_b_info['EmoDom'],
                    'SpkrIDB': sample_b_info['SpkrID'],
                    'GenderB': sample_b_info['Gender'],
                    'SplitSetB': sample_b_info['Split_Set'],

                    'MatchScore': match_score[0][0]
                }

                results.append(row)
        except KeyboardInterrupt:
            # save early if keeb interrupt
            final_df = pd.DataFrame(results)
            final_df.to_csv('ExperimentData/test1_results.csv')

            with open('ExperimentData/test1_speaker_embeddings.pickle', 'wb') as handle:
                pickle.dump(speaker_embed_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Data Saved')
    elif testing_set == 'test2':
        print('Test2 set selected')
        print(test2_set)
        combos = list(combinations(test2_set['FileName'], 2))
        print(len(combos), 'combinations of files in the TEST2 set')
        test2_set.set_index('FileName',inplace=True)
        
        speaker_embed_cache = {}

        #print(split_arrays)
        try:
            for sample_a, sample_b in tqdm(combos):
                sample_a_info = test2_set.loc[sample_a]
                sample_b_info = test2_set.loc[sample_b]
                #print(sample_a_info, sample_b_info)

                path_a = Path(sys.argv[2] + 'Audios/' +sample_a)
                path_b = Path(sys.argv[2] + 'Audios/' +sample_b)
                #print(path_a, path_b)
                

                # Simple caching system for computed embeddings
                if sample_a in speaker_embed_cache:
                    #print('using cache')
                    embed1 = speaker_embed_cache[sample_a]
                else:
                    embed1 = get_speaker_embedding(path_a)
                    speaker_embed_cache[sample_a] = embed1

                if sample_b in speaker_embed_cache:
                    #print('using cache')
                    embed2 = speaker_embed_cache[sample_b]
                else:
                    embed2 = get_speaker_embedding(path_b)
                    speaker_embed_cache[sample_b] = embed2

                if embed1 is None or embed2 is None:
                    continue # skip this test

                assert embed1.shape == embed2.shape

                match_score = cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))
                #print('Match Score', match_score)
                
                row = {
                    'FileNameA': sample_a_info.name,
                    'EmoClassA': sample_a_info['EmoClass'],
                    'EmoActA': sample_a_info['EmoAct'],
                    'EmoValA': sample_a_info['EmoVal'],
                    'EmoDomA': sample_a_info['EmoDom'],
                    'SpkrIDA': sample_a_info['SpkrID'],
                    'GenderA': sample_a_info['Gender'],
                    'SplitSetA': sample_a_info['Split_Set'],

                    'FileNameB': sample_b_info.name,
                    'EmoClassB': sample_b_info['EmoClass'],
                    'EmoActB': sample_b_info['EmoAct'],
                    'EmoValB': sample_b_info['EmoVal'],
                    'EmoDomB': sample_b_info['EmoDom'],
                    'SpkrIDB': sample_b_info['SpkrID'],
                    'GenderB': sample_b_info['Gender'],
                    'SplitSetB': sample_b_info['Split_Set'],

                    'MatchScore': match_score[0][0]
                }

                results.append(row)
        except KeyboardInterrupt:
            # save early if keeb interrupt
            final_df = pd.DataFrame(results)
            final_df.to_csv('ExperimentData/test2_results.csv')

            with open('ExperimentData/test2_speaker_embeddings.pickle', 'wb') as handle:
                pickle.dump(speaker_embed_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Data Saved')
    else:
        print('oops that set doesnt exist. try again')

    final_df = pd.DataFrame(results)
    final_df.to_csv('ExperimentData/'+testing_set+'_results.csv')

    with open('ExperimentData/'+testing_set+'_speaker_embeddings.pickle', 'wb') as handle:
        pickle.dump(speaker_embed_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Data Saved')


