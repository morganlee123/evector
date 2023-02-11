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
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import pickle


# TODO: Write this program s.t. you may enter ```python speaker_verificiation_mass.py verification``` or ```python speaker_verificiation_mass.py test1``` and it will auto
# fetch the proper speaker_verficiation_dataset.py copy of the dataset and pull batches of samples in vectors to encode via a model/step of choice and then perform
# the cosine similartiy [-1, 1]. Note that with the cosine similarity in this range 0 is not a terrible evaluation. It's just perfectly in-between a match and non-match. Threshold
#i'm thinking could be around .1. Not sure yet though. I need to plot the distribution

# USAGE: python speaker_verification_mass.py /research/iprobe-sandle20/Playground/evector/Data/EVec/encoder/Validation/ 1 0
# COMMENTARY: Generates combos from scratch (0) and assumes the data is .npy format (1) 

# USAGE: python speaker_verification_mass.py /research/iprobe-sandle20/Playground/evector/Data/EVec/encoder/Validation/ 0 0
# COMMENTARY: Generates combos from scratch (0) and assumes the data is .wav format (0) 


def get_model():
    model_save_path = Path('/research/iprobe-sandle20/Playground/evector/encoder/saved_models/first_backups/first_bak_105000.pt') # NOTE: Add your own path here to your saved model. Be careful of which step/model you are loading
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
    print(ref_audio.shape)
    embed, partial_embeds, _  = encoder.embed_utterance(ref_audio, return_partials=True)

    if(normalize):
        embed = embed / np.linalg.norm(embed)
    return embed

def get_combinations(root_of_data : Path, preprocessed_already=True):
    if not preprocessed_already:
        # Assumes that the input is .wav files
        list_of_samples = []
        
        paths = root_of_data.glob('**/*.wav')
        
        for sampleA, sampleB in tqdm(list(combinations(paths, 2))):
            identificationA = sampleA.parent.name
            identificationB = sampleB.parent.name
            list_of_samples.append(((identificationA, sampleA), (identificationB, sampleB)))
        
        
        #print(list_of_samples[0:10])
        return list_of_samples
        
    else:
        # Assumes the input is .npy files (already preprocessed)
        # run the get_speaker_embedding with preprocess = False. This will allow for the use
        # of the pre-computed .npy files in the Data/ directory. If you do not have this
        # then using preprocessed = False on this function

        list_of_samples = []
        # TODO @Morgan remove this when done
        paths = root_of_data.glob('**/*.npy')
        
        for sampleA, sampleB in tqdm(list(combinations(paths, 2))):
            identificationA = sampleA.parent.name
            identificationB = sampleB.parent.name
            list_of_samples.append(((identificationA, sampleA), (identificationB, sampleB)))
        
        
        #print(list_of_samples[0:10])
        return list_of_samples

def speaker_verification(path1, path2, already_preprocessed):
    embed1 = get_speaker_embedding(path1, preprocess=not already_preprocessed)
    embed2 = get_speaker_embedding(path2, preprocess=not already_preprocessed)

    assert embed1.shape == embed2.shape

    print('Match Score', cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1)))

# Main program is as follows

# Get all the combos of vectors and their corresp. IDs

if not int(sys.argv[3]):
    combos = get_combinations(Path(sys.argv[1])) # returns this format: 
    # [((ID_SAMPLEA, VECTOR_SAMPLEA), (ID_SAMPLEB, VECTOR_SAMPLEB)) ..... ]
    # Save the combos. This part can be computationally expensive. That's why
    with open('ExperimentData/experiment_combos.pickle', 'wb') as handle:
        pickle.dump(combos, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # else load precomputed
    with open('ExperimentData/experiment_combos.pickle', 'rb') as handle:
        combos = pickle.load(handle)

results = []

for sampleA, sampleB in tqdm(combos):
    print(sampleA, sampleB)
    #match_score = speaker_verification()
    #print(match_score)
    break
    
    #results.append(('gen' if sampleA[0] == sampleB[0] else 'imp', match_score, emotion))

# Save the results of the experiment
#np.savetxt("ExperimentData/SpeakerID_Experiment_Results.csv", 
#           results,
#           delimiter =", ", 
#           fmt ='% s')

