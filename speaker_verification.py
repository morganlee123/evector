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

#
# Author: Morgan Sandler (sandle20@msu.edu)
# The purpose of this file is to perform a speaker verification between two directly provided .wav audio samples. 
# Note: this file will also perform the VAD/other preprocessing necessary if the get_speaker_embedding flag is set to preprocessing=True (which is default nature)
#




# TRAINING SET EXAMPLES
# Genuine Pair Example: python speaker_verification.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0014_0252.wav /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0023_0001.wav 
# MATCH SCORE: 0.70248353

# Impostor Pair Example: python speaker_verification.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0014_0252.wav /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_1588_0012_0002.wav 
# MATCH SCORE: 0.08596447

################################################################

# VALIDATION SET EXAMPLES
# Genuine Pair Example: python speaker_verification.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0260_0469.wav /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0260_0485_0001.wav
# MATCH SCORE: 0.84595656

# Impostor Pair Example: python speaker_verification.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0260_0469.wav /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0856_0923.wav
# MATCH SCORE: 0.09403051

################################################################

# TESTING SET 1 EXAMPLES
# Genuine Pair Example: python speaker_verification.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0147_0225.wav /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0448_0190.wav
# MATCH SCORE: 

# Impostor Pair Example: python speaker_verification.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0147_0225.wav /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0498_0375_0001.wav
# MATCH SCORE: 





def get_model():
    model_save_path = Path('/research/iprobe-sandle20/Playground/evector/encoder/saved_models/first.pt') # NOTE: Add your own path here to your saved model
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
    embed, partial_embeds, _  = encoder.embed_utterance(ref_audio, return_partials=True)

    if(normalize):
        embed = embed / np.linalg.norm(embed)
    return embed

# NOTE: Here is the command to run: python speaker_verification.py <FILE1> <FILE2>
# write the main program here
embed1 = get_speaker_embedding(sys.argv[1])
embed2 = get_speaker_embedding(sys.argv[2])

assert embed1.shape == embed2.shape

print('Match Score', cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1)))