# Example speaker verification (2 samples at a time with a pretrained ECAPA-TDNN using VoxCeleb data)

import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb") # TODO: Train this on MSP from scratch

# this is a genuine match
signal1, fs =torchaudio.load('/research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0014_0252.wav')
signal2, fs =torchaudio.load('/research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Audios/MSP-PODCAST_0023_0001.wav')

embed1 = classifier.encode_batch(signal1)[0][0]
embed2 = classifier.encode_batch(signal2)[0][0]

assert embed1.shape == embed2.shape

print('Match Score', cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1)))