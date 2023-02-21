from encoder.models import model_GST
from encoder.models import ecapa_tdnn
from speechbrain.lobes import ECAPA
from speechbrain.pretrained import EncoderClassifier
# Author: Morgan Sandler (sandle20@msu.edu)
# Simple script to load the model and count the parameters

print('E-Vector model parameters:',end=' ')
model = model_GST.SpeakerEncoder('cpu', 'cpu')
print(model.get_parm_count())

print('ECAPA-TDNN MSP model parameters:',end=' ')
model2 = ecapa_tdnn.SpeakerEncoder('cpu', 'cpu')
print(model2.get_parm_count())

print('ECAPA-TDNN Vox model parameters: 22.2 M') # source: https://drive.google.com/drive/folders/1-ahC1xeyPinAHp2oAohL-02smNWO41Cc

