# E-Vector a.k.a Emotion-Scenario Vector for Speaker Recognition 
## Morgan Sandler
This is mostly my thesis work! Computed embeddings for all test sets are included in this repository within the ExperimentData/ folders. In addition, the code and figures from the papers may be found in their respective folders. If you would like pre-trained models, please reach out to sandle20@msu.edu. I hope to soon upload them for inference on HuggingFace :-)

Training and Evaluation data can be requested from the MSP lab at UTDallas. Thank you to Prof. Carlos Busso for granting permission of the data. The dataset details are here: [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)

Architecture of E-Vector ([Full Paper](https://arxiv.org/abs/2305.07997))
![Architecture](Figures/EVectorArchitecture.pdf)

Here is a brief description of each relevant file:
- encoder_preprocess.py preprocess the raw data from the MSP-Podcast root directory and stores preprocessed it in Data/
- encoder_train.py will train models from scratch or could be refined to fine-tune models if you so desire.
- speaker_verification_MSP.py will test SV on the MSP-Podcast testing sets.
- Ignore speaker_verification_mass.py. It is **vestigial and mostly trash**. I did not use this file in the end, but I am double-checking the code to make sure it isn't useful in some capacity.

Of course, you may have many questions about the code since it is vast and complicated. If you have any questions, feel free to reach out and I will do my best to answer them! - Morgan


### Resources/References
- TODO: I'll list the papers here
- https://github.com/lawlict/ECAPA-TDNN
- https://github.com/iPRoBe-lab/DeepTalk
- https://speechbrain.github.io/
