"""
Data preparation for MSP-Podcast Train for ECAPA-TDNN in Speechbrain
Author: Morgan Sandler (sandle20@msu.edu)

Example: CUDA_VISIBLE_DEVICES=2 python msp_finetune.py /research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/ train_ecapa_tdnn.yaml
"""

import sys
import pandas as pd
import speechbrain
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN 
from speechbrain.utils.parameter_transfer import Pretrainer

# Load the pandas dataframe and correct format
dataset_root = sys.argv[1] + 'Labels/labels_concensus.csv'
df = pd.read_csv(dataset_root)

df_train = df[df['Split_Set'] == 'Train'][df['SpkrID'] != 'Unknown']
df_train.rename(columns={'FileName':'ID'}, inplace=True)
df_train['file_path'] = df_train['ID'].apply(lambda x: sys.argv[1] + 'Audios/' + x)

# print(df_train.head()) debug line

df_train.to_csv('./train.csv', index=False)

# Load the Dynamic Item Dataset

dataset = DynamicItemDataset.from_csv("train.csv")

# Define the item pipelines
@speechbrain.utils.data_pipeline.takes("file_path")
@speechbrain.utils.data_pipeline.provides("signal")
def audio_pipeline(file_path):
      sig = speechbrain.dataio.dataio.read_audio(file_path)
      return sig

dataset.add_dynamic_item(audio_pipeline) 
dataset.add_dynamic_item(int, takes="SpkrID", provides="sID")
dataset.set_output_keys(["sID", "signal", "file_path", "id"],
)
print(dataset[0])



model = ECAPA_TDNN(input_size= 80,
                   channels= [1024, 1024, 1024, 1024, 3072],
                   kernel_sizes= [5, 3, 3, 3, 1],
                   dilations= [1, 2, 3, 4, 1],
                   attention_channels= 128,
                   lin_neurons = 192)

# Initialization of the pre-trainer 
#pretrain = Pretrainer(loadables={'model': model}, paths={'model': 'speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt'})

# We download the pretrained model from HuggingFace in this case
#pretrain.collect_files()
#pretrain.load_collected(device='cpu')

#print(pretrain.mods)

from speechbrain.pretrained import EncoderClassifier
speakerrecog_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

#print(speakerrecog_model.mods)

# FINE TUNING PROCEDURE
from speechbrain.lobes.features import Fbank
import speechbrain as sb
import torch 

# Define fine-tuning procedure 
class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.signal

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid = batch.sID 


        
        # START HACKY FIX
        #predictions = torch.squeeze(predictions, 1)
        #print(predictions.shape)
        spkid = torch.unsqueeze(spkid, 1)
        #print(spkid.shape)
        # END HACKY FIX

        loss = self.hparams.compute_cost(predictions, spkid, lens) 

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
        # enable grad for all modules we want to fine-tune
        if stage == sb.Stage.TRAIN:
            for module in [self.modules.classifier]:
                for p in module.parameters():
                    p.requires_grad = True

            for module in [self.modules.embedding_model, self.modules.mean_var_norm, self.modules.compute_features]:
                for p in module.parameters():
                    p.requires_grad = False

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

from hyperpyyaml import load_hyperpyyaml

# This flag enables the inbuilt cudnn auto-tuner
torch.backends.cudnn.benchmark = True

# CLI:
hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[2:])

# Initialize ddp (useful only for multi-GPU DDP training)
sb.utils.distributed.ddp_init_group(run_opts)

# Load hyperparameters file with command-line overrides
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
)

modules = {
            "compute_features": speakerrecog_model.mods.compute_features,
            "embedding_model": speakerrecog_model.mods.embedding_model,
            "classifier": speakerrecog_model.mods.classifier,
            "mean_var_norm": speakerrecog_model.mods.mean_var_norm,
          }


# Brain class initialization
speaker_brain = SpeakerBrain(
    modules=modules, # using prior learned modules
    opt_class=hparams["opt_class"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
)

#train_data = SaveableDataLoader(dataset, batch_size=hparams["batch_size"], collate_fn=PaddedBatch)

# Training
speaker_brain.fit(
    speaker_brain.hparams.epoch_counter,
    dataset,
    train_loader_kwargs=hparams["dataloader_options"],
    valid_loader_kwargs=hparams["dataloader_options"],
)