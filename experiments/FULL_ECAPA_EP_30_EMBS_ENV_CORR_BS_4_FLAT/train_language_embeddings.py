#!/usr/bin/python3
"""Recipe for training langauge embeddings using the CommonVoice.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import random
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import speechbrain as sb
from train import dataio_prep # noqa
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import download_file
from speechbrain.lobes.models.ECAPA_TDNN import Classifier
from common_voice_lid_prepare import prepare_common_voice_for_lid # noqa

class DoubleClassifier(nn.Module):
    """Two independent classifiers for languages and families
    """

    def __init__(self, n_lang_families=24, n_languages=45, *args, **kwargs):
        super().__init__()

        # Classifier for language families
        self.language_family_clf = Classifier(out_neurons=n_lang_families, *args, **kwargs)
        # Classifier for language families
        self.language_name_clf = Classifier(out_neurons=n_languages, *args, **kwargs)
        
    def forward(self, inputs):
        language_family = self.language_family_clf(inputs)

        language_name = self.language_name_clf(inputs)
        

        return language_family, language_name

class FlatClassifier(nn.Module):
    """Classifier with output layer with flattened hierarchical levels
    """
    def __init__(self, nodes_per_level=[], *args, **kwargs):
        super().__init__()
        if nodes_per_level[0] != 0:
            nodes_per_level.insert(0, 0)
        self.nodes_per_level = np.cumsum(nodes_per_level, dtype=int)
        out_neurons = sum(nodes_per_level)

        self.classifier = Classifier(out_neurons=out_neurons, *args, **kwargs)
        
    def forward(self, inputs):
        # Forward pass
        outputs = self.classifier(inputs)

        predictions = tuple(
            outputs[:, :, self.nodes_per_level[i]:self.nodes_per_level[i+1]] 
            for i in range(len(self.nodes_per_level) - 1)
        )

        return predictions



class LIDBrain(sb.core.Brain):
    """Class for language embedding training"
    """

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs) 
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + language classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Compute features
        feats, lens = self.prepare_features(batch.sig, stage)

        # Embeddings
        embeddings = self.modules.embedding_model(feats)

        # Normalize embeddings
        embeddings = self.modules.mean_var_norm_emb(
            x=embeddings, 
            lengths=torch.ones(embeddings.shape[0], device=embeddings.device)
        )
        
        # Classify
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        # lang_name_pred, lens = inputs
        predictions, lens = inputs
        targets = [
            batch.language_family_encoded.data,
            batch.language_name_encoded.data
        ]

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            targets = [
                torch.cat([targets[i], targets[i]], dim=0)
                for i in range(len(targets))
            ]
            lens = torch.cat([lens, lens], dim=0)

        if isinstance(predictions, tuple):
            loss = sum(
                self.hparams.compute_cost(predictions[i], targets[i]) 
                for i in range(len(predictions))
            )
            
        else:
            loss = self.hparams.compute_cost(predictions, targets[-1])
            predictions = [predictions]

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions[-1], targets[-1], lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, 
                min_keys=["error"]
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # # Download verification list (to exlude verification sentences from train)
    # veri_file_path = os.path.join(
    #     hparams["save_folder"], os.path.basename(hparams["verification_file"])
    # )
    # download_file(hparams["verification_file"], veri_file_path)

    # # Dataset prep (parsing VoxCeleb and annotation into csv files)
    # from voxceleb_prepare import prepare_voxceleb  # noqa

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_common_voice_for_lid,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "max_duration": hparams.get("max_duration") , # TODO check if this works
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, language_family_encoder, language_name_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if retrained modules are required
    if hparams.get("pretrainer"):
        sb.utils.distributed.run_on_main(
            hparams["pretrainer"].collect_files
        )
        hparams["pretrainer"].load_collected(
            device=run_opts["device"]
        )


    # Brain class initialization
    language_id_brain = LIDBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    language_id_brain.fit(
        epoch_counter=language_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = language_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )