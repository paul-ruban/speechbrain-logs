#!/usr/bin/env python3
import os
import sys
import csv
import torch
import logging
import torchaudio
import speechbrain as sb
from utils import create_prediction_csv # noqa
from tqdm.contrib import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from common_voice_lid_prepare import prepare_common_voice_for_lid, dataio_prep, LANGUAGES # noqa

"""Recipe for training a LID classifier system with CommonVoice.
# TODO
The system employs ...

To run this recipe, do the following:
> python train.py hparams/train.yaml

Authors
 * Mirco Ravanelli 2021
 * Pavlo Ruban 2021
 * Oleksandr Dymov 2021
"""

logger = logging.getLogger(__name__)

# Brain class for Language ID training
class LID(sb.Brain):
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
            if self.hparams.apply_env_corruption:
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if self.hparams.apply_augmentation:
                wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs) 
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_embeddings(self, feats, lens):
        # Compute embeddings
        embeddings = self.modules.embedding_model(feats, lens)
        # Normalize embeddings
        embeddings = self.modules.mean_var_norm_emb(
            x=embeddings, 
            lengths=torch.ones(embeddings.shape[0], device=embeddings.device)
        )

        return embeddings

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.compute_embeddings(feats, lens)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        
        predictions, lens = inputs

        targets = batch.language_name_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and self.hparams.apply_augmentation:
            targets = torch.cat([targets, targets], dim=0)

        loss = self.hparams.compute_cost(predictions, targets)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, targets, lens, reduction="batch"
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                batch.id, predictions, targets, lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

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

    def predict_batch(self, batch, topk=1):
        output, lens = self.compute_forward(batch, stage=sb.Stage.TEST)
        output = output.softmax(dim=-1)
        probas, idx = torch.topk(input=output, k=topk, dim=-1)

        return probas, idx
    
    def predict(
        self,
        data_set,
        max_key=None,
        min_key=None,
        progressbar=True,
        data_loader_kwargs={},
        topk=1
    ):

        if not isinstance(data_set, DataLoader):
            data_loader_kwargs["ckpt_prefix"] = None
            data_set = self.make_dataloader(
                data_set, sb.Stage.TEST, **data_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.modules.eval()
        
        # Create a dictionary with ids and predictions
        predictions = {
            'id' : [],
            **{f"{i+1}":[] for i in range(topk)}
        }

        with torch.no_grad():
            for batch in tqdm(data_set, dynamic_ncols=True, disable=not progressbar):
                # Get a list of batch predictions 
                probas, idx = self.predict_batch(batch, topk=topk)
                predictions['id'].extend(batch.id)
                for k in range(topk):
                    k_predictions = idx[:, :, k].squeeze(dim=-1).cpu()
                    predictions[f'{k+1}'].extend(k_predictions.tolist())

        return predictions

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if not set(hparams["predict_dataset"]).issubset({'train', 'dev', 'test'}):
        raise ValueError('Illegal predict_dataset parameter.')

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_common_voice_for_lid,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "max_duration": hparams.get("max_duration"),
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects "train", "dev", and "test" and label_encoders
    datasets, language_family_encoder, language_name_encoder = dataio_prep(hparams)

    # Check if retrained modules are required
    if hparams['use_pretrained_modules']:
        sb.utils.distributed.run_on_main(
            hparams["pretrainer"].collect_files
        )
        hparams["pretrainer"].load_collected(
            device=run_opts["device"]
        )

    # Initialize the Brain object to prepare for mask training.
    language_id_brain = LID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
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

    # Handle single dataset as input
    if not isinstance(hparams["predict_dataset"], list):
        hparams["predict_dataset"] = [hparams["predict_dataset"]]

    topk = min(hparams["topk"], hparams["n_languages"])

    # Running inference and writing results to files
    for dataset in hparams["predict_dataset"]:

        msg = "\tPredicting for %s dataset.." % (dataset)
        logger.info(msg)

        # Load the best checkpoint for predictions
        predictions = language_id_brain.predict(
            data_set=datasets[dataset],
            min_key="error",
            data_loader_kwargs=hparams["test_dataloader_options"],
            topk=topk
        )

        for k in range(1, topk+1):
            # Decode predicted language names
            predictions[f'{k}'] = language_name_encoder.decode_ndim(
                predictions[f'{k}']
            )

        # Output file with predictions
        predictions_filename = dataset + "_predictions.csv"
        csv_file = os.path.join(hparams["save_folder"], predictions_filename)

        # Write predicitons to the file
        create_prediction_csv(predictions, csv_file)