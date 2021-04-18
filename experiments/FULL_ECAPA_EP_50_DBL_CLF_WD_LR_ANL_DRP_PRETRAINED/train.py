#!/usr/bin/env python3
import os
import sys
import csv
import torch
import logging
import torchaudio
import speechbrain as sb
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from custom_model import LID, HierarchicalClassifier, HierarchicalSoftmax, hierarchical_loss # noqa
from common_voice_lid_prepare import prepare_common_voice_for_lid, LANGUAGES # noqa

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


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_voice_for_lid` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'lang01': 0, 'lang02': 1, ..)
    language_family_encoder = sb.dataio.encoder.CategoricalEncoder()
    language_name_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("language_family", "language_name")
    @sb.utils.data_pipeline.provides("language_family", 
                                     "language_family_encoded",
                                     "language_name",
                                     "language_name_encoded")
    def label_pipeline(language_family, language_name):
        yield language_family
        language_family_encoded = language_family_encoder.encode_label_torch(language_family)
        yield language_family_encoded
        yield language_name
        language_name_encoded = language_name_encoder.encode_label_torch(language_name)
        yield language_name_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=os.path.join(hparams["save_folder"], dataset + '.csv'),
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "language_family_encoded", "language_name_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    language_family_encoder_file = os.path.join(hparams["save_folder"], 
                                                "language_family_encoder.txt")
    language_family_encoder.load_or_create(
        path=language_family_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="language_family"
    )

    language_name_encoder_file = os.path.join(hparams["save_folder"], "language_name_encoder.txt")
    language_name_encoder.load_or_create(
        path=language_name_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="language_name",
    )

    return datasets, language_family_encoder, language_name_encoder

def create_csv(predictions, csv_file):
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        # Write the header
        csv_writer.writerow(predictions.keys())

        for i in range(len(predictions['id'])):
            csv_writer.writerow(
                [
                    predictions['id'][i],
                    predictions['language_family'][i],
                    predictions['language_name'][i],
                ]
            )
        
        msg = "\tPredictions file created at %s" % (csv_file)
        logger.info(msg)

def make_hierarchical_mask(language_family_encoder, language_name_encoder, LANGUAGES):
    """This function creates a hierarchical mask for 2-level hierarchy

    Arguments
    ---------
    language_family_encoder : CategoricalEncoder
        Encoded parents
    language_name_encoder : CategoricalEncoder
        Encoded children
    LANGUAGES : dict
        Dictionary with parent-child relations

    Returns
    -------
    hierarchical_mask : tensor
        Tensor of shape [len(language_family_encoder), len(language_name_encoder)], 
        with 1. where parent-child connection exists and 0. othewise
    """
    hierarchical_mask = torch.zeros(
        len(language_family_encoder), 
        len(language_name_encoder)
    )

    for lang_name, fam_name in LANGUAGES.items():
        try:
            j = language_name_encoder.encode_label(label=lang_name, allow_unk=False)
        except KeyError:
            continue
        i = language_family_encoder.encode_label(label=fam_name, allow_unk=False)
        hierarchical_mask[i, j] = 1.

    return hierarchical_mask

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
            "max_duration": hparams.get("max_duration") , # TODO check if this works
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects "train", "dev", and "test" and label_encoders
    datasets, language_family_encoder, language_name_encoder = dataio_prep(hparams)

    # Check if retrained modules are required
    if hparams.get("pretrainer"):
        sb.utils.distributed.run_on_main(
            hparams["pretrainer"].collect_files
        )
        hparams["pretrainer"].load_collected(
            device=run_opts["device"]
        )

    # Initialize the Brain object to prepare for mask training.
    language_id_brain = LID(
        hierarchical_mask=make_hierarchical_mask(
            language_family_encoder=language_family_encoder, 
            language_name_encoder=language_name_encoder, 
            LANGUAGES=LANGUAGES
        ),
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

    # Running inference and writing results to files
    for dataset in hparams["predict_dataset"]:

        msg = "\tPredicting for %s dataset.." % (dataset)
        logger.info(msg)

        # Load the best checkpoint for predictions
        predictions = language_id_brain.predict(
            data_set=datasets[dataset],
            min_key="error",
            data_loader_kwargs=hparams["test_dataloader_options"],
        )

        
        # Decode predicted language families
        predictions['language_family'] = language_family_encoder.decode_ndim(
            predictions['language_family']
        )
        # Decode predicted language names
        predictions['language_name'] = language_name_encoder.decode_ndim(
            predictions['language_name']
        )

        # Output file with predictions
        predictions_filename = dataset + "_predictions.csv"
        csv_file = os.path.join(hparams["save_folder"], predictions_filename)

        # Write predicitons to the file
        create_csv(predictions, csv_file)
