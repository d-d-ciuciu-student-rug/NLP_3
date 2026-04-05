# We had a rather long startup-time. Initially we thought it has to do with needing to minimize the imports.
#  It turned out to be related to OpenMP and the number of cores on the machine.
#  To solve the problem we had to set some environment variables `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.
#
# This is now incorporated in the `run.sh` script, which calls `python3` through the package manager `uv`,
#  while also enabling the profiling of the program, to figure out which function calls take up the longest
#  amount of CPU-time.
#
# As it stands, this program is actually CPU-bound.

from copy import deepcopy
import gc
import math
from random import seed as random_seed
from typing import Any

import pandas as pd
from numpy.random import seed as numpy_seed

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from download import download_data
from preprocess import (concatenate_title_and_description_into_merged,
                        align_labels_to_zero_index_and_rename,
                        split_training_data,
                        build_vocabulary,
                        encode,
                        )
from tuning import ParameterRange, ParameterSpace
from models import CNNTextClassifier, LSTMTextClassifier
from training import train_time_and_evaluate, evaluate, fit
from analysis import (get_misclassified_examples,
                      get_confusion_matrix,
                      compute_classification_metrics,
                      plot_learning_curves,
                      )


if __name__ == "__main__":

    def set_seed(seed: int) -> None:
        """
            Making sure we seed the PRNG of every library we are making use of,
             so that the initial state is the same for every run.
        """
        random_seed(seed)
        numpy_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def load_best_model(specific_model: tuple[dict[str, Any], dict[str, Any], float],
                        model_type: str,
                        device: torch.device
                        ) -> nn.Module:
        
        state_dict: dict[str, Any] = specific_model[1]
        init_params: dict[str, Any] = specific_model[2]
        
        if model_type == "CNNTextClassifier":
            model: nn.Module = CNNTextClassifier(**init_params)
        else:
            model: nn.Module = LSTMTextClassifier(**init_params)
        
        model.load_state_dict(state_dict)
        model.to(device)
        
        return model


    def pipeline():
        """
            Executes the complete pipeline, from retrieving the data, seeding the PRNGs,
             pre-processing the input, training the models while fine-tuning them,
             evaluating the models and logging some metrics to file (for later visualization).
        """

        print("Hardware selection.")

        # Device selection.
        if torch.cuda.is_available():
            device: torch.device = torch.device("cuda")
            print(f"\nUsing GPU: '{torch.cuda.get_device_name(0)}'.")
        else:
            device: torch.device = torch.device("cpu")
            print("\nUsing CPU.")

        # Determine how many worker threads the CPU-side can afford.
        num_worker_threads: int = 4

        print("\nDownloading dataset.")

        # Download the data from Hugging Face.
        data: tuple[pd.DataFrame, pd.DataFrame] = download_data()
        train_120k_samples: pd.DataFrame = data[0]
        test_7k_samples: pd.DataFrame = data[1]

        print(f"\n[120k, downloaded] DataFrame schema:\n{train_120k_samples.dtypes}")
        print(f"\n[7k, downloaded] DataFrame schema:\n{test_7k_samples.dtypes}")


        print("\nMerging 'title' and 'description' into a single 'input' vector.")

        # Concatenate the "title" and "description" vectors into a single vector.
        concatenate_title_and_description_into_merged(train_120k_samples, "title", "description", "input")
        concatenate_title_and_description_into_merged(test_7k_samples, "title", "description", "input")
 

        print("\nAligning labels to be zero-indexed, and alising/renaming the vector to 'output'.")

        # The labels are probably indexed starting at 1 (, instead of the desired 0).
        mapping_120k: dict[int, int] = align_labels_to_zero_index_and_rename(data = train_120k_samples,
                                                                             label_alias = "label",
                                                                             label_new_alias = "output")

        mapping_7k: dict[int, int] = align_labels_to_zero_index_and_rename(data = test_7k_samples, 
                                                                           label_alias = "label",
                                                                           label_new_alias = "output")

        print( "\nLabel mappings:"
              f"\n\t120k: {mapping_120k}"
              f"\n\t7k: {mapping_7k}")


        print(f"\n[120k, final] DataFrame schema:\n{train_120k_samples.dtypes}\n")
        train_120k_samples.info()
        print(f"\nHead:\n{train_120k_samples.head(5)}")
        print(f"\nTail:\n{train_120k_samples.tail(5)}")

        print(f"\n[7k, final] DataFrame schema:\n{test_7k_samples.dtypes}\n")
        test_7k_samples.info()
        print(f"\nHead:\n{test_7k_samples.head(5)}")
        print(f"\nTail:\n{test_7k_samples.tail(5)}")


        REPRODUCIBILITY_PRNG_SEED: Final[int] = 0
        print(f"\nSeeding PRNGs: {REPRODUCIBILITY_PRNG_SEED}.")

        # Setting a reproducible PRNG state.
        set_seed(REPRODUCIBILITY_PRNG_SEED)


        print("\nSplitting the training dataset (80% training, 10% validation, 10% testing).")

        # Then, using a fixed seed, split the training data (120k samples)
        #  into (80% train, 10% development/validation, and 10% test).
        #
        # We will be using the remaining 7k test samples at the very end.
        #
        # Stratified sampling is used to ensure a more even representation of
        #  all possible outputs (labels/"classes").
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = split_training_data(data = train_120k_samples,
                                                                                      stratify_column = "output",
                                                                                      seed = REPRODUCIBILITY_PRNG_SEED)

        # These are our data-frames.
        train_df: pd.DataFrame = splits[0]
        validate_df: pd.DataFrame = splits[1]
        initial_test_df: pd.DataFrame = splits[2]
        final_test_df: pd.DataFrame = test_7k_samples


        print("\nInitializing two lists that will keep all of the metrics for later analysis and visualization.")

        # Columns: "model", "identifier", "epoch",
        #          "loss", "accuracy", "precision", "recall", "f1"
        epoch_metrics: list[dict[str, Any]] = []

        # Columns: "model", "identifier", "total_training_time_in_seconds",
        #          "validate_accuracy", "validate_precision", "validate_recall", "validate_f1",
        #          "test_accuracy", "test_precision", "test_recall", "test_f1"])
        model_final_metrics: list[dict[str, Any]] = []

        print('\n[epoch metrics]\n\t"model", "identifier", "epoch",'
              '\n\t"loss", "accuracy", "precision", "recall", "f1"')

        print('\n[model final metrics]\n\t"model", "identifier", "total_training_time_in_seconds"'
              '\n\t"validate_accuracy", "validate_precision", "validate_recall", "validate_f1"'
              '\n\t"test_accuracy", "test_precision", "test_recall", "test_f1"')


        print("\nInitializing a `dict[str, tuple[Any, Any]]` that will keep the weights of the best model"
              " (for each model type), together with the vocabulary used for encoding the input text.")

        # The tuples are meant to keep: (f1-score, model-name, ...all of the hyper-parameter `dict`'s...).
        models_tracker: dict[str, list[tuple[float, str, ...]]] = {
            "CNNTextClassifier": [],
            "LSTMTextClassifier": [],
        }

        print(f"\nModels tracker:\n{models_tracker}")


        print("\nDefining the parameter space for the (tokenization and encoding) phase.")

        # Defining a parameter space for the tokenization-encoding phase.
        TOKENIZATION_ENCODING_PARAMETER_SPACE = ParameterSpace({
            "to_lower_and_regex_substitution_rules": [
                            # Make everything lower-case, only allow {alpha, digit, space}.
                            (True, [(r"[^a-z0-9 ]", "")]),

#                            # Both {lower, upper}-case, only allow {alpha, digit, space},
#                            #  compact multi-space sequences into one space.
#                            (False, [(r"[^a-zA-Z0-9 ]", ""), (r"\s+", " ")]),
                           ],

            # The filter-lambda can technically accomodate string-pattern filtering as well,
            #  but we only use it for filtering based on a minimum frequency.
            "token_filter_condition": [(lambda token, frequency: frequency >= 2, "(lambda token, frequency: frequency >= 2)")],

            # It is preferable that these values be multiples of 32, because the L40s GPU also has 32 warps per core.

            # Note: ablation test -> controlling the vocabulary size.
            "maximum_vocabulary_size": [64, 4096],

            #"target_input_list_length": [64, 128, 256],
            "target_input_list_length": [256],

            # We also need a few special tokens, for handling:
            #
            #   i) when the embedded representation of an input text (converted into a list
            #       of tokens, themselves mapped to integer values) would be too short. For
            #       this we use the "padding token" to fill in the list.
            #
            #   ii) when the input contains tokens that are not recognized (and thus do not
            #        have their own integer mapping). In this case, the "unknown token" is
            #        a universal integer to be used.
            "padding_token": [("<PAD>", 0)],
            "unknown_token": [("<UNK>", 1)],

            # Having looked at the data, we found 4 classes. But, we switched to determining
            #  this dynamically.
            "num_classes": [len(mapping_120k)],
        })

        print("\nDefining the parameter space for the (training) phase.")

        TRAINING_PARAMETER_SPACE: ParameterSpace = ParameterSpace({
            # When computing the gradient descent, more than one sample gets averaged in one go.
            #  Averaging more samples might make it harder for the optimizer to escape local minima.
            "batch_size": [512],

            "learning_rate":  [1e-3],
            "maximum_epochs": [12],
            "patience": [3],
            "clip_gradient_norm": [1.0],
        })

        print("\nDefining the parameter space for the (model-specific parameter tuning) for the CNN model.")

        # Defining the model-tuning ranges for the Convolutional Neural Network.

        CNN_PARAMETER_SPACE: ParameterSpace = ParameterSpace({
            "embed_dim": ParameterRange([64]),
            "num_filters": ParameterRange([64]),
            "kernel_sizes": ParameterRange([(3, 4, 5)]),
            "dropout": ParameterRange([0.3]),
        })

        print("\nDefining the parameter space for the (model-specific parameter tuning) for the LSTM model.")

        # Defining the model-tuning ranges for the Long Short-Term Memory recurrent network.

        LSTM_PARAMETER_SPACE: ParameterSpace = ParameterSpace({
            "embed_dim": ParameterRange([64]),
            "hidden_dim": ParameterRange([64]),
            "num_layers": ParameterRange([2]),
            "dropout": ParameterRange([0.3]),
            "bidirectional": ParameterRange([False]),
        })

        # We tried to reduce the looping-structure to a nicer form.
        #  We train using various combinations of maximum string length
        #   and minimum-frequency filtering for vocabulary building.
        #
        #  We also try out various model-specific parameters, based on a "parameter
        #   space" defined for each model.

        for tokenization_parameters in TOKENIZATION_ENCODING_PARAMETER_SPACE:

            print(f"\n---- Tokenizing with parameters: {tokenization_parameters} ----")

            
            print("\n\tBuilding the vocabulary.")

            # Build vocabulary based on the current point in tokenization's parameter space.
            #  Only use the training sub-set for building the vocabulary, to avoid "leakage"
            #  between the sub-sets, and prevent overfitting.

            vocabulary: dict[str, int] = build_vocabulary(text_vector = train_df["input"],
                                                          rules = tokenization_parameters["to_lower_and_regex_substitution_rules"],
                                                          padding_token = tokenization_parameters["padding_token"],
                                                          unknown_token = tokenization_parameters["unknown_token"],
                                                          token_filter_condition = tokenization_parameters["token_filter_condition"][0],
                                                          maximum_vocabulary_size = tokenization_parameters["maximum_vocabulary_size"])

            print(f"\n\tObtained vocabulary:\n{vocabulary}")


            # Compute encoding of the "input" vector based on the vocabulary.
            #  The validation and test sub-sets will also make use of the same vocabulary when encoded.

            print(f"\n\tEncoding the \"input\" vectors (training sub-set)."
                  f"\n\t\tMaximum vocabulary size: {tokenization_parameters['maximum_vocabulary_size']}."
                  f"\n\t\tTarget input length: {tokenization_parameters['target_input_list_length']}.")

            train_results: tuple[list[list[int]], list[int]] = encode(text_vector = train_df["input"],
                                                                      rules = tokenization_parameters["to_lower_and_regex_substitution_rules"],
                                                                      padding_token = tokenization_parameters["padding_token"],
                                                                      unknown_token = tokenization_parameters["unknown_token"],
                                                                      vocabulary = vocabulary,
                                                                      target_input_list_length = tokenization_parameters["target_input_list_length"])
            train_encodings: list[list[int]] = train_results[0]
            train_non_padded_lengths: list[int] = train_results[1]

            print( "\n\t[DEBUG] (train sub-set) Encodings:"
                  f"\n\t{train_encodings[0]}\n\t{train_encodings[1]}\n\t...")
            print( "\n\t[DEBUG] (train sub-set) Non-padded lengths:"
                  f"\n\t[{train_non_padded_lengths[0]}, {train_non_padded_lengths[1]}, ...]")


            print(f"\n\tEncoding the \"input\" vectors (validate sub-set).")

            validate_results: tuple[list[list[int]], list[int]] = encode(text_vector = validate_df["input"],
                                                                         rules = tokenization_parameters["to_lower_and_regex_substitution_rules"],
                                                                         padding_token = tokenization_parameters["padding_token"],
                                                                         unknown_token = tokenization_parameters["unknown_token"],
                                                                         vocabulary = vocabulary,
                                                                         target_input_list_length = tokenization_parameters["target_input_list_length"])
            validate_encodings: list[list[int]] = validate_results[0]
            validate_non_padded_lengths: list[int] = validate_results[1]

            print( "\n\t[DEBUG] (validate sub-set) Encodings:"
                  f"\n\t{validate_encodings[0]}\n\t{validate_encodings[1]}\n\t...")
            print( "\n\t[DEBUG] (validate sub-set) Non-padded lengths:"
                  f"\n\t[{validate_non_padded_lengths[0]}, {validate_non_padded_lengths[1]}, ...]")


            print(f"\n\tEncoding the \"input\" vectors (initial-test sub-set)."
                   "\n\t\tThe final-test sub-set will be used only on the best model we can get through fine-tuning, at the very end.")

            initial_test_results: tuple[list[list[int]], list[int]] = encode(text_vector = initial_test_df["input"],
                                                                             rules = tokenization_parameters["to_lower_and_regex_substitution_rules"],
                                                                             padding_token = tokenization_parameters["padding_token"],
                                                                             unknown_token = tokenization_parameters["unknown_token"],
                                                                             vocabulary = vocabulary,
                                                                             target_input_list_length = tokenization_parameters["target_input_list_length"])
            initial_test_encodings: list[list[int]] = initial_test_results[0]
            initial_test_non_padded_lengths: list[int] = initial_test_results[1]

            print( "\n\t[DEBUG] (initial-test sub-set) Encodings:"
                  f"\n\t{initial_test_encodings[0]}\n\t{initial_test_encodings[1]}\n\t...")
            print( "\n\t[DEBUG] (initial-test sub-set) Non-padded lengths:"
                  f"\n\t[{initial_test_non_padded_lengths[0]}, {initial_test_non_padded_lengths[1]}, ...]")


            # Now it's time to convert the (input encoding, outputs, non-padded lengths) lists of integers into tensors,
            #  then wrap them into a `TensorDataset`, which will further be batch-loaded onto the GPU by `DataLoader`.
            #
            # Because all of the values used are `int`, the "dtype" will be `torch.long`:
            #
            #   dtype = torch.long
            #
            #  which corresponds to a 64 bit integer.

            print("\n\tTransforming the encodings (inputs, outputs, non-padded lengths) into `torch.tensor`.")

            train_input_tensor: Tensor = torch.tensor(train_encodings, dtype = torch.long)
            train_output_tensor: Tensor = torch.tensor(train_df["output"].values, dtype = torch.long)
            train_non_padded_lengths_tensor: Tensor = torch.tensor(train_non_padded_lengths, dtype = torch.long)

            print(f"\n\tTransformed the train sub-set into `torch.tensor`'s (for input, output, non-padded lengths):"
                  f"\n\tInput:\n{train_input_tensor}"
                  f"\n\tOutput:\n{train_output_tensor}"
                  f"\n\tNon-padded lengths:\n{train_non_padded_lengths_tensor}")


            validate_input_tensor: Tensor = torch.tensor(validate_encodings, dtype = torch.long)
            validate_output_tensor: Tensor = torch.tensor(validate_df["output"].values, dtype = torch.long)
            validate_non_padded_lengths_tensor: Tensor = torch.tensor(validate_non_padded_lengths, dtype = torch.long)

            print(f"\n\tTransformed the validate sub-set into `torch.tensor`'s (for input, output, non-padded lengths).")


            initial_test_input_tensor: Tensor = torch.tensor(initial_test_encodings, dtype = torch.long)
            initial_test_output_tensor: Tensor = torch.tensor(initial_test_df["output"].values, dtype = torch.long)
            initial_test_non_padded_lengths_tensor: Tensor = torch.tensor(initial_test_non_padded_lengths, dtype = torch.long)

            print(f"\n\tTransformed the initial-test sub-set into `torch.tensor`'s (for input, output, non-padded lengths).")


            # Wrapping the tensors into `TensorDataset`. This type is not an owner of the data, meaning that it doesn't make
            #  copies to it. Instead, it is a "meta-data wrapper", using references/pointers to the original data.
            #
            # `TensorDataset` provides an interface for accessing multiple `torch.tensor` together.

            print("\n\tPackaging (training, validate, initial-test) `torch.tensor`'s into `TensorDataset`'s.")

            train_td: TensorDataset = TensorDataset(train_input_tensor, train_output_tensor, train_non_padded_lengths_tensor)
            validate_td: TensorDataset = TensorDataset(validate_input_tensor, validate_output_tensor, validate_non_padded_lengths_tensor)
            initial_test_td: TensorDataset = TensorDataset(initial_test_input_tensor, initial_test_output_tensor, initial_test_non_padded_lengths_tensor)

            # Model-specific tuning and training sub-loops.
            #
            # Now a particular "problem" with this setup. It would be desirable to have "adaptive batch-sizes",
            #  based on feedback from the optimizer, but the current setup doesn't nicely support that.
            #
            # Although, it is actually possible to use a function to determine the next batch-size,
            #  instead of using hard-coded values. Maybe in a future iteration.

            for training_parameters in TRAINING_PARAMETER_SPACE:

                print(f"\n\t---- Training with parameters: {training_parameters} ----")

                print("\n\t\tWrapping `TensorDataset`'s into `DataLoader`'s, for batched loading onto the GPU.")

                # We still need to tell `DataLoader` to shuffle the order in which it presents the training sub-set,
                #  whereas the other sub-sets should be presented in unmodified order.

                train_loader: DataLoader = DataLoader(train_td, batch_size = training_parameters["batch_size"],
                                                      shuffle = True, num_workers = num_worker_threads, pin_memory = True, drop_last = True)
                validate_loader: DataLoader = DataLoader(validate_td, batch_size = training_parameters["batch_size"],
                                                         shuffle = False, num_workers = num_worker_threads, pin_memory = True, drop_last = True)
                initial_test_loader: DataLoader = DataLoader(initial_test_td, batch_size = training_parameters["batch_size"],
                                                             shuffle = False, num_workers = num_worker_threads, pin_memory = True, drop_last = True)


                for model_parameters_CNN in CNN_PARAMETER_SPACE:

                    print(f"\n\t\t---- Training CNN model with model parameters: {model_parameters_CNN} ----")

                    model_CNN: CNNTextClassifier = CNNTextClassifier(vocab_size = len(vocabulary),
                                                                     embed_dim = model_parameters_CNN["embed_dim"],
                                                                     num_filters = model_parameters_CNN["num_filters"],
                                                                     kernel_sizes = model_parameters_CNN["kernel_sizes"],
                                                                     dropout = model_parameters_CNN["dropout"],
                                                                     pad_idx = tokenization_parameters["padding_token"][1],
                                                                     num_classes = tokenization_parameters["num_classes"],
                                                                     ).to(device)

                    print(f"\n\t\t\tNumber of trainable parameters: {model_CNN.count_parameters()}.")


                    print("\n\t\t\tStarting training (and timer).")
                    results_CNN: dict[str, Any] = train_time_and_evaluate(device = device,
                                                                          model = model_CNN,
                                                                          train_loader = train_loader,
                                                                          validate_loader = validate_loader,
                                                                          test_loader = initial_test_loader,
                                                                          learning_rate = training_parameters["learning_rate"],
                                                                          maximum_epochs = training_parameters["maximum_epochs"],
                                                                          patience = training_parameters["patience"],
                                                                          clip_gradient_norm = training_parameters["clip_gradient_norm"]
                                                                          )

                    print(f"\n\t\t\tFinished training in {results_CNN['total_time_in_seconds']} seconds."
                          f"\n\t\t\taccuracy: {results_CNN['test']['accuracy']},"
                          f" precision: {results_CNN['test']['precision']},"
                          f" recall: {results_CNN['test']['recall']},"
                          f" f1: {results_CNN['test']['f1']}."
                          )


                    print("\n\t\t\tLogging metrics.")

                    index_model: int = len(models_tracker["CNNTextClassifier"])

                    # Log all of the metrics
                    fit_history: list[dict[str, Any]] = results_CNN["history"]
                    for entry in fit_history:
                        epoch_metrics.append({"model": "CNNTextClassifier", "identifier": index_model, "epoch": entry["epoch"],
                                              "loss": entry["loss"], "accuracy": entry["accuracy"], "precision": entry["precision"],
                                               "recall": entry["recall"], "f1": entry["f1"]})

                    model_final_metrics.append({
                        "model": "CNNTextClassifier", "identifier": index_model,
                        "total_training_time_in_seconds": results_CNN["total_time_in_seconds"],
                        "validate_accuracy": results_CNN["validate"]["accuracy"], "validate_precision": results_CNN["validate"]["precision"],
                        "validate_recall": results_CNN["validate"]["recall"], "validate_f1": results_CNN["validate"]["f1"],
                        "test_accuracy": results_CNN["test"]["accuracy"], "test_precision": results_CNN["test"]["precision"],
                        "test_recall": results_CNN["test"]["recall"], "test_f1": results_CNN["test"]["f1"]
                    })


                    # Note: initially we were only keeping the best, but because of the ablation experiment,
                    #  we will instead append to the ::models_tracker.

                    current_f1_score: float = results_CNN["test"]["f1"]
                    print(f"\n\t\t\tCurrent best CNN model has changed, f1-score = {current_f1_score}.")

                    # Make a copy of the model's state (including the weights), copy it CPU-side
                    #  and reference it in the ::model_tracker, together with all of the hyper-parameters.

                    best_weights: dict[str, Any] = deepcopy(model_CNN.state_dict())
                    for key in best_weights:
                        best_weights[key] = best_weights[key].cpu()

                    models_tracker["CNNTextClassifier"].append((
                        current_f1_score,
                        best_weights,
                        {"vocab_size": len(vocabulary),
                         "embed_dim": model_parameters_CNN["embed_dim"],
                         "num_filters": model_parameters_CNN["num_filters"],
                         "kernel_sizes": model_parameters_CNN["kernel_sizes"],
                         "dropout": model_parameters_CNN["dropout"],
                         "pad_idx": tokenization_parameters["padding_token"][1],
                         "num_classes": tokenization_parameters["num_classes"],
                        },
                        deepcopy(vocabulary),
                        model_parameters_CNN,
                        training_parameters,
                        tokenization_parameters,
                    ))

                    # Explicitly freeing up memory, particularly from the GPU-side.
                    #  It should also work without being explicit.

                    del model_CNN
                    gc.collect()
                    torch.cuda.empty_cache()


                for model_parameters_LSTM in LSTM_PARAMETER_SPACE:

                    print(f"\n\t\t---- Training LSTM model with model parameters: {model_parameters_LSTM} ----")

                    model_LSTM = LSTMTextClassifier(vocab_size = len(vocabulary),
                                                    embed_dim = model_parameters_LSTM["embed_dim"],
                                                    hidden_dim = model_parameters_LSTM["hidden_dim"],
                                                    num_layers = model_parameters_LSTM["num_layers"],
                                                    dropout = model_parameters_LSTM["dropout"],
                                                    pad_idx = tokenization_parameters["padding_token"][1],
                                                    num_classes = tokenization_parameters["num_classes"],
                                                    bidirectional = model_parameters_LSTM["bidirectional"],
                                                    ).to(device)

                    print(f"\n\t\t\tNumber of trainable parameters: {model_LSTM.count_parameters()}.")


                    print("\n\t\t\tStarting training (and timer).")
                    results_LSTM: dict[str, Any] = train_time_and_evaluate(device = device,
                                                                           model = model_LSTM,
                                                                           train_loader = train_loader,
                                                                           validate_loader = validate_loader,
                                                                           test_loader = initial_test_loader,
                                                                           learning_rate = training_parameters["learning_rate"],
                                                                           maximum_epochs = training_parameters["maximum_epochs"],
                                                                           patience = training_parameters["patience"],
                                                                           clip_gradient_norm = training_parameters["clip_gradient_norm"]
                                                                           )

                    print(f"\n\t\t\tFinished training in {results_LSTM['total_time_in_seconds']} seconds."
                          f"\n\t\t\taccuracy: {results_LSTM['test']['accuracy']},"
                          f" precision: {results_LSTM['test']['precision']},"
                          f" recall: {results_LSTM['test']['recall']},"
                          f" f1: {results_LSTM['test']['f1']}."
                          )


                    print("\n\t\t\tLogging metrics.")

                    index_model: int = len(models_tracker["LSTMTextClassifier"])

                    # Log all of the metrics
                    fit_history: list[dict[str, Any]] = results_LSTM["history"]
                    for entry in fit_history:
                        epoch_metrics.append({"model": "LSTMTextClassifier", "identifier": index_model, "epoch": entry["epoch"],
                                              "loss": entry["loss"], "accuracy": entry["accuracy"], "precision": entry["precision"],
                                               "recall": entry["recall"], "f1": entry["f1"]})

                    model_final_metrics.append({
                        "model": "LSTMTextClassifier", "identifier": index_model,
                        "total_training_time_in_seconds": results_LSTM["total_time_in_seconds"],
                        "validate_accuracy": results_LSTM["validate"]["accuracy"], "validate_precision": results_LSTM["validate"]["precision"],
                        "validate_recall": results_LSTM["validate"]["recall"], "validate_f1": results_LSTM["validate"]["f1"],
                        "test_accuracy": results_LSTM["test"]["accuracy"], "test_precision": results_LSTM["test"]["precision"],
                        "test_recall": results_LSTM["test"]["recall"], "test_f1": results_LSTM["test"]["f1"]
                    })


                    # Note: initially we were only keeping the best, but because of the ablation experiment,
                    #  we will instead append to the ::models_tracker.

                    current_f1_score: float = results_LSTM["test"]["f1"]
                    print(f"\n\t\t\tCurrent best LSTM model has changed, f1-score = {current_f1_score}.")

                    # Make a copy of the model's state (including the weights), copy it CPU-side
                    #  and reference it in the ::model_tracker, together with all of the hyper-parameters.

                    best_weights: dict[str, Any] = deepcopy(model_LSTM.state_dict())
                    for key in best_weights:
                        best_weights[key] = best_weights[key].cpu()

                    models_tracker["LSTMTextClassifier"].append((
                        current_f1_score,
                        best_weights,
                        {"vocab_size": len(vocabulary),
                         "embed_dim": model_parameters_LSTM["embed_dim"],
                         "hidden_dim": model_parameters_LSTM["hidden_dim"],
                         "num_layers": model_parameters_LSTM["num_layers"],
                         "dropout": model_parameters_LSTM["dropout"],
                         "pad_idx": tokenization_parameters["padding_token"][1],
                         "num_classes": tokenization_parameters["num_classes"],
                         "bidirectional": model_parameters_LSTM["bidirectional"],
                        },
                        deepcopy(vocabulary),
                        model_parameters_LSTM,
                        training_parameters,
                        tokenization_parameters,
                    ))

                    del model_LSTM
                    gc.collect()
                    torch.cuda.empty_cache()

        # Error analysis
        #  First, we need the inverse map of how the labels have been mapped at the very begginning,
        #  when we were pre-processing the data. 
        inverse_mapping_7k: dict[int, int] = {v: k for k, v in mapping_7k.items()}

        # Now traverse all models, and run them against the ::final_test_df (7k samples).
        for tracked_model in models_tracker.keys():
            for index_specific_model in range(len(models_tracker[tracked_model])):
                specific_model = models_tracker[tracked_model][index_specific_model]


                print(f"\nAnalysis for model {tracked_model}.")

                model: nn.Module = load_best_model(specific_model, tracked_model, device)
                tokenizer_rules = specific_model[6]["to_lower_and_regex_substitution_rules"]
                padding_token = specific_model[6]["padding_token"]
                unknown_token = specific_model[6]["unknown_token"]
                vocabulary = specific_model[3]
                target_input_list_length = specific_model[6]["target_input_list_length"]
                num_classes = specific_model[6]["num_classes"]

                # Tuple format: (expected label, predicted label, associated input text).
                errors: list[tuple[int, int, str]] = get_misclassified_examples(device = device,
                                                                                model = model,
                                                                                data = final_test_df,
                                                                                maximum_error_samples = 20,
                                                                                tokenizer_rules = tokenizer_rules,
                                                                                padding_token = padding_token,
                                                                                unknown_token = unknown_token,
                                                                                vocabulary = vocabulary,
                                                                                target_input_list_length = target_input_list_length
                                                                                )
                # Mapping labels according to inverse mapping.
                errors = [(inverse_mapping_7k[error[0]], inverse_mapping_7k[error[1]], error[2]) for error in errors]

                print(f"{tracked_model}[{index_specific_model}] errors:")
                for error in errors:
                    print(f"\t{error}")


                confusion_matrix: torch.Tensor = get_confusion_matrix(device = device,
                                                                      model = model,
                                                                      data = final_test_df,
                                                                      tokenizer_rules = tokenizer_rules,
                                                                      padding_token = padding_token,
                                                                      unknown_token = unknown_token,
                                                                      vocabulary = vocabulary,
                                                                      target_input_list_length = target_input_list_length,
                                                                      num_classes = num_classes
                                                                      )

                # We need to return to the original labels, but the confusion matrix is computed as a `torch.Tensor`.
                #  We turn the `torch.Tensor` into a `pd.DataFrame` and assign the original labels.

                class_names: list[str] = [str(inverse_mapping_7k[i]) for i in range(num_classes)]

                confusion_df: pd.DataFrame = pd.DataFrame(confusion_matrix.cpu().numpy(), 
                                                          index = [f"True: {name}" for name in class_names], 
                                                          columns=[f"Pred: {name}" for name in class_names])

                print(f"\nConfusion matrix for model {tracked_model}[{index_specific_model}]:\n{confusion_df}")


                per_label_confusion: pd.DataFrame = compute_classification_metrics(confusion_matrix)
                
                print(f"\nPer-label confusion:\n{per_label_confusion}")


                saved_at: str = plot_learning_curves(epoch_metrics,
                                                     tracked_model,
                                                     index_specific_model,
                                                     key = "accuracy",
                                                     output_dir = "plots")

                print(f"Saved rendered learning curve plot to file: {saved_at}.")


        for entry in model_final_metrics:
            model_name = entry["model"]
            identifier = entry["identifier"]
            test_acc = entry["test_accuracy"]
            test_prec = entry["test_precision"]
            test_rec = entry["test_recall"]
            test_f1 = entry["test_f1"]

            print(f"\nModel: {model_name} (instance {identifier})")
            print(f"  Test Accuracy : {test_acc:.4f}")
            print(f"  Test Precision: {test_prec:.4f}")
            print(f"  Test Recall   : {test_rec:.4f}")
            print(f"  Test F1-score : {test_f1:.4f}")


    def main():
        pipeline()

    main()
