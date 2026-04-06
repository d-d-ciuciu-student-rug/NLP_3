# uv run python3 main.py
#
# OR
#
# ./run.sh

from copy import deepcopy
import gc
import math
import random
from random import seed as random_seed
from time import perf_counter
from typing import Any

import pandas as pd
import numpy as np
from numpy.random import seed as numpy_seed

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


from download import download_data
from preprocess import (concatenate_title_and_description_into_merged,
                        align_labels_to_zero_index_and_rename,
                        split_training_data,
                        build_vocabulary,
                        encode,
                        )
from tuning import ParameterRange, ParameterSpace
from models import CNNTextClassifier, LSTMTextClassifier
from training import (train_time_and_evaluate,
                      evaluate,
                      train_CNN,
                      train_LSTM,
                      )
from analysis import (get_misclassified_examples_CNN_LSTM,
                      get_misclassified_examples_transformer,
                      get_confusion_matrix,
                      compute_classification_metrics,
                      plot_learning_curves,
                      )


if __name__ == "__main__":

    def get_device() -> torch.device:
        print("Hardware selection.")

        # Device selection.
        if torch.cuda.is_available():
            device: torch.device = torch.device("cuda")
            print(f"\nUsing GPU: '{torch.cuda.get_device_name(0)}'.")
        else:
            device: torch.device = torch.device("cpu")
            print("\nUsing CPU.")

        return device


    def set_seed(seed: int) -> None:
        """
            Making sure we seed the PRNG of every library we are making use of,
             so that the initial state is the same for every run.
        """
        random_seed(seed)
        numpy_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    downloaded: bool = False
    cached_seed: int = 0
    train_120k_samples: pd.DataFrame = None
    test_7k_samples: pd.DataFrame = None
    train_df: pd.DataFrame = None
    validate_df: pd.DataFrame = None
    initial_test_df: pd.DataFrame = None
    final_test_df: pd.DataFrame = None
    mapping_120k: dict[int, int] = None
    mapping_7k: dict[int, int] = None

    def get_preprocessed_dataset(seed: int) -> tuple[pd.DataFrame, pd.DataFrame,
                                                     pd.DataFrame, pd.DataFrame]:
        global downloaded
        global cached_seed
        global train_120k_samples
        global test_7k_samples
        global train_df
        global validate_df
        global initial_test_df
        global final_test_df
        global mapping_120k
        global mapping_7k

        if not downloaded:
            print("\nDownloading dataset.")

            # Download the data from Hugging Face.
            data: tuple[pd.DataFrame, pd.DataFrame] = download_data()
            train_120k_samples = data[0]
            test_7k_samples = data[1]

            print(f"\n[120k, downloaded] DataFrame schema:\n{train_120k_samples.dtypes}")
            print(f"\n[7k, downloaded] DataFrame schema:\n{test_7k_samples.dtypes}")


            print("\nMerging 'title' and 'description' into a single 'input' vector.")

            # Concatenate the "title" and "description" vectors into a single vector.
            concatenate_title_and_description_into_merged(train_120k_samples, "title", "description", "input")
            concatenate_title_and_description_into_merged(test_7k_samples, "title", "description", "input")
     

            print("\nAligning labels to be zero-indexed, and alising/renaming the vector to 'output'.")

            # The labels are probably indexed starting at 1 (, instead of the desired 0).
            mapping_120k = align_labels_to_zero_index_and_rename(data = train_120k_samples,
                                                                 label_alias = "label",
                                                                 label_new_alias = "output")

            mapping_7k = align_labels_to_zero_index_and_rename(data = test_7k_samples, 
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


        if (not downloaded or 
            seed != cached_seed
        ):
        
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
                                                                                          seed = seed)

            # These are our data-frames.
            train_df = splits[0]
            validate_df = splits[1]
            initial_test_df = splits[2]
            final_test_df = test_7k_samples

            downloaded = True
            cached_seed = seed

        print("\nRetrieving pre-processed dataset from cache.")

        return (train_df, validate_df, initial_test_df, final_test_df, mapping_120k, mapping_7k)


    def build_parameter_space_LSTM_and_CNN() -> tuple[int, ParameterSpace, ParameterSpace,
                                                           ParameterSpace, ParameterSpace]:
        # Determine how many worker threads the CPU-side can afford.
        num_worker_threads: int = 0
        print(f"Using {num_worker_threads} worker threads for PyTorch.")

        print("\nDefining the parameter space for the (tokenization and encoding) phase.")

        # Defining a parameter space for the tokenization-encoding phase.
        TOKENIZATION_ENCODING_PARAMETER_SPACE = ParameterSpace({
            "to_lower_and_regex_substitution_rules": [
                            # Make everything lower-case, only allow {alpha, digit, space}.
                            (True, [(r"[^a-z0-9 ]", "")]),
                           ],

            "token_filter_condition": [(lambda token, frequency: frequency >= 2, "(lambda token, frequency: frequency >= 2)")],
            "maximum_vocabulary_size": [4096],
            "target_input_list_length": [256],

            "padding_token": [("<PAD>", 0)],
            "unknown_token": [("<UNK>", 1)],

            "num_classes": [4],
        })

        print("\nDefining the parameter space for the (training) phase.")

        TRAINING_PARAMETER_SPACE: ParameterSpace = ParameterSpace({
            "batch_size": [128],

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

        return (num_worker_threads, TOKENIZATION_ENCODING_PARAMETER_SPACE, TRAINING_PARAMETER_SPACE,
                                    CNN_PARAMETER_SPACE, LSTM_PARAMETER_SPACE)


    def build_models_tracker() -> tuple[list, list, dict]:
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
        models_tracker: dict[str, list[dict]] = {
            "CNNTextClassifier": [],
            "LSTMTextClassifier": [],
            "BERTTextClassifier": []
        }

        print(f"\nModels tracker:\n{models_tracker}")

        return (epoch_metrics, model_final_metrics, models_tracker)


    def tokenize_encode_1(tokenization_parameters: dict[str, Any],
                          train_df: pd.DataFrame,
                          validate_df: pd.DataFrame,
                          initial_test_df: pd.DataFrame) -> tuple[dict[str, int], list[list[int]], list[int],
                                                                                  list[list[int]], list[int],
                                                                                  list[list[int]], list[int]]:

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

        print(f"\n\tObtained vocabulary.")


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

        print(f"\n\tEncoding the \"input\" vectors (validate sub-set).")

        validate_results: tuple[list[list[int]], list[int]] = encode(text_vector = validate_df["input"],
                                                                     rules = tokenization_parameters["to_lower_and_regex_substitution_rules"],
                                                                     padding_token = tokenization_parameters["padding_token"],
                                                                     unknown_token = tokenization_parameters["unknown_token"],
                                                                     vocabulary = vocabulary,
                                                                     target_input_list_length = tokenization_parameters["target_input_list_length"])
        validate_encodings: list[list[int]] = validate_results[0]
        validate_non_padded_lengths: list[int] = validate_results[1]


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

        return (vocabulary, train_encodings, train_non_padded_lengths,
                            validate_encodings, validate_non_padded_lengths,
                            initial_test_encodings, initial_test_non_padded_lengths)


    def encode_2(train_encodings, train_non_padded_lengths, train_df,
                 validate_encodings, validate_non_padded_lengths, validate_df,
                 initial_test_encodings, initial_test_non_padded_lengths, initial_test_df
                 ) -> tuple[TensorDataset, TensorDataset, TensorDataset]:

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

        print(f"\n\tTransformed the train sub-set into `torch.tensor`'s (for input, output, non-padded lengths).")


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

        return (train_td, validate_td, initial_test_td)


    def loader_1(training_parameters: dict[str, Any],
                 num_worker_threads: int,
                 train_td: TensorDataset,
                 validate_td: TensorDataset,
                 initial_test_td: TensorDataset
                 ) -> tuple[DataLoader, DataLoader, DataLoader]:

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

        return (train_loader, validate_loader, initial_test_loader)


    def load_best_model(specific_model: tuple[dict[str, Any], dict[str, Any], float] | dict,
                        model_type: str,
                        device: torch.device
                        ) -> nn.Module:
    
        if model_type == "CNNTextClassifier":
            state_dict: dict[str, Any] = specific_model["weights"]
            init_params: dict[str, Any] = specific_model["model"]
        
            model: nn.Module = CNNTextClassifier(**init_params)
            model.load_state_dict(state_dict)

        elif model_type == "LSTMTextClassifier":
            state_dict: dict[str, Any] = specific_model["weights"]
            init_params: dict[str, Any] = specific_model["model"]
            
            model: nn.Module = LSTMTextClassifier(**init_params)
            model.load_state_dict(state_dict)

        elif model_type == "BERTTextClassifier":
            model = AutoModelForSequenceClassification.from_pretrained(
                specific_model["bert_parameters"]["initial_model"],
                num_labels = specific_model["bert_parameters"]["num_classes"]
            )
            model.load_state_dict(specific_model["weights"])

        model.to(device)
        model.eval()
        
        return model


    def pipeline_train_best_LSTM_and_CNN(seed: int, device: torch.device,
                                         train_df, validate_df, initial_test_df, final_test_df,
                                         mapping_120k, mapping_7k,
                                         epoch_metrics: dict, model_final_metrics: dict, models_tracker: dict):
        """
            Executes the complete pipeline, from retrieving the data, seeding the PRNGs,
             pre-processing the input, training the models while fine-tuning them,
             evaluating the models and logging some metrics to file (for later visualization).
        """

        # Setting a reproducible PRNG state, per pipeline (not just for data-splitting).
        print(f"\nSeeding PRNGs for the LSTM+CNN pipeline: {seed}.")
        set_seed(seed)

        (num_worker_threads,
         TOKENIZATION_ENCODING_PARAMETER_SPACE,
         TRAINING_PARAMETER_SPACE,
         CNN_PARAMETER_SPACE, LSTM_PARAMETER_SPACE) = build_parameter_space_LSTM_and_CNN()


        for tokenization_parameters in TOKENIZATION_ENCODING_PARAMETER_SPACE:

            (vocabulary,
             train_encodings, train_non_padded_lengths,
             validate_encodings, validate_non_padded_lengths,
             initial_test_encodings, initial_test_non_padded_lengths) = tokenize_encode_1(tokenization_parameters,
                                                                                          train_df, validate_df, initial_test_df)

            (train_td,
             validate_td,
             initial_test_td) = encode_2(train_encodings, train_non_padded_lengths, train_df,
                                         validate_encodings, validate_non_padded_lengths, validate_df,
                                         initial_test_encodings, initial_test_non_padded_lengths, initial_test_df)

            for training_parameters in TRAINING_PARAMETER_SPACE:

                (train_loader,
                 validate_loader,
                 initial_test_loader) = loader_1(training_parameters, num_worker_threads,
                                                 train_td, validate_td, initial_test_td)

                for model_parameters_LSTM in LSTM_PARAMETER_SPACE:
                    train_LSTM(device, vocabulary,
                               tokenization_parameters, training_parameters, model_parameters_LSTM,
                               train_loader, validate_loader, initial_test_loader,
                               epoch_metrics, model_final_metrics, models_tracker)

                for model_parameters_CNN in CNN_PARAMETER_SPACE:
                    train_CNN(device, vocabulary,
                              tokenization_parameters, training_parameters, model_parameters_CNN,
                              train_loader, validate_loader, initial_test_loader,
                              epoch_metrics, model_final_metrics, models_tracker)

            del train_loader, validate_loader, initial_test_loader
            gc.collect()
            torch.cuda.empty_cache()


    def build_parameter_space_BERT(try_combinations: bool) -> ParameterSpace:
        print("\nDefining the parameter space for BERT.")

        # Assignment requirements:
        #  - Fine-tune the Transformer (document tokenizer, max length, LR, batch size, epochs, early stopping).

        if try_combinations:
            BERT_PARAMETER_SPACE = ParameterSpace({
                "initial_model": ["bert-base-uncased"],
                "tokenizer_name": ["bert-base-uncased"],
                "num_worker_threads": [0],

                "maximum_length": [64, 128],
                "learning_rate": [2e-5, 5e-5],
                "batch_size": [16, 128],
                "epochs": [1, 3, 5],
                "patience": [0],
                "num_classes": [4],
            })

        else:
            BERT_PARAMETER_SPACE = ParameterSpace({
                "initial_model": ["bert-base-uncased"],
                "tokenizer_name": ["bert-base-uncased"],
                "num_worker_threads": [0],

                "maximum_length": [128],
                "learning_rate": [2e-5],
                "batch_size": [16],
                "epochs": [3],
                "patience": [0],
                "num_classes": [4],
            })

        return BERT_PARAMETER_SPACE


    def pipeline_finetune_pretrained_BERT(seed: int, device: torch.device,
                                          train_df, validate_df, initial_test_df, final_test_df,
                                          mapping_120k, mapping_7k,
                                          epoch_metrics: dict, model_final_metrics: dict, models_tracker: dict,
                                          try_combinations: bool):

        # Setting a reproducible PRNG state, per pipeline (not just for data-splitting).
        print(f"\nSeeding PRNGs for the BERT pipeline: {seed}.")
        set_seed(seed)

        BERT_PARAMETER_SPACE: ParameterSpace = build_parameter_space_BERT(try_combinations)


        for bert_parameters in BERT_PARAMETER_SPACE:

            print(f"\n\tFine tuning with BERT parameters: {bert_parameters}")

            # Load model & tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(bert_parameters["initial_model"],
                                                                       num_labels = bert_parameters["num_classes"])
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(bert_parameters["tokenizer_name"])

            # Tokenize helper method
            def tokenize_dataframe(dataframe):
                encoding = tokenizer(dataframe["input"].tolist(),
                                     truncation = True,
                                     padding = "max_length",
                                     max_length = bert_parameters["maximum_length"],
                                     return_tensors = "pt")
                labels = torch.tensor(dataframe["output"].values)
                return TensorDataset(encoding["input_ids"], encoding["attention_mask"], labels)

            train_dataset = tokenize_dataframe(train_df)
            validate_dataset = tokenize_dataframe(validate_df)
            test_dataset = tokenize_dataframe(initial_test_df)

            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=bert_parameters["batch_size"],
                                      shuffle = True, pin_memory = True,
                                      num_workers = bert_parameters["num_worker_threads"])

            validate_loader = DataLoader(validate_dataset, batch_size = bert_parameters["batch_size"],
                                         shuffle = False, pin_memory = True,
                                         num_workers = bert_parameters["num_worker_threads"])

            test_loader = DataLoader(test_dataset, batch_size = bert_parameters["batch_size"],
                                     shuffle = False, pin_memory = True,
                                     num_workers = bert_parameters["num_worker_threads"])

            # Optimizer and mixed-precision scaler
            optimizer = AdamW(model.parameters(), lr = bert_parameters["learning_rate"])
            scaler = torch.amp.GradScaler()

            # Training loop
            best_val_f1 = 0.0
            patience = bert_parameters["patience"]
            wait = 0

            model.train()
            for epoch in range(bert_parameters["epochs"]):
                loop = tqdm(train_loader, desc = f"Epoch {epoch+1}")
                for input_ids, attention_mask, labels in loop:
                    input_ids = input_ids.to(device, non_blocking = True)
                    attention_mask = attention_mask.to(device, non_blocking = True)
                    labels = labels.to(device, non_blocking = True)

                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type = "cuda"):
                        outputs = model(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        labels = labels)
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loop.set_postfix(loss = loss.item())

                model.eval()
                val_preds, val_labels = [], []

                with torch.no_grad():
                    for input_ids, attention_mask, labels in validate_loader:
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=1)

                        val_preds.append(preds.cpu())
                        val_labels.append(labels.cpu())

                val_preds = torch.cat(val_preds).numpy()
                val_labels = torch.cat(val_labels).numpy()
                val_f1 = f1_score(val_labels, val_preds, average="macro")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    wait = 0
                    # save best model weights
                    best_weights = {k: v.cpu() for k, v in deepcopy(model.state_dict()).items()}
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break


            # Evaluation on initial-test split
            print("Evaluating BERT.")
            model.load_state_dict(best_weights)
            model.eval()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for input_ids, attention_mask, labels in tqdm(test_loader, desc = "Evaluation"):
                    input_ids = input_ids.to(device, non_blocking = True)
                    attention_mask = attention_mask.to(device, non_blocking = True)
                    labels = labels.to(device, non_blocking = True)

                    with torch.amp.autocast(device_type = "cuda"):
                        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                    preds = torch.argmax(outputs.logits, dim = 1)

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # Metrics
            print("Computing initial test metrics, for BERT.")

            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average = "macro")
            macro_precision = precision_score(all_labels, all_preds, average = "macro")
            macro_recall = recall_score(all_labels, all_preds, average = "macro")
            confusion_matrix_metric = confusion_matrix(all_labels, all_preds)

            print(f"Accuracy: {acc:.4f}"
                  f"\nMacro-Precision: {macro_precision:.4f}"
                  f"\nMacro-Recall: {macro_recall:.4f}"
                  f"\nMacro-F1: {macro_f1:.4f}"
                  f"\nConfusion Matrix:\n{confusion_matrix_metric}")


            index_model = len(models_tracker["BERTTextClassifier"])

            model_final_metrics.append({
                "model": "BERTTextClassifier",
                "identifier": index_model,

                "test_accuracy": acc,
                "test_precision": macro_precision,
                "test_recall": macro_recall,
                "test_f1": macro_f1,
            })

            # Save weights on CPU
            best_weights = {k: v.cpu() for k, v in deepcopy(model.state_dict()).items()}

            models_tracker["BERTTextClassifier"].append({
                "f1_score": macro_f1,
                "weights": best_weights,

                "bert_parameters": bert_parameters
            })

            # Error analysis
            final_test_dataset = tokenize_dataframe(final_test_df)

            final_test_loader = DataLoader(final_test_dataset,
                                          batch_size = bert_parameters["batch_size"],
                                          shuffle = False,
                                          pin_memory = True,
                                          num_workers = bert_parameters["num_worker_threads"])

            model.eval()

            all_preds_final = []
            all_labels_final = []
            all_texts_final = final_test_df["input"].tolist()

            with torch.no_grad():
                for input_ids, attention_mask, labels in tqdm(final_test_loader, desc="Final Test Evaluation"):
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    preds = torch.argmax(outputs.logits, dim=1)

                    all_preds_final.append(preds.cpu())
                    all_labels_final.append(labels.cpu())

            all_preds_final = torch.cat(all_preds_final).numpy()
            all_labels_final = torch.cat(all_labels_final).numpy()

            # Collect misclassified indices
            misclassified_indices = []
            for i in range(len(all_preds_final)):
                if all_preds_final[i] != all_labels_final[i]:
                    misclassified_indices.append(i)

            print(f"Total misclassified (final test): {len(misclassified_indices)}")

            num_samples = min(20, len(misclassified_indices))
            sampled_indices = random.sample(misclassified_indices, num_samples)

            print("\nSampled FINAL test misclassifications:\n")

            for idx in sampled_indices:
                print("=" * 80)
                print(f"TRUE: {all_labels_final[idx]} | PRED: {all_preds_final[idx]}")
                print(f"TEXT: {all_texts_final[idx][:300]}")


            del model, optimizer, scaler, train_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()


    def final_results(device: torch.device,
                      epoch_metrics: dict, model_final_metrics: dict, models_tracker: dict,
                      final_test_df: pd.DataFrame,
                      num_maximum_error_samples: int,
                      inverse_mapping_7k):

        # Now traverse all models, and run them against the ::final_test_df (7k samples).
        for tracked_model in models_tracker.keys():
            for index_specific_model in range(len(models_tracker[tracked_model])):
                specific_model = models_tracker[tracked_model][index_specific_model]


                print(f"\nAnalysis for model {tracked_model}[{index_specific_model}].")

                if tracked_model in ["CNNTextClassifier", "LSTMTextClassifier"]:
                    model: nn.Module = load_best_model(specific_model, tracked_model, device)
                    tokenizer_rules = specific_model["tokenization_parameters"]["to_lower_and_regex_substitution_rules"]
                    padding_token = specific_model["tokenization_parameters"]["padding_token"]
                    unknown_token = specific_model["tokenization_parameters"]["unknown_token"]
                    vocabulary = specific_model["vocabulary"]
                    target_input_list_length = specific_model["tokenization_parameters"]["target_input_list_length"]
                    num_classes = specific_model["tokenization_parameters"]["num_classes"]


                    # Tuple format: (expected label, predicted label, associated input text).
                    errors: list[tuple[int, int, str]] = get_misclassified_examples_CNN_LSTM(device = device,
                                                                                             model = model,
                                                                                             data = final_test_df,
                                                                                             maximum_error_samples = num_maximum_error_samples,
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

                elif tracked_model in ["BERTTextClassifier"]:
                    model = load_best_model(specific_model, tracked_model, device)
                    model.to(device)
                    model.eval()

                    tokenizer = AutoTokenizer.from_pretrained(specific_model["bert_parameters"]["tokenizer_name"])
                    max_length = specific_model["bert_parameters"]["maximum_length"]
                    num_classes = specific_model["bert_parameters"]["num_classes"]

                    # Tokenize entire dataset once
                    encoding = tokenizer(final_test_df["input"].tolist(),
                                         truncation = True,
                                         padding = "max_length",
                                         max_length = max_length,
                                         return_tensors = "pt")

                    dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"],
                                            torch.tensor(final_test_df["output"].values))

                    loader = DataLoader(dataset, batch_size = 32, shuffle = False)

                    all_preds = []
                    all_labels = []
                    errors = []

                    with torch.no_grad():
                        for batch_idx, (input_ids, attention_mask, labels) in enumerate(loader):
                            input_ids = input_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            labels = labels.to(device)

                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            preds = torch.argmax(outputs.logits, dim=1)

                            all_preds.append(preds.cpu())
                            all_labels.append(labels.cpu())

                            if len(errors) < num_maximum_error_samples:
                                index_begin: int = batch_idx * loader.batch_size
                                index_end: int = (batch_idx + 1) * loader.batch_size

                                batch_texts = final_test_df["input"].iloc[index_begin : index_end].tolist()

                                for t, p, text in zip(labels.cpu(), preds.cpu(), batch_texts):
                                    if t.item() != p.item() and len(errors) < num_maximum_error_samples:
                                        errors.append((inverse_mapping_7k[t.item()],
                                                       inverse_mapping_7k[p.item()],
                                                       text))

                    all_preds = torch.cat(all_preds).numpy()
                    all_labels = torch.cat(all_labels).numpy()

                    print(f"{tracked_model}[{index_specific_model}] errors:")
                    for error in errors:
                        print(f"\t{error}")


        for entry in model_final_metrics:
            model_name = entry["model"]
            identifier = entry["identifier"]
            test_acc = entry["test_accuracy"]
            test_f1 = entry["test_f1"]

            print(f"\nModel: {model_name}[{identifier}]")
            print(f"\tAccuracy : {test_acc:.4f}")
            print(f"\tF1 score : {test_f1:.4f}")


    def subsample_training_data(train_df: pd.DataFrame, fraction: float, seed: int):
        return train_df.sample(frac = fraction, random_state = seed).reset_index(drop = True)


    def main():
        REPRODUCIBILITY_PRNG_SEED: Final[int] = 0
        NUM_MAXIMUM_ERROR_SAMPLES: Final[int] = 20

        # Setting a reproducible PRNG state.
        print(f"\nSeeding PRNGs: {REPRODUCIBILITY_PRNG_SEED}.")
        set_seed(REPRODUCIBILITY_PRNG_SEED)


        (epoch_metrics,
         model_final_metrics,
         models_tracker) = build_models_tracker()

        (train_df,
         validate_df,
         initial_test_df,
         final_test_df,
         mapping_120k,
         mapping_7k) = get_preprocessed_dataset(REPRODUCIBILITY_PRNG_SEED)

        device: torch.device = get_device()


        pipeline_train_best_LSTM_and_CNN(REPRODUCIBILITY_PRNG_SEED, device,
                                         train_df, validate_df, initial_test_df, final_test_df,
                                         mapping_120k, mapping_7k,
                                         epoch_metrics, model_final_metrics, models_tracker)

        pipeline_finetune_pretrained_BERT(REPRODUCIBILITY_PRNG_SEED, device,
                                          train_df, validate_df, initial_test_df, final_test_df,
                                          mapping_120k, mapping_7k,
                                          epoch_metrics, model_final_metrics, models_tracker, True)


        # Further analyses:
        # 1) Label-noise sensitivity
        train_25_df = subsample_training_data(train_df, 0.25, REPRODUCIBILITY_PRNG_SEED)
        train_50_df = subsample_training_data(train_df, 0.50, REPRODUCIBILITY_PRNG_SEED)
        
        pipeline_finetune_pretrained_BERT(REPRODUCIBILITY_PRNG_SEED, device,
                                          train_25_df, validate_df, initial_test_df, final_test_df,
                                          mapping_120k, mapping_7k,
                                          epoch_metrics, model_final_metrics, models_tracker, False)

        pipeline_finetune_pretrained_BERT(REPRODUCIBILITY_PRNG_SEED, device,
                                          train_50_df, validate_df, initial_test_df, final_test_df,
                                          mapping_120k, mapping_7k,
                                          epoch_metrics, model_final_metrics, models_tracker, False)

        
        # 2) Input field stress test
        train_title_df = train_df.copy()
        validate_title_df = validate_df.copy()
        initial_test_title_df = initial_test_df.copy()
        final_test_title_df = final_test_df.copy()

        train_title_df["input"] = train_title_df["title"]
        validate_title_df["input"] = validate_title_df["title"]
        initial_test_title_df["input"] = initial_test_title_df["title"]
        final_test_title_df["input"] = final_test_title_df["title"]

        pipeline_finetune_pretrained_BERT(REPRODUCIBILITY_PRNG_SEED, device,
                                          train_title_df, validate_title_df, initial_test_title_df, final_test_title_df,
                                          mapping_120k, mapping_7k,
                                          epoch_metrics, model_final_metrics, models_tracker, False)

        # Printing results
        inverse_mapping_7k: dict[int, int] = {v: k for k, v in mapping_7k.items()}
        final_results(device,
                      epoch_metrics, model_final_metrics, models_tracker,
                      final_test_df, NUM_MAXIMUM_ERROR_SAMPLES, inverse_mapping_7k)

    main()
