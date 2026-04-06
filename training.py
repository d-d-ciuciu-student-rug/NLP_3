from copy import deepcopy
import gc


import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from time import perf_counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, Dict, List

from models import CNNTextClassifier, LSTMTextClassifier


def evaluate(device: torch.device, model: Module, loader: DataLoader) -> dict:
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            preds.append(logits.argmax(dim=1).cpu())
            golds.append(y.cpu())
    preds = torch.cat(preds).numpy()
    golds = torch.cat(golds).numpy()
    return {
        "accuracy": accuracy_score(golds, preds),
        "precision": precision_score(golds, preds, average="macro", zero_division=0),
        "recall": recall_score(golds, preds, average="macro", zero_division=0),
        "f1": f1_score(golds, preds, average="macro", zero_division=0)
    }


def train_time_and_evaluate(device: torch.device,
                            model: Module,
                            train_loader: DataLoader,
                            validate_loader: DataLoader,
                            test_loader: DataLoader,
                            learning_rate: float,
                            maximum_epochs: int,
                            patience: int,
                            clip_gradient_norm: float) -> dict:

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    history = []
    best_f1, wait = 0.0, 0
    start_time = perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            y = y.clamp(0, model.num_classes - 1)

            optimizer.zero_grad()
            with torch.amp.autocast(enabled=use_amp, device_type="cuda"):
                logits = model(x, lengths)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if clip_gradient_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), clip_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach().item()

        avg_loss = running_loss / len(train_loader)
        val_metrics = evaluate(device, model, validate_loader)
        history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    total_time = perf_counter() - start_time

    del optimizer
    del scaler
    torch.cuda.empty_cache()

    return {"history": history,
            "validate": evaluate(device, model, validate_loader),
            "test": evaluate(device, model, test_loader),
            "total_time_in_seconds": total_time}


def train_CNN(device, vocabulary,
              tokenization_parameters, training_parameters, model_parameters_CNN,
              train_loader, validate_loader, initial_test_loader,
              epoch_metrics, model_final_metrics, models_tracker):
    print(f"\n\t\t---- Training CNN model with model parameters: {model_parameters_CNN} ----")

    # Instantiate model and move to device
    model_CNN: CNNTextClassifier = CNNTextClassifier(
        vocab_size=len(vocabulary),
        embed_dim=model_parameters_CNN["embed_dim"],
        num_filters=model_parameters_CNN["num_filters"],
        kernel_sizes=model_parameters_CNN["kernel_sizes"],
        dropout=model_parameters_CNN["dropout"],
        pad_idx=tokenization_parameters["padding_token"][1],
        num_classes=tokenization_parameters["num_classes"],
    ).to(device)

    print(f"\n\t\t\tNumber of trainable parameters: {model_CNN.count_parameters()}.")

    # Optimizer, loss, and mixed precision scaler
    optimizer = torch.optim.Adam(model_CNN.parameters(), lr=training_parameters["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()  # Only effective if device is GPU

    best_f1 = 0.0
    wait = 0
    history: list[dict[str, Any]] = []

    print("\n\t\t\tStarting training (and timer).")
    start_time = perf_counter()

    for epoch in range(1, training_parameters["maximum_epochs"] + 1):
        model_CNN.train()
        running_loss = 0.0

        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            optimizer.zero_grad()

            # Automatic Mixed Precision
            with torch.amp.autocast(enabled = (device.type == "cuda"), device_type="cuda"):
                logits = model_CNN(x, lengths)
                # Clamp labels just in case
                y_clamped = y.clamp(0, model_CNN.num_classes - 1) if hasattr(model_CNN, "num_classes") else y
                loss = criterion(logits, y_clamped)

            scaler.scale(loss).backward()

            if training_parameters["clip_gradient_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_CNN.parameters(), training_parameters["clip_gradient_norm"])

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach().item()

        avg_loss = running_loss / len(train_loader)
        validate_metrics = evaluate(device, model_CNN, validate_loader)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            **validate_metrics
        })

        # Early stopping based on F1
        if validate_metrics["f1"] > best_f1:
            best_f1 = validate_metrics["f1"]
            wait = 0
        else:
            wait += 1
            if wait >= training_parameters["patience"]:
                print(f"\t\t\tEarly stopping at epoch {epoch}")
                break

    end_time = perf_counter()

    test_metrics = evaluate(device, model_CNN, initial_test_loader)
    total_time = end_time - start_time
    print(f"\t\t\tTraining took {total_time} seconds.")

    results_CNN = {
        "history": history,
        "validate": validate_metrics,
        "test": test_metrics,
        "total_time_in_seconds": total_time
    }

    # Logging and model tracking (same as before)
    index_model = len(models_tracker["CNNTextClassifier"])
    for entry in history:
        epoch_metrics.append({"model": "CNNTextClassifier", "identifier": index_model, **entry})

    model_final_metrics.append({
        "model": "CNNTextClassifier", "identifier": index_model,
        "total_training_time_in_seconds": total_time,
        "validate_accuracy": validate_metrics["accuracy"],
        "validate_precision": validate_metrics["precision"],
        "validate_recall": validate_metrics["recall"],
        "validate_f1": validate_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"]
    })

    best_weights = {k: v.cpu() for k, v in deepcopy(model_CNN.state_dict()).items()}
    models_tracker["CNNTextClassifier"].append({
        "f1_score": best_f1,
        "weights": best_weights,
        "model": {"vocab_size": len(vocabulary),
                  "embed_dim": model_parameters_CNN["embed_dim"],
                  "num_filters": model_parameters_CNN["num_filters"],
                  "kernel_sizes": model_parameters_CNN["kernel_sizes"],
                  "dropout": model_parameters_CNN["dropout"],
                  "pad_idx": tokenization_parameters["padding_token"][1],
                  "num_classes": tokenization_parameters["num_classes"],
                 },
        "vocabulary": deepcopy(vocabulary),
        "model_parameters": model_parameters_CNN,
        "training_parameters": training_parameters,
        "tokenization_parameters": tokenization_parameters
    })

    # Cleanup GPU memory
    del model_CNN, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()


def train_LSTM(device, vocabulary,
               tokenization_parameters, training_parameters, model_parameters_LSTM,
               train_loader, validate_loader, initial_test_loader,
               epoch_metrics, model_final_metrics, models_tracker):
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

    models_tracker["LSTMTextClassifier"].append({
        "f1_score": current_f1_score,
        "weights": best_weights,
        "model": {"vocab_size": len(vocabulary),
                  "embed_dim": model_parameters_LSTM["embed_dim"],
                  "hidden_dim": model_parameters_LSTM["hidden_dim"],
                  "num_layers": model_parameters_LSTM["num_layers"],
                  "dropout": model_parameters_LSTM["dropout"],
                  "pad_idx": tokenization_parameters["padding_token"][1],
                  "num_classes": tokenization_parameters["num_classes"],
                  "bidirectional": model_parameters_LSTM["bidirectional"],
                  },
        "vocabulary": deepcopy(vocabulary),
        "model_parameters": model_parameters_LSTM,
        "training_parameters": training_parameters,
        "tokenization_parameters": tokenization_parameters,
    })

    del model_LSTM
    gc.collect()
    torch.cuda.empty_cache()