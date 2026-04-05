import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module

from preprocess import tokenize, encode
from tuning import ParameterRange, ParameterSpace


def get_misclassified_examples(device: torch.device,
                               model: Module,
                               data: pd.DataFrame,
                               maximum_error_samples: int,
                               tokenizer_rules: tuple[bool, list[tuple[str, str]]],
                               padding_token: tuple[str, int],
                               unknown_token: tuple[str, int],
                               vocabulary: dict[str, int],
                               target_input_list_length: int
                               ) -> list[tuple[int, int, str]]:
    model.eval()
    errors: list[tuple[int, int, str]] = []


    encoding_results: tuple[Any, Any] = encode(text_vector = data["input"],
                                               rules = tokenizer_rules,
                                               padding_token = padding_token,
                                               unknown_token = unknown_token,
                                               vocabulary = vocabulary,
                                               target_input_list_length = target_input_list_length)

    encodings: list[list[int]] = encoding_results[0]
    non_padded_lengths: list[int] = encoding_results[1]

    text: str
    x: list[int]
    y: int
    length: int

    for text, x, y, length in zip(data["input"], encodings, data["output"], non_padded_lengths):
        x_tensor: Tensor = torch.tensor([x], dtype = torch.long, device = device)
        length_tensor: Tensor = torch.tensor([length], dtype = torch.long, device = "cpu")

        with torch.no_grad():
            logits: Tensor = model(x_tensor, length_tensor)
            prediction: int = int(logits.argmax(dim = 1).item())

        if prediction != y:
            snippet: str = text.replace("\n", " ")
            snippet = snippet[:250] + ("..." if len(snippet) > 250 else "")
            errors.append((y, prediction, snippet))

        if len(errors) >= maximum_error_samples:
            break

    return errors


def get_confusion_matrix(device: torch.device,
                         model: Module,
                         data: pd.DataFrame,
                         tokenizer_rules: tuple[bool, list[tuple[str, str]]],
                         padding_token: tuple[str, int],
                         unknown_token: tuple[str, int],
                         vocabulary: dict[str, int],
                         target_input_list_length: int,
                         num_classes: int
                         ) -> torch.Tensor:

    model.eval()

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    encoding_results: tuple[Any, Any] = encode(
        text_vector=data["input"],
        rules=tokenizer_rules,
        padding_token=padding_token,
        unknown_token=unknown_token,
        vocabulary=vocabulary,
        target_input_list_length=target_input_list_length
    )

    encodings: list[list[int]] = encoding_results[0]
    non_padded_lengths: list[int] = encoding_results[1]

    for x, y, length in zip(encodings, data["output"], non_padded_lengths):

        x_tensor: Tensor = torch.tensor([x], dtype = torch.long, device = device)
        length_tensor: Tensor = torch.tensor([length], dtype = torch.long, device = "cpu")

        with torch.no_grad():
            logits: Tensor = model(x_tensor, length_tensor)
            prediction: int = int(logits.argmax(dim = 1).item())

        confusion[y, prediction] += 1

    return confusion


def compute_classification_metrics(confusion: Tensor
                                   ) -> pd.DataFrame:

    cm: np.ndarray = confusion.cpu().numpy()
    num_classes: int = cm.shape[0]

    metrics: list[tuple[float, float, float]] = []

    for index in range(num_classes):

        TP: int = int(cm[index, index])
        FP: int = int(cm[:, index].sum() - TP)
        FN: int = int(cm[index, :].sum() - TP)

        precision: float = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall: float = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        f1: float = ( 2 * precision * recall / (precision + recall) if (precision + recall) > 0
                      else 0.0 )

        metrics.append((index, precision, recall, f1))

    return pd.DataFrame(metrics,
                        columns=["label", "precision", "recall", "f1"])


def plot_learning_curves(epoch_metrics: list[dict[str, Any]],
                         model_name: str,
                         identifier: str,
                         key = "accuracy",
                         output_dir = "plots"
                         ) -> str:

    os.makedirs(output_dir, exist_ok = True)
    
    # filter the epoch_metrics for this model and identifier
    filtered = [m for m in epoch_metrics if m["model"] == model_name and m["identifier"] == identifier]
    filtered = sorted(filtered, key=lambda x: x["epoch"])
    
    epochs = [m["epoch"] for m in filtered]
    values = [m[key] for m in filtered]
    
    plt.figure(figsize=(8,5))
    plt.plot(epochs, values, marker='o', label=f"{model_name}-{identifier}")
    plt.xlabel("Epoch")
    plt.ylabel(key.capitalize())
    plt.title(f"{key.capitalize()} learning curve: {model_name}-{identifier}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"{model_name}_{identifier}_{key}.png")
    plt.savefig(out_file)
    plt.close()

    return out_file

