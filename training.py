from time import perf_counter
from typing import Any

from numpy import ndarray
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             )
from torch import (Tensor,
                   tensor as torch_tensor,
                   device as torch_device,
                   argmax as torch_argmax,
                   cat as torch_cat,
                   no_grad as torch_no_grad,
                   )
from torch.nn import Module, CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam


def evaluate(device: torch_device,
             model: Module,
             loader: DataLoader
             ) -> dict[str, float]:

    model.eval()
    predictions: list[Tensor] = []
    gold: list[Tensor] = []

    with torch_no_grad():
        x: Tensor
        y: Tensor
        lengths: Tensor

        for x, y, lengths in loader:
            x = x.to(device)
            #lengths = lengths.to(device)

            logits: Tensor = model(x, lengths)
            prediction: Tensor = torch_argmax(logits, dim = 1)

            predictions.append(prediction.cpu())
            #gold.append(y.cpu())
            gold.append(y)

    merged_predictions: ndarray = torch_cat(predictions).numpy()
    merged_gold: ndarray = torch_cat(gold).numpy()

    return {"accuracy": accuracy_score(merged_gold, merged_predictions),

            "precision": precision_score(merged_gold, merged_predictions,
                                         average = "macro", zero_division = 0),

            "recall": recall_score(merged_gold, merged_predictions,
                                   average = "macro", zero_division = 0),

            "f1": f1_score(merged_gold, merged_predictions,
                           average = "macro", zero_division = 0)}


def fit(device: torch_device,
        model: Module,
        train_loader: DataLoader,
        validate_loader: DataLoader,
        learning_rate: float,
        maximum_epochs: int,
        patience: int,
        clip_gradient_norm: float
        ) -> list[dict[str, Any]]:

    optimizer: Adam = Adam(model.parameters(),
                           lr = learning_rate)
    criterion: CrossEntropyLoss = CrossEntropyLoss()

    best_f1: float = 0
    wait: int = 0
    history: list[dict[str, Any]] = []

    epoch: int
    for epoch in range(1, maximum_epochs + 1):
        model.train()

        cummulated_loss: Tensor = torch_tensor(0.0, device = device)

        x: Tensor
        y: Tensor
        lengths: Tensor

        for x, y, lengths in train_loader:
            x = x.to(device)
            y = y.to(device)
            #lengths = lengths.to(device)

            optimizer.zero_grad()

            logits: Tensor = model(x, lengths)
            loss: Tensor = criterion(logits, y)
            loss.backward()

            clip_grad_norm_(model.parameters(), clip_gradient_norm)
            optimizer.step()

            cummulated_loss += loss.detach()

        epoch_loss: float = cummulated_loss.item() / len(train_loader)
        validate: dict[str, float] = evaluate(device, model, validate_loader)

        history.append({"epoch": epoch,
                        "loss": epoch_loss,
                        "accuracy": validate["accuracy"],
                        "precision": validate["precision"],
                        "recall": validate["recall"],
                        "f1": validate["f1"],
                        })

        if validate["f1"] > best_f1:
            best_f1 = validate["f1"]
            wait = 0

        else:
            wait += 1
            if wait >= patience:
                break

    return history


def train_time_and_evaluate(device: torch_device,
                            model: Module,
                            train_loader: DataLoader,
                            validate_loader: DataLoader,
                            test_loader: DataLoader,
                            learning_rate: float,
                            maximum_epochs: int,
                            patience: int,
                            clip_gradient_norm: float
                            ) -> dict[str, Any]:

    begin_time: float = perf_counter()

    history: list[dict[str, Any]] = fit(device,
                                        model,
                                        train_loader,
                                        validate_loader,
                                        learning_rate,
                                        maximum_epochs,
                                        patience,
                                        clip_gradient_norm
                                        )

    end_time: float = perf_counter()

    total_time: float = end_time - begin_time
    validate: dict[str, float] = evaluate(device, model, validate_loader)
    test: dict[str, float] = evaluate(device, model, test_loader)

    return {"history": history,
            "validate": validate,
            "test": test,
            "total_time_in_seconds": total_time
            }
