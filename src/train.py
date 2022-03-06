import os
import sys
import typing
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from transformers import logging

from .model import BertClassifier
from .utils import set_random_seed

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from config import DEVICE, METRICS, MODEL_DIR, THRESHOLD

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def run_training(
    train_dataloader,
    valid_dataloader,
    epochs,
    lr,
    random_seed=None,
) -> float:

    if random_seed is not None:
        set_random_seed(random_seed=random_seed)

    model = BertClassifier().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    max_metrics = 0

    for epoch in range(epochs):

        running_train_loss = 0.0
        running_train_examples_count = 0

        model.train()

        for batch, data in enumerate(train_dataloader):

            outputs = model(
                {
                    data_key: data_value.to(DEVICE)
                    for data_key, data_value in data.items()
                    if data_key != "targets"
                }
            )
            loss = criterion(outputs, data["targets"])
            running_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_examples_count += data["targets"].size(0)

            if batch % 100 == 0:

                print(
                    f"Epoch: [{epoch + 1}/{epochs}], "
                    f"Batch: [{batch + 1}]/[{len(train_dataloader)}], "
                    f"Loss: {round(running_train_loss / running_train_examples_count, 4)}"
                )

        # Evaluate Model
        validation_output = evaluate(model=model, dataloader=valid_dataloader)

        print(
            f"Epoch Loss: {round(running_train_loss / len(train_dataloader), 4)}, "
            f"Valid Accuracy: {round(validation_output['accuracy'], 2)}, ",
            f"F1_Micro: {round(validation_output['f1_micro'], 2)}, ",
            f"F1_Macro: {round(validation_output['f1_macro'], 2)}",
        )

        if validation_output[METRICS] > max_metrics:

            model.save_pretrained(MODEL_DIR)
            max_metrics = validation_output[METRICS]

    return validation_output


def evaluate(
    model, dataloader, threshold: float = THRESHOLD
) -> typing.Union[float, torch.Tensor]:

    model.eval()
    with torch.no_grad():

        model_targets, batch_outputs = [], []
        for _, data in enumerate(dataloader):

            outputs = model(
                {
                    data_key: data_value.to(DEVICE)
                    for data_key, data_value in data.items()
                    if data_key != "targets"
                }
            )

            model_targets.extend(
                data["targets"].cpu().detach().numpy().astype(int).tolist()
            )
            batch_outputs.extend(outputs.cpu().detach().numpy().tolist())

        model_outputs = (np.array(batch_outputs) >= threshold).astype(int).tolist()
        accuracy = metrics.accuracy_score(model_targets, model_outputs)
        f1_score_micro = metrics.f1_score(model_targets, model_outputs, average="micro")
        f1_score_macro = metrics.f1_score(model_targets, model_outputs, average="macro")

        return {
            "accuracy": accuracy,
            "f1_micro": f1_score_micro,
            "f1_macro": f1_score_macro,
            "outputs": model_outputs,
        }
