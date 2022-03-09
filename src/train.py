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
    optimizer_parameters,
    epochs,
    random_seed=None,
    lr_step_parameters={"step_size": 4, "gamma": 0.1},
    logging_interval=500,
) -> float:

    if random_seed is not None:
        set_random_seed(random_seed=random_seed)

    model = BertClassifier().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_parameters)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_step_parameters)

    max_metrics = 0

    for epoch in range(epochs):

        training_loss = []

        model.train()

        for batch, data in enumerate(train_dataloader):

            targets = data.pop("targets").to(DEVICE)
            inputs = {
                data_key: data_value.to(DEVICE) for data_key, data_value in data.items()
            }

            optimizer.zero_grad()

            outputs = model(**inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            training_loss.append(loss_value)

            if batch % logging_interval == 0:

                print(
                    f"Epoch: [{epoch + 1} / {epochs}], "
                    f"Batch: [{batch + 1} / {len(train_dataloader)}], "
                    f"Loss: {round(loss_value, 4)}"
                )

        # Evaluate Model
        print("Running Evaluation...")
        validation_output = evaluate(model=model, dataloader=valid_dataloader)

        print(
            f"Epoch {epoch + 1} Loss: {round(sum(training_loss) / len(train_dataloader), 4)}, "
            f"Valid Accuracy: {round(validation_output['accuracy'], 2)}, ",
            f"F1 Micro: {round(validation_output['f1_micro'], 2)}, ",
            f"F1 Macro: {round(validation_output['f1_macro'], 2)}",
        )

        if validation_output[METRICS] > max_metrics:

            model.save_pretrained(MODEL_DIR)
            max_metrics = validation_output[METRICS]

        scheduler.step()

    return validation_output


def evaluate(
    model, dataloader, threshold: float = THRESHOLD
) -> typing.Union[float, torch.Tensor]:

    model.eval()
    with torch.no_grad():

        model_targets, batch_outputs = [], []
        for _, data in enumerate(dataloader):

            targets = data.pop("targets").to(DEVICE)
            inputs = {
                data_key: data_value.to(DEVICE) for data_key, data_value in data.items()
            }

            outputs = model(**inputs)

            model_targets.extend(targets.cpu().detach().numpy().astype(int).tolist())
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
