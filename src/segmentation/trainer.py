from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from utils.logging.logger import LoggingConfigurator

configurator = LoggingConfigurator()


class SegmentationTrainer:
    """
    Class for training a segmentation model.

    Returns:
      A dictionary of evaluation metrics

    Eg:
    model_loader = LoadSegmentationModel()

    epochs = 10
    loss_fun = torch.nn.CrossEntropyLoss()
    LR = 0.001
    optimizer = torch.optim.Adam(model_loader.model.parameters(), lr=LR)
    model_dir = "./TestProject/models"

    trainer = SegmentationTrainer(
        model_loader.model,
        train_data_loader,
        val_data_loader,
        optimizer,
        model_dir,
        loss_fun,
        num_epochs=epochs,
    )
    metrics_dict = trainer.train()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model_dir: Path,
        loss_fun: torch.nn.Module = nn.CrossEntropyLoss(),
        num_epochs: int = 15,
        patience: int = 3,
        print_freq: int = 10,
    ):
        """
        Args:
            model (torch.nn.Module): The segmentation model.
            train_data_loader (torch.utils.data.DataLoader): The training data loader.
            val_data_loader (torch.utils.data.DataLoader): The validation data loader.
            optimizer (torch.optim.Optimizer): The optimizer.
            model_dir (str): The directory to store the best model.
            loss_fun (torch.nn.Module): The loss function.
            num_epochs (int): The number of training epochs.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 3.
            print_freq: How often to print training information (batches). Defaults to 5.
            criterion (str): The criterion for early stopping. Defaults to "loss".
            best_score (float): Initial best score. Defaults to infinity.
        """
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.loss_fun = loss_fun
        self.num_epochs = num_epochs
        self.patience = patience
        self.print_freq = print_freq
        self.criterion = "loss"
        self.best_score = float("inf")
        self.metrics: dict = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    def train(self):
        """
        Trains a model for a specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            # train
            self._train_epoch(epoch)
            # evaluate on validation data
            self.metrics = self._evaluate(epoch)

            # Check for early stopping
            if self._check_improvement(epoch):
                break

        return self.metrics

    def _train_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Trains the model for a single epoch.

        Args:
            epoch: The current epoch number.


        """
        # Set model to training mode
        self.model.train()

        # Initialize GradScaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()

        epoch_loss = 0.0
        num_batches = len(self.train_data_loader)
        for i, (x_batch, y_true_batch) in enumerate(self.train_data_loader):
            # Zero gradients before each step
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward pass
                if type(self.model).__name__.lower() == "unet":
                    y_pred = self.model(x_batch)
                else:
                    y_pred = self.model(x_batch)["out"]

                # Calculate loss
                loss = self.loss_fun(y_pred, y_true_batch)

            # Scale loss for backpropagation
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # Calculate epoch loss
            epoch_loss += loss.item() / (num_batches * len(x_batch))

            # log verbose training information
            if (i + 1) % self.print_freq == 0 or i == num_batches - 1:
                configurator.logger.info(
                    f"Epoch {epoch} - Training: Batch {i + 1}/{num_batches}, Loss: {loss:.4f}"
                )
        configurator.logger.info(f"Epoch {epoch} - Training loss: {epoch_loss}")

    def _evaluate(self, epoch):
        """
        Evaluates the model on the validation data.

        Args:
            epoch: The current epoch number.

        Returns:
            Updated metrics dictionary.
        """
        # Set model to evaluation mode
        self.model.eval()
        # Initialize
        y_true_list, y_pred_list = [], []
        total_loss = 0

        # Iterate over the validation data loader
        with torch.no_grad():
            for x_batch, y_true_batch in iter(self.val_data_loader):
                # Forward pass
                if type(self.model).__name__.lower() == "unet":
                    y_pred = self.model(x_batch)
                else:
                    y_pred = self.model(x_batch)["out"]

                # calculate loss
                total_loss += self.loss_fun(y_pred, y_true_batch)

                # Flatten predictions and ground truth
                y_pred_flat = (y_pred[0][0] - y_pred[0][1]).flatten() > 0.0
                y_true_flat = y_true_batch[0][0].flatten() != 0.0
                # Convert tensors to lists and append
                y_pred_list.extend(y_pred_flat.tolist())
                y_true_list.extend(y_true_flat.tolist())

        # Calculate evaluation metrics
        loss = total_loss / len(self.val_data_loader)
        acc = accuracy_score(y_true_list, y_pred_list)
        precision = precision_score(y_true_list, y_pred_list)
        recall = recall_score(y_true_list, y_pred_list)
        f1 = f1_score(y_true_list, y_pred_list)

        # Update metrics dict
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(acc)
        self.metrics["precision"].append(precision)
        self.metrics["recall"].append(recall)
        self.metrics["f1"].append(f1)

        # log evaluation results
        configurator.logger.info(
            f"Epoch {epoch} - Validation loss: {loss:.4f}, accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f} \n"
        )

        return self.metrics

    def _check_improvement(self, epoch):
        """
        Checks if the model has improved based on the specified criterion.

        This function implements early stopping based on the following logic:
          1. Checks if the current metric value (based on the `criterion`) is lower
             than the previously recorded best score.
          2. If there's improvement, updates the best score and saves the model.
          3. If there's no improvement, checks if a certain number of epochs
             (defined by `patience`) have passed without improvement.
          4. If the patience limit is reached and no improvement is observed,
             early stopping is triggered.

        Args:
          epoch: The current epoch number.

        Returns:
          bool: True if early stopping is triggered, False otherwise.
        """
        # Raise an error if the specified criterion is not found in the metrics dictionary
        if self.criterion not in self.metrics:
            raise ValueError(
                f"Invalid criterion: {self.criterion}. Valid options are: {self.metrics.keys()}"
            )

        # Get the current metric value for the specified criterion
        current_metric = self.metrics[self.criterion][-1]
        # Check if the current metric is lower than the previously recorded best score
        if current_metric < self.best_score:
            # Update the best score with the current metric if it's an improvement
            self.best_score = current_metric
            # Save the model state dictionary to the specified model directory
            torch.save(self.model.state_dict(), self.model_dir / "model.pt")
            # Early stopping is not triggered, return False
            return False

        # Early stopping check: only perform this if we have enough epochs for comparison
        if len(self.metrics[self.criterion]) > self.patience:
            # Count the number of epochs in the last 'patience' epochs where the metric was not better than the best score
            p_count = sum(
                metric > self.best_score
                for metric in self.metrics[self.criterion][-self.patience :]
            )
            # Trigger early stopping if the number of epochs with no improvement reaches the patience limit
            if p_count >= self.patience:
                configurator.logger.info(
                    f"Early stopping triggered after {epoch+1} epochs due to no improvement on {self.criterion} in the last {self.patience} epochs."
                )
                return True
        # Early stopping is not triggered, return False
        return False
