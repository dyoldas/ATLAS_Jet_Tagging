import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Callable

from jet_tagging.data import (
    BinaryJets, BinaryWeightedJets, FeatureWeightedJets,
    MultiJets, MultiWeightedJets,
    ConstituentBinaryJets, ConstituentBinaryWeightedJets, ConstituentMultiJets, ConstituentMultiWeightedJets
)
from jet_tagging.networks import weighted_bce_loss

# Utility class for checkpointing and saving the best model
class ModelCheckpoint:
    def __init__(self, model_path: str = "best_model.pth"):
        """
        Initializes loss and save path.
        """
        self.best_loss = float("inf")
        self.model_path = model_path

    def check(self, model: Module, loss: float):
        """
        Saves model only if validation loss improves.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(model.state_dict(), self.model_path)
            print(f"New best model saved → {self.model_path} (val loss: {loss:.4f})")

# Utility class for early stoppage
class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initializes patience, min_delta, counter, val loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, val_loss: float) -> bool:
        """
        Keeps training going unless val loss fails to improve beyond min_delta for patience epochs.
        """
        if (val_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TrainingSummary:
    """
    Stores and visualizes training/validation loss and accuracy.
    Prints metrics every N epochs and generates a plot after training.
    """
    def __init__(self, print_every: int = 1):
        """
        Initializes metric lists and epoch counter.
        """
        self.print_every = print_every
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def update(self, train_loss, train_acc, val_loss, val_acc):
        """
        Appends epoch metrics and prints progress if due.
        """
        if (self.epoch + 1) % self.print_every == 0:
            print(f"Epoch {self.epoch + 1} | "
                  f"Train Loss: {train_loss:.3e}, Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.3e}, Acc: {val_acc*100:.2f}%")
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.epoch += 1

    def plot(self, path: Path):
        """
        Save loss/accuracy curves to the given path.
        """
        # Initialize figure with one main axis, epochs as x-axis
        fig, ax1 = plt.subplots()
        epochs = range(self.epoch)

        # Plots training loss and validation loss on the primary y-axis (ax1).
        ax1.plot(epochs, self.train_losses, label="Train Loss", color="C0")
        ax1.plot(epochs, self.val_losses, label="Val Loss", color="C1")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")

        # Creates a second y-axis (ax2) that shares the same x-axis.
        ax2 = ax1.twinx()

        # Overlay accuracy on the same plot without mixing y-scales.
        ax2.plot(epochs, self.train_accuracies, linestyle="--", label="Train Acc", color="C0")
        ax2.plot(epochs, self.val_accuracies, linestyle="--", label="Val Acc", color="C1")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(path)


def compute_epoch_loss(model: Module, dataloader: DataLoader, loss_fn: Callable[[Tensor, Tensor], Tensor], optimizer: Optimizer = None) -> float:
    """
    Runs one epoch over the data and returns the average loss.
    If optimizer is provided, performs training; otherwise, evaluates.
    Supports optional per-sample weights if the dataloader yields (x, y, w) tuples.
    """
    # Initializes the running total for the sum of all batch losses.
    total_loss = 0.0

    # Loop over the entire dataset, broken into mini-batches.
    for batch in dataloader:
        # Unpack batch → (features, labels) or (features, labels, weights)
        if len(batch) == 3:
            features, labels, weights = batch
            output = model(features)
            loss = weighted_bce_loss(output, labels, weights)
        else:
            features, labels = batch
            output = model(features)
            loss = loss_fn(output, labels)

        # Clear & propogate gradients, update the weights
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Accumulate losses per batch
        total_loss += loss.item()

    # Return the average loss per batch.
    return total_loss / len(dataloader)


def compute_accuracy(model: Module, dataloader: DataLoader, threshold: float = 0.5, multi = False) -> float:
    """
    Computes accuracy for binary classification by thresholding sigmoid output.
    """
    # Evaluation mode
    model.eval()

    # Initialize correct and total counts
    correct = 0
    total = 0

    # Loop over the entire dataset, broken into mini-batches.
    for batch in dataloader:
        # Unpack batch → (features, labels) or (features, labels, weights)
        if len(batch) == 3:
            features, labels, weights = batch
        else:
            features, labels = batch
        
        output = model(features)
        if multi:
            # Raw predictions, no threshold, shape: [N]
            predicted = torch.argmax(output, dim=1)
        else:
            # Flatten prediction tensor [N, 1] to [N]
            output = output.view(-1)
        
            # Creates and converts Boolean tensor to integer values (True → 1, False → 0)
            predicted = (output >= threshold).long()

        # Counts number of true predictions, converts to scalar, adds to correct counter
        correct += (predicted == labels.view(-1)).sum().item()

        # Adds number of samples in the batch to the total counter.
        total += labels.size(0)
    
    # Return accuracy ratio
    return correct / total


def train_model(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    plot_path: Path,
    checkpoint_path: Path = Path("best_model.pth"),
    early_stopping_patience: int = 5,
    early_stopping_delta: float = 0.0,
    threshold: float = 0.5,
    multi: bool = False
):
    """
    Full training loop with metric tracking and final plot.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        loss_fn: binary loss function
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        epochs: number of epochs
        plot_path: where to save the training curve
        checkpoint_path: where to save the best model
        early_stopping_patience: patience on validation loss
        early_stopping_delta: delta between val losses
        threshold: labeling threshold for calculating accuracies
        multi: Multiclassification task or not

    Returns:
        TrainingSummary object with full metric history
    """
    # Build utility objects
    summary = TrainingSummary()
    check_pointer = ModelCheckpoint(model_path=checkpoint_path)
    early_stopper = EarlyStopper(patience=early_stopping_patience, min_delta=early_stopping_delta)

    # Perform training loop for given number of epochs
    for _ in range(epochs):
        # Training mode
        model.train()
        train_loss = compute_epoch_loss(model, train_loader, loss_fn, optimizer)
        train_acc = compute_accuracy(model, train_loader, threshold, multi=multi)

        # Evaluation mode, disable gradient tracking for feasibility
        model.eval()
        with torch.no_grad():
            val_loss = compute_epoch_loss(model, val_loader, loss_fn)
            val_acc = compute_accuracy(model, val_loader, threshold, multi=multi)

        # Update training summary
        summary.update(train_loss, train_acc, val_loss, val_acc)
        
        # Check & save the best model, check early stoppage
        check_pointer.check(model, val_loss)
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {summary.epoch}")
            break

    # Save plot and return the summary object
    summary.plot(plot_path)
    return summary
