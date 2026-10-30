import argparse
from pathlib import Path
import logging
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from dacite import from_dict
from dataclasses import dataclass
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
)

from jet_tagging import (
    BinaryJets, BinaryWeightedJets, FeatureWeightedJets,
    MultiJets, MultiWeightedJets,
    ConstituentBinaryJets, ConstituentBinaryWeightedJets, ConstituentMultiJets, ConstituentMultiWeightedJets
)
from jet_tagging.networks import (
    JetClassificationMLP,
    DeepSetsClassifier,
    build_network
)
from jet_tagging import compute_accuracy
from train import TrainingConfig, load_config, build_dataloader

# -----------------------
# Hyperparameters
# -----------------------
@dataclass
class Hyperpars:
    """
    Class representing model metadata loaded from YAML.
    Used to reconstruct model architecture during evaluation.
    """
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: str
    final: str
    architecture: str
    use_dropout: bool
    use_batchnorm: bool
    multi: bool
    use_deepset: bool

def load_hyperparameters(path: Path) -> Hyperpars:
    """
    Create and return Hyperpars object from a YAML file.
    """
    with open(path, "r") as f:
        metadata = yaml.safe_load(f)
    return from_dict(data_class=Hyperpars, data=metadata)


# -----------------------
# Extract prediction and labels
# -----------------------
def get_predictions_and_labels(model, dataloader, multi: bool):
    """
    Run the model on the test set and return:
    - y_true: Ground truth labels
    - y_pred: Predicted class labels
    - y_score: Raw output probabilities (used for ROC/AUC)

    For binary: score = sigmoid output
    For multi:  score = softmax probabilities (shape: [N, 3])
    """

    # Set model to eval mode (no dropout, no batchnorm updates)
    model.eval()
    y_true, y_pred, y_score = [], [], []

    # Disable gradient tracking for inference
    with torch.no_grad():
        # Iterate ovr mini-batches
        # x_batch: input features, shape either [B, D] or [B, C, D]
        # y_batch: ground-truth labels, shape [B]
        for x_batch, y_batch in dataloader:
            # Raw model output: logits
            outputs = model(x_batch)

            if multi:
                # Converts logits into probabilities across 3 classes.
                probs = torch.softmax(outputs, dim=1)

                # Picks the index of the highest probability class, predicted label (0, 1, or 2).
                preds = torch.argmax(probs, dim=1)
                y_score.extend(probs.numpy())  # shape: [N, 3]
            else:
                # Flattens tensor from shape [B, 1] to [B].
                probs = outputs.view(-1)
                preds = (probs >= 0.5).int()
                y_score.extend(probs.numpy())

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_score)


# -----------------------
# Best threshold for binary class labeling
# -----------------------
def find_best_threshold(y_true, y_prob):
    """
    Search over thresholds to find the one that maximizes F1 score
    for binary classification only.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold


# -----------------------
# Print metrics
# -----------------------
def print_metrics(y_true, y_pred, label=""):
    """
    Print evaluation metrics for binary classification.
    """
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    bacc = balanced_accuracy_score(y_true, y_pred)
    logging.info(f"--- {label} ---")
    logging.info(f"Precision          = {precision:.4f}")
    logging.info(f"Recall             = {recall:.4f}")
    logging.info(f"F1 Score           = {f1:.4f}")
    logging.info(f"Balanced Accuracy  = {bacc:.4f}")


# -----------------------
# Plot ROC
# -----------------------
def plot_roc(y_true, y_score, save_path):
    """
    Plot ROC curve and save to file.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------
# Plot Confusion
# -----------------------
def plot_confusion(y_true, y_pred, save_path, labels=None):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------
# Plot raw logits
# -----------------------
def plot_prediction_histogram(y_prob, save_path):
    """
    Plot distribution of model confidence scores (predicted probabilities).
    """
    plt.figure()
    plt.hist(y_prob, bins=100)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Prediction Histogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------
# AUC vs Jet Mass
# -----------------------
def plot_mass_analysis(model, data, plots_dir, frac_train, frac_val):
    """
    Run mass analysis
    """
    # Run inference again to collect scores and mass
    all_masses = []
    all_outputs = []
    all_targets = []

    # Extract unpreprocessed test_data as NDarray
    n_samples = data.shuffled_np.shape[0]
    n_train = int(n_samples * frac_train)
    n_val = int(n_samples * frac_val)
    test_data_unprocessed = data.shuffled_np[n_train + n_val:]

    # Original test_dataloader
    test_data = data.test_dataloader

    with torch.no_grad():
        for batch in test_data:
            # Unpack batch, expected shape: (features, labels)
            # Except BinaryFeatureWeightedJets, expected shape: (features, labels, weights)
            if len(batch) == 3:
                x_batch, y_batch, _ = batch
            else:
                x_batch, y_batch = batch
            
            # Shape [B]
            probs = model(x_batch).view(-1)
            all_outputs.append(probs)
            all_targets.append(y_batch)
    
    # Select original mass value, [pT, η, ϕ, mass,...]
    all_masses.append(test_data_unprocessed[:, 3])

    # Convert to full arrays, shape [n_test]
    masses = np.concatenate(all_masses)
    probs = torch.cat(all_outputs).numpy()
    targets = torch.cat(all_targets).numpy()

    # Define mass bins (0 to 600 GeV in 100 bins)
    mass_bins = np.linspace(0, 600, 101)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    binned_auc = []

    # Compute AUC for each bin
    for i in range(len(mass_bins) - 1):
        mask = (masses >= mass_bins[i]) & (masses < mass_bins[i + 1])
        # avoid bins with very few jets
        if np.sum(mask) > 10:
            auc_bin = roc_auc_score(targets[mask], probs[mask])
        else:
            auc_bin = np.nan
        binned_auc.append(auc_bin)

    # Plot
    plt.figure()
    plt.plot(bin_centers, binned_auc, marker='o')
    plt.xlabel("Jet Mass [GeV]")
    plt.ylabel("AUC")
    plt.title("Classifier AUC vs Jet Mass")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "auc_vs_mass.png")
    plt.close()
    logging.info(f"Saved AUC vs Jet Mass plot to: {plots_dir / 'auc_vs_mass.png'}")


# -----------------------
# Evaluation entry point
# -----------------------
def main():
    # Use --run_name to locate the output directory
    parser = argparse.ArgumentParser()
    ####
    parser.add_argument("--run_name", required=True, help="Run name (outputs/{run_name}/)")
    args = parser.parse_args()

    # Load model metadata
    metadata_path = Path(f"outputs/{args.run_name}/classifier/hyperparameters.yaml")
    metadata = load_hyperparameters(metadata_path)
    multi = metadata.multi

    # Load config and build appropriate dataloaders
    config_path = Path(f"outputs/{args.run_name}/config.yaml")
    config = load_config(config_path)
    dataset = build_dataloader(config, metadata.use_deepset)
    test_data = dataset.test_dataloader

    # Plot output directory
    plots_dir = Path(f"outputs/{args.run_name}/plots")

    # Setup logging to file, crucial evaluation outputs will be printed & saved
    log_path = Path(f"outputs/{args.run_name}/evaluation_logs.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
    )


    # Rebuild model and load weights
    logging.info(f"Loaded model: {metadata.architecture} (DeepSets={metadata.use_deepset}, Multi={multi})")
    model = build_network(
        input_dim=metadata.input_dim,
        hidden_dim=metadata.hidden_dim,
        architecture=metadata.architecture,
        use_dropout=metadata.use_dropout,
        use_batchnorm=metadata.use_batchnorm,
        multi=multi,
        use_deepset=metadata.use_deepset
    )
    model_path = Path(f"outputs/{args.run_name}/classifier/parameters.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if not multi:
        # Binary Classification
        accuracy = compute_accuracy(model, test_data)
        logging.info(f"Accuracy at threshold 0.5 = {accuracy:.4f}")

        y_true, _, y_prob = get_predictions_and_labels(model, test_data, multi=False)

        best_thresh = find_best_threshold(y_true, y_prob)
        logging.info(f"\nBest threshold (F1 optimized) = {best_thresh:.4f}")

        # Check class predictions with 0.5 threshold
        y_pred_default = (y_prob >= 0.5).astype(int)
        print_metrics(y_true, y_pred_default, label="Threshold = 0.5")
        plot_confusion(y_true, y_pred_default, plots_dir / "confusion_matrix_0.5.png")

        # Check class predictions with best threshold
        y_pred_best = (y_prob >= best_thresh).astype(int)
        print_metrics(y_true, y_pred_best, label=f"Threshold = {best_thresh:.4f}")
        plot_confusion(y_true, y_pred_best, plots_dir / "confusion_matrix_best.png")

        plot_roc(y_true, y_prob, plots_dir / "roc_curve.png")
        plot_prediction_histogram(y_prob, plots_dir / "prediction_hist.png")

        # Do plot_mass_analysis only on binary classification between small data.
        if isinstance(dataset, BinaryJets) or isinstance(dataset, BinaryWeightedJets) or isinstance(dataset, FeatureWeightedJets):
                plot_mass_analysis(model, dataset, plots_dir, config.frac_train, config.frac_val)

    else:
        # Multiclass Classification
        y_true, y_pred, _ = get_predictions_and_labels(model, test_data, multi=True)

        f1 = f1_score(y_true, y_pred, average="weighted")
        bacc = balanced_accuracy_score(y_true, y_pred)

        logging.info(f"\nBalanced Accuracy  = {bacc:.4f}")
        logging.info(f"Weighted F1 Score  = {f1:.4f}")

        # Confusion matrix with class labels
        plot_confusion(y_true, y_pred, plots_dir / "confusion_matrix_multiclass.png", labels=["QCD", "Top", "WZ"])

    logging.info(f"Saved evaluation plots to {plots_dir}/")

if __name__ == "__main__":
    main()