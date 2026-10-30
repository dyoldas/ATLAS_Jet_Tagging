import os
import shutil
import logging
import yaml
import torch
import argparse
from simple_parsing import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from dacite import from_dict

# Jet tagging dataset variants, networks, and training utilities
from jet_tagging import (
    BinaryJets, BinaryWeightedJets, FeatureWeightedJets,
    MultiJets, MultiWeightedJets,
    ConstituentBinaryJets, ConstituentBinaryWeightedJets, ConstituentMultiJets, ConstituentMultiWeightedJets, train_model
)

from jet_tagging.networks import (
    JetClassificationMLP,
    WeightedBinaryCrossEntropy,
    DeepSetsClassifier,
    build_network
)

# -----------------------
# CLI Argument Interface
# -----------------------
@dataclass
class TrainingCli:
    """
    Command-line interface (CLI) options for launching a training run.

    Attributes:
        run_name: Name of the run directory (inside the outputs/ folder).
        Used to save model, plots, config, etc. Default: "default_run".

        no_config_edit: If True, skips the interactive prompt to edit the YAML config.
        Useful for automated runs or scripting. Default: False.
    """
    run_name: str = "default_run"
    no_config_edit: bool = False


# -----------------------
# Configuration schema
# -----------------------
@dataclass
class TrainingConfig:
    """
    YAML-loaded training configuration for jet tagging.
    """
    dataset_type: str
    qcd_path: str
    top_path: str
    wz_path: str
    qcd_const_path: str
    top_const_path: str
    wz_const_path: str
    batch_size: int
    frac_train: float
    frac_val: float
    four_vector_cut: bool
    hidden_dim: int
    learning_rate: float
    epoch_count: int
    weight_regularization: bool
    weight_decay: float
    architecture: str
    use_dropout: bool
    use_batchnorm: bool
    threshold: float
    seed: int
    augment: bool

def load_config(path: Path) -> TrainingConfig:
    """
    Create and return TrainingConfig object from a YAML file.
    """
    # Open and parse the yaml file to nested python dictionary
    with open(path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Check if the config file is appropriate
    if "training" not in yaml_data:
        raise KeyError(f"Expected top-level key 'training'. Got keys: {list(yaml_data.keys())}")

    # Convert yaml dict and return TrainingConfig object
    return from_dict(data_class=TrainingConfig, data=yaml_data["training"])


# -----------------------
# Prepare output directory
# -----------------------
def prepare_run_directory(run_name: str, base_config_path: Path, allow_config_edit: bool = True) -> tuple[TrainingConfig, Path]:
    """
    Create a fresh output directory and copy config into it.
    If the run directory already exists, it will be deleted and recreated.
    """
    run_dir = Path("outputs") / run_name
    config_path = run_dir / "config.yaml"

    # Fully delete the existing run directory if it exists
    if run_dir.exists():
        shutil.rmtree(run_dir)

    # Create a fresh run directory
    run_dir.mkdir(parents=True)

    # Copy base config into the fresh run folder
    shutil.copy(base_config_path, config_path)

    # Optional, edit the run configuration
    if allow_config_edit:
        input(
            f"please edit the config in {config_path}"
            " to set the parameters for this run\n"
            "afterwards, please press ENTER to continue..."
        )

    # Load and return the run_dir and run configuration
    return load_config(config_path), run_dir



# -----------------------
# Build the Requested Jet Object and DataLoader
# -----------------------
def build_dataloader(config: TrainingConfig, const: bool):
    """
    Create the appropriate jet dataset wrapper based on config.
    """
    # Common arguments to build dataloader for given jet object
    if not const:
        kwargs = {
        "qcd_path": config.qcd_path,
        "top_path": config.top_path,
        "wz_path": config.wz_path,
        "batch_size": config.batch_size,
        "frac_train": config.frac_train,
        "frac_val": config.frac_val,
        "seed": config.seed
        }
    
    else:
        kwargs = {
        "qcd_const_path": config.qcd_const_path,
        "top_const_path": config.top_const_path,
        "wz_const_path": config.wz_const_path,
        "qcd_small_path": config.qcd_path,
        "top_small_path": config.top_path,
        "wz_small_path": config.wz_path,
        "batch_size": config.batch_size,
        "frac_train": config.frac_train,
        "frac_val": config.frac_val,
        "seed": config.seed,
        "augment": config.augment
        }

    # Check and return the appropriate jet object requested
    match config.dataset_type:
        case "BinaryJets":
            return BinaryJets(**kwargs)
        case "BinaryWeightedJets":
            return BinaryWeightedJets(**kwargs, four_vector_cut=config.four_vector_cut)
        case "FeatureWeightedJets":
            return FeatureWeightedJets(**kwargs)
        case "MultiJets":
            return MultiJets(**kwargs)
        case "MultiWeightedJets":
            return MultiWeightedJets(**kwargs, four_vector_cut=config.four_vector_cut)
        case "ConstituentBinaryJets":
            return ConstituentBinaryJets(**kwargs)
        case "ConstituentBinaryWeightedJets":
            return ConstituentBinaryWeightedJets(**kwargs)
        case "ConstituentMultiJets":
            return ConstituentMultiJets(**kwargs)
        case "ConstituentMultiWeightedJets":
            return ConstituentMultiWeightedJets(**kwargs)
        case other:
            raise ValueError(f"Unsupported dataset_type: {other}")


# -----------------------
# Loss and Optimizer setup
# -----------------------
def build_loss_and_optimizer(model, data_obj, config: TrainingConfig):
    """
    Select appropriate loss function and optimizer based on dataset and config.
    Supports optional weight regularization via AdamW.
    """
    # Select loss function wrt given jet class
    if isinstance(data_obj, BinaryJets):
        loss_fn = torch.nn.BCELoss()
    elif isinstance(data_obj, BinaryWeightedJets):
        loss_fn = WeightedBinaryCrossEntropy(data_obj.weight_0, data_obj.weight_1)
    elif isinstance(data_obj, FeatureWeightedJets):
        loss_fn = torch.nn.BCELoss()
    elif isinstance(data_obj, MultiJets):
        loss_fn = torch.nn.CrossEntropyLoss()
    elif isinstance(data_obj, MultiWeightedJets):
        loss_fn = torch.nn.CrossEntropyLoss(weight=data_obj.class_weights)
    elif isinstance(data_obj, ConstituentBinaryJets):
        loss_fn = torch.nn.BCELoss()
    elif isinstance(data_obj, ConstituentBinaryWeightedJets):
        loss_fn = WeightedBinaryCrossEntropy(data_obj.weight_0, data_obj.weight_1)
    elif isinstance(data_obj, ConstituentMultiJets):
        loss_fn = torch.nn.CrossEntropyLoss()
    elif isinstance(data_obj, ConstituentMultiWeightedJets):
        loss_fn = torch.nn.CrossEntropyLoss(weight=data_obj.class_weights)
    else:
        raise TypeError("Unknown dataset class")

    # Select optimizer, optional weight regularization
    if config.weight_regularization:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return loss_fn, optimizer


# -----------------------
# Save model metadata
# -----------------------
def save_hyperparameters(model_dir: Path, input_dim: int, hidden_dim: int, architecture: str, use_dropout: bool, use_batchnorm: bool, multi: bool, deep: bool):
    """
    Store model architecture information in a YAML file.
    """
    metadata = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": 3 if multi else 1,
        "activation": "ReLU",
        "final": "Softmax" if multi else "Sigmoid",
        "architecture": architecture,
        "use_dropout": use_dropout,
        "use_batchnorm": use_batchnorm,
        "multi": multi,
        "use_deepset": deep
    }
    with open(model_dir / "hyperparameters.yaml", "w") as f:
        yaml.dump(metadata, f)


# -----------------------
# Training entry point
# -----------------------
def main():
    logging.basicConfig(level=logging.INFO)

    # Get the run_name from user and also load the default config path
    parser = ArgumentParser()
    parser.add_arguments(TrainingCli, dest="cli")
    args = parser.parse_args()
    cli: TrainingCli = args.cli
    base_config_path = Path("config_default.yaml")

    # Load configuration and prepare output directory
    config, run_dir = prepare_run_directory(cli.run_name, base_config_path, allow_config_edit=not cli.no_config_edit)

    # Determine if constituent data
    const = False
    if config.dataset_type.startswith("Con"):
        const = True

    # Load jet dataset and dataloader
    logging.info("Loading dataset...")
    data = build_dataloader(config, const)

    if not const:
        # Input dim, all columns except label
        input_dim = data.train.shape[1] - 1
    else:
        input_dim = 5 if config.augment else 4

    logging.info(f"Data: {data.__class__.__name__}")
    logging.info(f"Train: {len(data.train):,} | Val: {len(data.val):,} | Test: {len(data.test):,} | Seed: {config.seed} | Features: {input_dim}")

    # Choose binary or multi classification
    multi = False
    if isinstance(data, MultiJets) or isinstance(data, MultiWeightedJets) or isinstance(data, ConstituentMultiJets) or isinstance(data, ConstituentMultiWeightedJets):
        multi = True
        logging.info("Building multiclassifier")
    else:
        logging.info("Building binary classifier")
    
    # Build model
    logging.info("Building the model with following dimensions")
    logging.info(f"Architecture: {config.architecture}, input_dim: {input_dim}, hidden_dim: {config.hidden_dim}")
    if const:
        logging.info("DeepSet classifier for constituent data")
    else:
        logging.info("Regular mlp classifier")
    
    model = build_network(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            architecture=config.architecture,
            use_dropout=config.use_dropout,
            use_batchnorm=config.use_batchnorm,
            multi=multi,
            use_deepset=const
        )

    # Setup loss function and optimizer
    loss_fn, optimizer = build_loss_and_optimizer(model, data, config)

    # Define appropriate output paths
    plots_dir = run_dir / "plots"
    model_dir = run_dir / "classifier"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train the model using early stopping and checkpointing
    logging.info("Training model...")
    summary = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=data.train_dataloader,
        val_loader=data.val_dataloader,
        epochs=config.epoch_count,
        plot_path=plots_dir / "training_summary.png",
        checkpoint_path=model_dir / "parameters.pth",
        early_stopping_patience=10,
        early_stopping_delta=0.0,
        threshold=config.threshold,
        multi=multi
    )

    # Save training metadata
    save_hyperparameters(
        model_dir=model_dir,
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        architecture=config.architecture,
        use_dropout=config.use_dropout,
        use_batchnorm=config.use_batchnorm,
        multi=multi,
        deep=const
    )
    logging.info(f"Saved training plot → {plots_dir / 'training_summary.png'}")
    logging.info(f"Saved best model → {model_dir / 'parameters.pth'}")
    logging.info(f"Saved model metadata → {model_dir / 'hyperparameters.yaml'}")


if __name__ == "__main__":
    main()
