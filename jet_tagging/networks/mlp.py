import torch
from torch import nn


class JetClassificationMLP(nn.Module):
    """
    Flexible MLP for binary jet classification.
    Supports standard, deep, or downsized architectures with optional
    dropout and batch normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, architecture: str = "standard",
                 use_dropout: bool = False, use_batchnorm: bool = False, multi: bool = False):
        """
        Initialize the JetClassificationMLP as a subclass of nn.Module
        """
        super().__init__()
        # Add architecture, dropout, and batchnorm attributes
        self.architecture = architecture
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        # Initialize list to hold layers (will be used for nn.Sequential)
        layers = []

        # Minimum hidden_dim threshold of 16, otherwise downsized-multi model will struggle
        if hidden_dim < 16:
            hidden_dim = 16
        
        # Width of last hidden layer, useful for output layer construction
        last_hidden_dim = hidden_dim

        def add_block(in_dim, out_dim):
            """
            Adds a standard MLP block: Linear → (optional BatchNorm) → ReLU → (optional Dropout).
            """
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.2))

        # Architecture variants
        # Standard: 3 hidden layers, all with hidden_dim size.
        if architecture == "standard":
            add_block(input_dim, hidden_dim)
            add_block(hidden_dim, hidden_dim)
            add_block(hidden_dim, hidden_dim)

        # Deep: 5 hidden layers, all with same width.
        elif architecture == "deep":
            add_block(input_dim, hidden_dim)
            for _ in range(4):
                add_block(hidden_dim, hidden_dim)
        
        # Downsized: gradually shrinks layer sizes → helps in overparameterized settings.
        elif architecture == "downsized":
            add_block(input_dim, hidden_dim)
            add_block(hidden_dim, hidden_dim // 2)
            add_block(hidden_dim // 2, hidden_dim // 4)
            last_hidden_dim = hidden_dim // 4

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Final output layer, sigmoid pass for binary, raw logits for multi
        if multi:
            layers.append(nn.Linear(last_hidden_dim, 3))
        else:
            layers.append(nn.Linear(last_hidden_dim, 1))
            layers.append(nn.Sigmoid())

        # Wrap all layers into a sequential model.
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines how the model processes an input batch.
        """
        return self.network(x)


class WeightedBinaryCrossEntropy(nn.Module):
    """
    Weighted BCE loss for handling class imbalance.
    """
    def __init__(self, weight_0: float, weight_1: float):
        """
        Weight_0 for class 0 (e.g., QCD), weight_1 for class 1 (e.g., Top + WZ).
        """
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Constructs a per-sample weight vector depending on each sample’s label.
        Computes binary cross-entropy using that vector:
        """
        weights = (1 - target) * self.weight_0 + target * self.weight_1
        return nn.functional.binary_cross_entropy(pred, target, weight=weights)


def weighted_bce_loss(pred, target, weight):
    """
    Binary cross-entropy loss with per-sample weights.

    This loss is used for reweighting training samples based on feature distributions
    (e.g., correcting kinematic biases using histogram-based weights like pT, eta, etc.).

    Args:
        pred (Tensor): Predicted probabilities from the model (after sigmoid), shape [B]
        target (Tensor): Ground-truth binary labels (0 or 1), shape [B]
        weight (Tensor): Per-sample scalar weights, shape [B]

    Returns:
        Tensor: Scalar loss value — weighted average Bbce over batch
    """
    # Compute binary cross-entropy for each sample (no reduction, per-sample bce)
    bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')

    # Apply per-sample weights and return the weighted mean
    return (bce * weight).mean()


class DeepSetsClassifier(nn.Module):
    """
    Deep Set classifier for constituent-level jet inputs.
    Input shape: (batch_size, n_constituents, n_features)
    """
    def __init__(self, feature_dim: int, hidden_dim: int, architecture: str = "standard",
                 use_dropout: bool = False, use_batchnorm: bool = False, multi: bool = False):
        super().__init__()

        self.multi = multi
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # Phi network: applied to each constituent individually
        # A regular mlp, one constituent is fed at a time, constituents form rows, features are columns
        # For each constituent, returns vector of size hidden_dim
        self.phi = self._build_mlp(feature_dim, hidden_dim, architecture, hidden_dim)

        # Rho network: processes the aggregated jet-level feature
        # Operates after pooling, permutation invariant constituent information is embedded
        # Each jet is now a single vector of size hidden_dim
        self.rho = self._build_mlp(hidden_dim, hidden_dim, architecture, hidden_dim, rho=True)

    def _build_mlp(self, input_dim, hidden_dim, architecture, output_dim, rho = False):
        """
        Internal helper to construct the per-constituent MLP (φ network)
        """
        layers = []

        if hidden_dim < 16:
            hidden_dim = 16

        def add_block(in_dim, out_dim):
            layers.append(nn.Linear(in_dim, out_dim))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if self.use_dropout:
                layers.append(nn.Dropout(0.2))

        # No need for extra deep configuration
        if architecture == "standard" or architecture == "deep":
            add_block(input_dim, hidden_dim)
            add_block(hidden_dim, hidden_dim)
            if rho:
                if self.multi:
                    layers.append(nn.Linear(hidden_dim, 3))
                else:
                    layers.append(nn.Linear(hidden_dim, 1))
                    layers.append(nn.Sigmoid())
            else:
                add_block(hidden_dim, output_dim)
        
        elif architecture == "downsized":
            add_block(input_dim, hidden_dim)
            add_block(hidden_dim, hidden_dim // 2)
            if rho:
                if self.multi:
                    layers.append(nn.Linear(hidden_dim // 2, 3))
                else:
                    layers.append(nn.Linear(hidden_dim // 2, 1))
                    layers.append(nn.Sigmoid())
            else:
                add_block(hidden_dim // 2, output_dim)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, n_constituents, feature_dim)
        """
        B, N, D = x.shape
        # Apply φ to each constituent: reshape to (B*N, D) → (B, N, hidden_dim)
        x_flat = x.view(B * N, D)
        phi_out = self.phi(x_flat).view(B, N, -1)

        # Aggregate constituent embeddings (mean pooling), shape: (B, hidden_dim)
        jet_embedding = torch.einsum("bnh->bh", phi_out) / phi_out.size(1)

        # Apply ρ network to classify the jet
        return self.rho(jet_embedding)
