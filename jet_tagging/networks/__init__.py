"""
Network submodule for jet classification tasks.

Exposes:
- JetClassificationMLP: Flexible feedforward classifier (binary or multiclass)
- DeepSetsClassifier: Deep Sets model for constituent-level jet classification
- WeightedBinaryCrossEntropy: Class-reweighted BCE loss for imbalanced binary problems
- weighted_bce_loss: Per-smple feature-reweighted BCE loss for imbalanced binary problems
- build_network: Factory function for clean model construction
"""

from jet_tagging.networks.mlp import (
    JetClassificationMLP,
    WeightedBinaryCrossEntropy,
    weighted_bce_loss,
    DeepSetsClassifier,
)


def build_network(input_dim: int,
                  hidden_dim: int,
                  architecture: str = "standard",
                  use_dropout: bool = False,
                  use_batchnorm: bool = False,
                  multi: bool = False,
                  use_deepset: bool = False):
    """
    Factory function to build the appropriate jet classification model.

    Args:
        input_dim (int): Input feature dimension per jet or constituent.
        hidden_dim (int): Hidden layer width.
        architecture (str): {"standard", "deep", "downsized"}.
        use_dropout (bool): Apply dropout after activations.
        use_batchnorm (bool): Apply batch normalization.
        multi (bool): Enable multiclass classification (QCD, Top, WZ).
        use_deepset (bool): If True, instantiate DeepSetsClassifier (for constituent inputs).

    Returns:
        nn.Module: Instantiated model (MLP or DeepSets).
    """
    if use_deepset:
        return DeepSetsClassifier(
            feature_dim=input_dim,
            hidden_dim=hidden_dim,
            architecture=architecture,
            use_dropout=use_dropout,
            use_batchnorm=use_batchnorm,
            multi=multi
        )
    else:
        return JetClassificationMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            architecture=architecture,
            use_dropout=use_dropout,
            use_batchnorm=use_batchnorm,
            multi=multi
        )


__all__ = [
    "JetClassificationMLP",
    "DeepSetsClassifier",
    "WeightedBinaryCrossEntropy",
    "weighted_bce_loss",
    "build_network",
]
