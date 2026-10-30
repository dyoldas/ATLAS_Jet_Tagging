import pandas as pd
import numpy as np
import h5py
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, TensorDataset


class BinaryJets:
    """
    Loads, preprocesses, and provides DataLoader objects for binary jet classification tasks.

    This class supports:
    - Loading QCD, Top, and WZ jet data from HDF5 files
    - Cleaning data (handling NaNs and infs)
    - Labeling QCD class as 0 and other Top & WZ as 1
    - Splitting into train/validation/test sets
    - Standardizing features based on training statistics
    - Constructing PyTorch DataLoaders for model training and evaluation
    """
    def __init__(self, qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed):
        """
        Load, clean, preprocess, and split binary jet dataset into
        training, validation, and test sets. Provides DataLoader utilities.

        Args:
            qcd_path (str): Path to QCD jets HDF5 file.
            top_path (str): Path to Top jets HDF5 file.
            wz_path (str): Path to WZ jets HDF5 file.
            batch_size (int): PyTorch batch size.
            frac_train (float): Fraction of data used for training.
            frac_val (float): Fraction of data used for validation.
            seed (int): random shuffling seed
        """
        self.batch_size = batch_size

        # Load and clean data
        qcd_df = self.clean_data(pd.read_hdf(qcd_path))
        top_df = self.clean_data(pd.read_hdf(top_path))
        wz_df = self.clean_data(pd.read_hdf(wz_path))

        # Assign labels: QCD = 0, Top & WZ = 1
        qcd_df["label"] = 0
        top_df["label"] = 1
        wz_df["label"] = 1

        self.qcd_df = qcd_df
        self.top_df = top_df
        self.wz_df = wz_df

        self.qcd_np = qcd_df.to_numpy()
        self.top_np = top_df.to_numpy()
        self.wz_np = wz_df.to_numpy()

        # Combine and shuffle with given seed
        combined_df = pd.concat([qcd_df, top_df, wz_df], ignore_index=True)
        self.shuffled_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.shuffled_np = self.shuffled_df.to_numpy()

        # Split into train/val/test
        n_samples = self.shuffled_np.shape[0]
        n_train = int(n_samples * frac_train)
        n_val = int(n_samples * frac_val)
        self.train = self.shuffled_np[:n_train]
        self.val = self.shuffled_np[n_train:n_train + n_val]
        self.test = self.shuffled_np[n_train + n_val:]

        # Standardize features
        self.train, self.val, self.test = self.jet_preprocessor(self.train, self.val, self.test)

        # DataLoaders — shuffle training batches
        self.train_dataloader = self.get_dataloader(*self.get_tensors(self.train))

        # Validation and test DataLoaders are not used to train model weights.
        # The validation set is used for monitoring during training (e.g., early stopping).
        # To ensure reproducibility in evaluate.py, we disable batch shuffling.
        self.val_dataloader = self.get_dataloader(*self.get_tensors(self.val), shuffle=False)
        self.test_dataloader = self.get_dataloader(*self.get_tensors(self.test), shuffle=False)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces inf values with NaNs and drops all rows containing NaNs.
        """
        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(inplace=True)
        return data

    def jet_preprocessor(self, train, val, test):
        """
        Standardizes features (columns except last) using training set statistics.

        Args:
            train, val, test (NDArray): Arrays of shape (N, D+1), with last column = label.

        Returns:
            Tuple of standardized (train, val, test) arrays.
        """
        train_x, train_y = train[:, :-1], train[:, -1:]
        val_x, val_y = val[:, :-1], val[:, -1:]
        test_x, test_y = test[:, :-1], test[:, -1:]

        scaler = StandardScaler()
        scaler.fit(train_x)

        train_proc = np.hstack([scaler.transform(train_x), train_y])
        val_proc = np.hstack([scaler.transform(val_x), val_y])
        test_proc = np.hstack([scaler.transform(test_x), test_y])

        return train_proc, val_proc, test_proc

    def get_tensors(self, data):
        """
        Converts a NumPy array into PyTorch float tensors for features and labels.

        Returns:
            Tuple[Tensor, Tensor]: features (N, D), labels (N, 1)
        """
        x = torch.tensor(data[:, :-1], dtype=torch.float32)
        y = torch.tensor(data[:, -1], dtype=torch.float32).reshape(-1, 1)
        return x, y

    def get_dataloader(self, features, labels, shuffle=True):
        """
        Creates a PyTorch DataLoader from features and labels.
        """
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


class BinaryWeightedJets(BinaryJets):
    def __init__(self, qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed, four_vector_cut=False):
        """
        Dataset class with class-level reweighting for binary classification.
        Optionally removes first 4 features.

        Args:
            four_vector_cut (bool): If True, removes the first four features (e.g., pT, η, ϕ, mass) from all splits before training.
        """
        super().__init__(qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed)

        # Compute inverse-frequency class weights
        # Smaller class gets a higher weight.
        # QCD gets weight_0, Top & WZ gets weight_1
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=self.shuffled_np[:, -1])
        self.weight_0, self.weight_1 = class_weights

        # Optionally drop first 4 features (e.g., pT, eta, phi)
        if four_vector_cut:
            self.train = self.train[:, 4:]
            self.val = self.val[:, 4:]
            self.test = self.test[:, 4:]

        # Rebuild the DataLoaders
        self.train_dataloader = self.get_dataloader(*self.get_tensors(self.train))
        self.val_dataloader = self.get_dataloader(*self.get_tensors(self.val), shuffle=False)
        self.test_dataloader = self.get_dataloader(*self.get_tensors(self.test), shuffle=False)


class FeatureWeightedJets(BinaryJets):
    def __init__(self, qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed):
        """
        Dataset class with per-feature histogram-based reweighting.
        Useful for decorrelation and feature-based balancing.

        Computes QCD vs (Top+WZ) ratios for kinematic features.
        """
        self.batch_size = batch_size
        
        # Can't call the parent constructor, it uses overriden get_tensors method of the child
        # Just do what's been done in BinaryJets already
        qcd_df = self.clean_data(pd.read_hdf(qcd_path))
        top_df = self.clean_data(pd.read_hdf(top_path))
        wz_df = self.clean_data(pd.read_hdf(wz_path))
        qcd_df["label"] = 0
        top_df["label"] = 1
        wz_df["label"] = 1
        self.qcd_df = qcd_df
        self.top_df = top_df
        self.wz_df = wz_df
        self.qcd_np = qcd_df.to_numpy()
        self.top_np = top_df.to_numpy()
        self.wz_np = wz_df.to_numpy()
        combined_df = pd.concat([qcd_df, top_df, wz_df], ignore_index=True)
        self.shuffled_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.shuffled_np = self.shuffled_df.to_numpy()
        n_samples = self.shuffled_np.shape[0]
        n_train = int(n_samples * frac_train)
        n_val = int(n_samples * frac_val)
        self.train = self.shuffled_np[:n_train]
        self.val = self.shuffled_np[n_train:n_train + n_val]
        self.test = self.shuffled_np[n_train + n_val:]
        self.train, self.val, self.test = self.jet_preprocessor(self.train, self.val, self.test)

        # Split training data by label
        qcd = self.train[self.train[:, -1] == 0]
        signal = self.train[self.train[:, -1] == 1]

        # Feature-level weights
        self.pt_weight = self.feature_weighter(qcd, signal, self.train, feature_idx=0)
        self.eta_weight = self.feature_weighter(qcd, signal, self.train, feature_idx=1)
        self.phi_weight = self.feature_weighter(qcd, signal, self.train, feature_idx=2)
        self.mass_weight = self.feature_weighter(qcd, signal, self.train, feature_idx=3)

        # Each weight reflects bias correction for a single feature
        # Averaging them creates a composite correction that penalizes all kinematic shortcuts simultaneously.
        self.sample_weights = (self.pt_weight + self.eta_weight + self.phi_weight + self.mass_weight) / 4.0
        self.weights_tensor = torch.tensor(self.sample_weights, dtype=torch.float32)

        # Rebuild the DataLoaders, no need for weights in val/test datalaoders
        self.train_dataloader = self.get_dataloader(*self.get_tensors(self.train, self.weights_tensor, with_weights=True))
        self.val_dataloader = self.get_dataloader(*self.get_tensors(self.val, self.weights_tensor, with_weights=False), shuffle=False)
        self.test_dataloader = self.get_dataloader(*self.get_tensors(self.test, self.weights_tensor, with_weights=False), shuffle=False)

    def get_tensors(self, array, weights, with_weights=False):
        """
        Converts ndarray into PyTorch tensors.
        Args:
            array (ndarray): Full dataset split
            with_weights (bool): If True, returns sample weights as third tensor
        Returns:
            tuple: (features, labels, weights) or (features, labels)
        """
        x = torch.tensor(array[:, :-1], dtype=torch.float32)
        y = torch.tensor(array[:, -1], dtype=torch.float32).reshape(-1,1)
        if with_weights:
            if array.shape[0] != weights.shape[0]:
                raise ValueError("Sample weights size mismatch.")
            return x, y, weights
        return x, y

    def get_dataloader(self, x, y, weights=None, shuffle=True):
        """
        Builds DataLoader from tensor batches. If weights provided, yield 3-tuples.
        """
        if weights is not None:
            dataset = torch.utils.data.TensorDataset(x, y, weights)
        else:
            dataset = torch.utils.data.TensorDataset(x, y)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def feature_weighter(self, qcd_data, signal_data, full_data, feature_idx, n_bins=100, min_val=-8, max_val=8):
        """
        Computes histogram-based per-sample weights from QCD/Top ratios.

        Args:
            qcd_data (NDArray): Subset with QCD label.
            signal_data (NDArray): Subset with Top + WZ label.
            full_data (NDArray): Entire training set.
            feature_idx (int): Feature column to reweight by.

        Returns:
            NDArray: Weight array, same length as full_data.

        Note: Full logic is explained in data_analysis.ipynb
        """
        qcd_vals = np.clip(qcd_data[:, feature_idx], min_val, max_val)
        signal_vals = np.clip(signal_data[:, feature_idx], min_val, max_val)
        bins = np.linspace(min_val, max_val, n_bins + 1)

        qcd_counts, _ = np.histogram(qcd_vals, bins)
        signal_counts, _ = np.histogram(signal_vals, bins)
        ratio = np.divide(qcd_counts, signal_counts, out=np.zeros_like(qcd_counts, dtype=float), where=signal_counts != 0)

        features = np.clip(full_data[:, feature_idx], min_val, max_val)
        labels = full_data[:, -1]
        bin_indices = np.digitize(features, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(ratio) - 1)

        weights = np.where(labels == 1, 1.0, ratio[bin_indices])
        weights /= weights.mean()
        return weights


class MultiJets:
    """
    Loads, preprocesses, and provides DataLoader objects for multi jet classification tasks.

    This class supports:
    - Loading QCD, Top, and WZ jet data from HDF5 files
    - Cleaning data (handling NaNs and infs)
    - Labeling QCD as 0, Top as 1, and WZ as 2
    - Splitting into train/validation/test sets
    - Standardizing features based on training statistics
    - Constructing PyTorch DataLoaders for model training and evaluation
    
    Note: Same logical structure with binary versions, new changes are commented.
    """
    def __init__(self, qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed):
        """
        Load, clean, preprocess, and split multi jet dataset into
        training, validation, and test sets. Provides DataLoader utilities.

        New Arg:
            wz_path (str): Path speciifcally to WZ jets HDF5 file.
        """
        self.batch_size = batch_size

        qcd_df = self.clean_data(pd.read_hdf(qcd_path))
        top_df = self.clean_data(pd.read_hdf(top_path))
        wz_df = self.clean_data(pd.read_hdf(wz_path))

        # Assign labels: QCD = 0, Top = 1, WZ = 2
        qcd_df["label"] = 0
        top_df["label"] = 1
        wz_df["label"] = 2

        self.qcd_df = qcd_df
        self.top_df = top_df
        self.wz_df = wz_df

        self.qcd_np = qcd_df.to_numpy()
        self.top_np = top_df.to_numpy()
        self.wz_np = wz_df.to_numpy()

        combined_df = pd.concat([qcd_df, top_df, wz_df], ignore_index=True)
        self.shuffled_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.shuffled_np = self.shuffled_df.to_numpy()

        n_samples = self.shuffled_np.shape[0]
        n_train = int(n_samples * frac_train)
        n_val = int(n_samples * frac_val)
        self.train = self.shuffled_np[:n_train]
        self.val = self.shuffled_np[n_train:n_train + n_val]
        self.test = self.shuffled_np[n_train + n_val:]

        self.train, self.val, self.test = self.jet_preprocessor(self.train, self.val, self.test)

        self.train_dataloader = self.get_dataloader(*self.get_tensors(self.train))
        self.val_dataloader = self.get_dataloader(*self.get_tensors(self.val), shuffle=False)
        self.test_dataloader = self.get_dataloader(*self.get_tensors(self.test), shuffle=False)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(inplace=True)
        return data

    def jet_preprocessor(self, train, val, test):
        train_x, train_y = train[:, :-1], train[:, -1:]
        val_x, val_y = val[:, :-1], val[:, -1:]
        test_x, test_y = test[:, :-1], test[:, -1:]

        scaler = StandardScaler()
        scaler.fit(train_x)

        train_proc = np.hstack([scaler.transform(train_x), train_y])
        val_proc = np.hstack([scaler.transform(val_x), val_y])
        test_proc = np.hstack([scaler.transform(test_x), test_y])

        return train_proc, val_proc, test_proc

    def get_tensors(self, data):
        """
        Converts a NumPy array into PyTorch float & long tensors for features and labels.

        Returns:
            Tuple[Tensor, Tensor]: features (N, D), labels (N,)
        """
        x = torch.tensor(data[:, :-1], dtype=torch.float32)

        # long datatype for labels for CrossEntropyLoss
        y = torch.tensor(data[:, -1], dtype=torch.long)
        return x, y

    def get_dataloader(self, features, labels, shuffle=True):
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


class MultiWeightedJets(MultiJets):
    def __init__(self, qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed, four_vector_cut=False):
        """
        Dataset class with class-level reweighting for multiclassification.
        Optionally removes first 4 features (e.g., four-vector info).
        """
        super().__init__(qcd_path, top_path, wz_path, batch_size, frac_train, frac_val, seed)

        # Compute balanced class weights for multiclass
        labels = self.shuffled_np[:, -1].astype(int)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if four_vector_cut:
            self.train = self.train[:, 4:]
            self.val = self.val[:, 4:]
            self.test = self.test[:, 4:]

        self.train_dataloader = self.get_dataloader(*self.get_tensors(self.train))
        self.val_dataloader = self.get_dataloader(*self.get_tensors(self.val), shuffle=False)
        self.test_dataloader = self.get_dataloader(*self.get_tensors(self.test), shuffle=False)


class ConstituentBinaryJets:
    """
    Dataset class for binary classification using constituent-level input.
    Accepts preprocessed numpy arrays of shape (n_jets, 20, 4) with labels.
    """
    def __init__(self, qcd_const_path, top_const_path, wz_const_path, qcd_small_path, top_small_path, wz_small_path, batch_size, frac_train, frac_val, seed, augment = True):
        self.batch_size = batch_size

        # Load the data, (its already clean, see data_analysis.ipynb for more info)
        self.qcd_const = self.load_constituents(qcd_const_path)
        self.top_const = self.load_constituents(top_const_path)
        self.wz_const = self.load_constituents(wz_const_path)

        # Augment if true
        if augment:
            # add energy feautre to each constituent, new shape (n_jet, 20, 5)
            # [pt, eta, phi, mass, energy]
            qcd_const_5 = self.add_energy(self.qcd_const)
            top_const_5 = self.add_energy(self.top_const)
            wz_const_5 = self.add_energy(self.wz_const)

            # Load original small data to use in centering
            qcd_df = self.clean_data(pd.read_hdf(qcd_small_path))
            top_df = self.clean_data(pd.read_hdf(top_small_path))
            wz_df = self.clean_data(pd.read_hdf(wz_small_path))

            # Extract eta and phi columns as NDArray
            qcd_eta = qcd_df["eta"].to_numpy()
            qcd_phi = qcd_df["phi"].to_numpy()

            top_eta = top_df["eta"].to_numpy()
            top_phi = top_df["phi"].to_numpy()

            wz_eta = wz_df["eta"].to_numpy()
            wz_phi = wz_df["phi"].to_numpy()

            # Center eta and phi features
            qcd_const_5c = self.center_constituents(qcd_const_5, qcd_eta, qcd_phi)
            top_const_5c = self.center_constituents(top_const_5, top_eta, top_phi)
            wz_const_5c = self.center_constituents(wz_const_5, wz_eta, wz_phi)

            # Copy them to original tensors
            self.qcd_const = qcd_const_5c.copy()
            self.top_const = top_const_5c.copy()
            self.wz_const = wz_const_5c.copy()

        # Create the label vector: QCD=0, Top & WZ=0, shape: (n_jet, 1)
        qcd_labels = np.zeros((self.qcd_const.shape[0], 1))
        top_labels = np.ones((self.top_const.shape[0], 1))
        wz_labels = np.ones((self.wz_const.shape[0], 1))
 
        # qcd_labels[:, None, :]: shape (n_qcd, 1, 1)
        # Broadcast from (n_qcd, 1, 1) to (n_qcd, 20, 1)
        # Final rank-3 tensor with jet labels
        # [pt, eta, phi, mass, energy, label] or [pt, eta, phi, mass, label]
        qcd_labels_tiled = np.tile(qcd_labels[:, None, :], (1, self.qcd_const.shape[1], 1))
        top_labels_tiled = np.tile(top_labels[:, None, :], (1, self.top_const.shape[1], 1))
        wz_labels_tiled = np.tile(wz_labels[:, None, :], (1, self.wz_const.shape[1], 1))
        # Concatenate along the last axis → now shapes match perfectly
        qcd_labeled = np.concatenate([self.qcd_const, qcd_labels_tiled], axis=2)
        top_labeled = np.concatenate([self.top_const, top_labels_tiled], axis=2)
        wz_labeled = np.concatenate([self.wz_const, wz_labels_tiled], axis=2)


        # Combine qcd & top & wz constituent datasets and shuffle
        # Resulting shape (n_qcd + n_top + n_wz, 20, 5 or 6)
        all_data = np.concatenate([qcd_labeled, top_labeled, wz_labeled], axis=0)

        # Creates a NumPy random number generator instance (rng) using the default_rng() interface with seed.
        # Creates a reproducable shuffle with given seed
        rng = np.random.default_rng(seed)
        rng.shuffle(all_data)

        # Split
        n_total = all_data.shape[0]
        n_train = int(n_total * frac_train)
        n_val = int(n_total * frac_val)

        self.train = all_data[:n_train]
        self.val = all_data[n_train:n_train + n_val]
        self.test = all_data[n_train + n_val:]

        # Also store per-jet label arrays for direct access if needed, shape: (n_jet,)
        self.train_labels = self.train[:, 0, -1].astype(int)
        self.val_labels = self.val[:, 0, -1].astype(int)
        self.test_labels = self.test[:, 0, -1].astype(int)

        # Build the DataLoaders
        self.train_dataloader = self.get_dataloader(self.train)
        self.val_dataloader = self.get_dataloader(self.val, shuffle=False)
        self.test_dataloader = self.get_dataloader(self.test, shuffle=False)

    def load_constituents(self, path):
        """
        Load constitents file and return data as NDArray
        """
        with h5py.File(path, 'r') as f:
            return f["constituents"][:]
        
    def add_energy(self, constituents: np.ndarray) -> np.ndarray:
        """
        Adds energy as a fifth feature: E = pT * cosh(η)
        """
        pt = constituents[:, :, 0]
        eta = constituents[:, :, 1]
        E = pt * np.cosh(eta)

        # Reshape E, concatenate from last indices 
        return np.concatenate([constituents, E[..., None]], axis=-1)
    
    def center_constituents(self, constituents: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> np.ndarray:
        """
        Centers η and ϕ of constituents around their corresponding jet axes.
        Returning new array with centered η and wrapped Δϕ.
        """
        centered = constituents.copy()

        # Center η
        centered[:, :, 1] = jet_eta[:, np.newaxis] - centered[:, :, 1]

        # Center ϕ with wrap-around at ±π
        dphi = jet_phi[:, np.newaxis] - centered[:, :, 2]
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dphi = np.where(dphi >  np.pi, dphi - 2*np.pi, dphi)
        centered[:, :, 2] = dphi

        return centered
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces inf values with NaNs and drops all rows containing NaNs.
        """
        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(inplace=True)
        return data

    def get_dataloader(self, data, shuffle=True):
        """
        Creates a PyTorch DataLoader from features and labels.
        """
        # Torch tensor of features, shape: (n_jet, 20, 4 or 5)
        x = torch.tensor(data[:, :, :-1], dtype=torch.float32)

        # Torch tensor of labels, shape: (n_jet, 1)
        y = torch.tensor(data[:, 0, -1], dtype=torch.float32).unsqueeze(1)

        # Bundle in torch dataset, wrap and return the DataLoader object
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

class ConstituentBinaryWeightedJets(ConstituentBinaryJets):
    """
    Dataset class with class-level reweighting for binary classification of contituent data.
    """
    def __init__(self, qcd_const_path, top_const_path, wz_const_path, qcd_small_path, top_small_path, wz_small_path, batch_size, frac_train, frac_val, seed, augment=True):
        """
        Calls parent constructor, additional computation of class weights
        """
        super().__init__(qcd_const_path, top_const_path, wz_const_path, qcd_small_path, top_small_path, wz_small_path, batch_size, frac_train, frac_val, seed, augment)

        # Compute class weights based on total label counts
        labels = np.concatenate([
            np.zeros(len(self.qcd_const)),
            np.ones(len(self.top_const)),
            np.ones(len(self.wz_const))
        ])
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
        self.weight_0, self.weight_1 = class_weights


class ConstituentMultiJets:
    """
    Dataset class for multiclass classification using constituent-level input.
    Accepts numpy arrays of shape (n_jets, 20, D), where D = 4 (no augment), 5 (with energy), or 6 (with label).
    Labels: QCD=0, Top=1, WZ=2
    """
    def __init__(self, qcd_const_path, top_const_path, wz_const_path,
                 qcd_small_path, top_small_path, wz_small_path,
                 batch_size, frac_train, frac_val, seed, augment=True):
        self.batch_size = batch_size

        # Load constituent-level inputs
        self.qcd_const = self.load_constituents(qcd_const_path)
        self.top_const = self.load_constituents(top_const_path)
        self.wz_const  = self.load_constituents(wz_const_path)

        if augment:
            # Energy-augmented
            qcd_const_5 = self.add_energy(self.qcd_const)
            top_const_5 = self.add_energy(self.top_const)
            wz_const_5  = self.add_energy(self.wz_const)

            # Load jet-level η/ϕ values
            qcd_df = self.clean_data(pd.read_hdf(qcd_small_path))
            top_df = self.clean_data(pd.read_hdf(top_small_path))
            wz_df  = self.clean_data(pd.read_hdf(wz_small_path))

            qcd_eta, qcd_phi = qcd_df["eta"].to_numpy(), qcd_df["phi"].to_numpy()
            top_eta, top_phi = top_df["eta"].to_numpy(), top_df["phi"].to_numpy()
            wz_eta,  wz_phi  = wz_df["eta"].to_numpy(),  wz_df["phi"].to_numpy()

            # Center η and ϕ
            self.qcd_const = self.center_constituents(qcd_const_5, qcd_eta, qcd_phi)
            self.top_const = self.center_constituents(top_const_5, top_eta, top_phi)
            self.wz_const  = self.center_constituents(wz_const_5,  wz_eta,  wz_phi)

        # Assign labels: QCD=0, Top=1, WZ=2
        qcd_labels = np.zeros((self.qcd_const.shape[0], 1))
        top_labels = np.ones((self.top_const.shape[0], 1))
        wz_labels  = np.full((self.wz_const.shape[0], 1), 2)

        # Tile labels from (n_jet, 1, 1) → (n_jet, 20, 1)
        qcd_labels_tiled = np.tile(qcd_labels[:, None, :], (1, self.qcd_const.shape[1], 1))
        top_labels_tiled = np.tile(top_labels[:, None, :], (1, self.top_const.shape[1], 1))
        wz_labels_tiled  = np.tile(wz_labels[:, None, :],  (1, self.wz_const.shape[1], 1))

        # Concatenate tiled labels with constituent data (final shape: n_jet × 20 × 6)
        qcd_labeled = np.concatenate([self.qcd_const, qcd_labels_tiled], axis=2)
        top_labeled = np.concatenate([self.top_const, top_labels_tiled], axis=2)
        wz_labeled  = np.concatenate([self.wz_const,  wz_labels_tiled],  axis=2)

        # Combine and shuffle
        all_data = np.concatenate([qcd_labeled, top_labeled, wz_labeled], axis=0)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_data)

        # Split
        n_total = all_data.shape[0]
        n_train = int(n_total * frac_train)
        n_val = int(n_total * frac_val)

        self.train = all_data[:n_train]
        self.val = all_data[n_train:n_train + n_val]
        self.test = all_data[n_train + n_val:]

        # Save label arrays (as ints)
        self.train_labels = self.train[:, 0, -1].astype(int)
        self.val_labels = self.val[:, 0, -1].astype(int)
        self.test_labels = self.test[:, 0, -1].astype(int)

        # Build DataLoaders
        self.train_dataloader = self.get_dataloader(self.train)
        self.val_dataloader = self.get_dataloader(self.val, shuffle=False)
        self.test_dataloader = self.get_dataloader(self.test, shuffle=False)

    def load_constituents(self, path):
        with h5py.File(path, 'r') as f:
            return f["constituents"][:]

    def add_energy(self, constituents: np.ndarray) -> np.ndarray:
        pt = constituents[:, :, 0]
        eta = constituents[:, :, 1]
        E = pt * np.cosh(eta)
        return np.concatenate([constituents, E[..., None]], axis=-1)

    def center_constituents(self, constituents: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> np.ndarray:
        centered = constituents.copy()
        centered[:, :, 1] = jet_eta[:, np.newaxis] - centered[:, :, 1]
        dphi = jet_phi[:, np.newaxis] - centered[:, :, 2]
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dphi = np.where(dphi >  np.pi, dphi - 2*np.pi, dphi)
        centered[:, :, 2] = dphi
        return centered

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(inplace=True)
        return data

    def get_dataloader(self, data, shuffle=True):
        x = torch.tensor(data[:, :, :-1], dtype=torch.float32)
        y = torch.tensor(data[:, 0, -1], dtype=torch.long)  # long for CrossEntropyLoss
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


class ConstituentMultiWeightedJets(ConstituentMultiJets):
    """
    Dataset class with class-level reweighting for multiclass classification of constituent-level data.
    Labels: QCD=0, Top=1, WZ=2
    """
    def __init__(self, qcd_const_path, top_const_path, wz_const_path,
                 qcd_small_path, top_small_path, wz_small_path,
                 batch_size, frac_train, frac_val, seed, augment=True):
        super().__init__(
            qcd_const_path, top_const_path, wz_const_path,
            qcd_small_path, top_small_path, wz_small_path,
            batch_size, frac_train, frac_val, seed, augment
        )

        # Compute class weights based on total label counts
        labels = np.concatenate([
            np.zeros(len(self.qcd_const)),
            np.ones(len(self.top_const)),
            np.full(len(self.wz_const), 2)
        ])

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=labels
        )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        