import os
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class BalancedHandoverPredictionDataset(Dataset):
    """
    Dataset for handover prediction using precomputed balanced indices.
    Each sample consists of a sequence of inputs (of length seq_len) and a sequence of binary labels
    for the next pred_len time steps.
    """
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, seq_len: int, pred_len: int, indices: np.ndarray):
        """
        Args:
            inputs (np.ndarray): Array of input features with shape (n_samples, n_features).
            labels (np.ndarray): Array of binary labels with shape (n_samples, 1) or (n_samples,).
            seq_len (int): Length of the input sequence.
            pred_len (int): Number of time steps to predict (target sequence length).
            indices (np.ndarray): Precomputed balanced starting indices.
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        # Convert labels to float32, as BCEWithLogitsLoss requires targets of type float.
        self.labels = torch.tensor(labels.squeeze(), dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual starting index for this sample.
        start_idx = self.indices[idx]
        # Extract the input sequence.
        input_seq = self.inputs[start_idx : start_idx + self.seq_len]
        # Extract the target sequence of length pred_len.
        label_seq = self.labels[start_idx + self.seq_len : start_idx + self.seq_len + self.pred_len]
        return input_seq, label_seq
    
def compute_balanced_indices(labels, seq_len, negative_ratio=1.0, random_state=42, threshold=0.5):
    """
    Compute balanced starting indices for sequences.
    Args:
        labels (np.ndarray): Array of labels with shape (n_samples, 1) or (n_samples,).
        seq_len (int): The length of the sequence used for prediction.
        negative_ratio (float): Desired ratio of negative to positive samples (1.0 gives equal numbers).
        random_state (int): Seed for reproducibility.
        threshold (float): Threshold to decide if a sample is positive.
    Returns:
        np.ndarray: Array of balanced starting indices.
    """
    # Valid starting indices for which we have a sequence and a following label.
    valid_indices = np.arange(0, len(labels) - seq_len)
    # Ensure labels is 1D.
    labels_1d = labels.squeeze()
    # The label for a sequence starting at i is taken at position i + seq_len.
    labels_for_samples = labels_1d[seq_len:]
    
    # Use a threshold to decide if a label is positive.
    positive_indices = valid_indices[labels_for_samples > threshold]
    negative_indices = valid_indices[labels_for_samples <= threshold]

    print("DEBUG: Total valid samples:", len(valid_indices))
    print("DEBUG: Positives in valid samples:", len(positive_indices))
    print("DEBUG: Negatives in valid samples:", len(negative_indices))
    
    np.random.seed(random_state)
    n_positives = len(positive_indices)
    n_negatives_to_keep = int(n_positives * negative_ratio)
    n_negatives_to_keep = min(n_negatives_to_keep, len(negative_indices))
    print("DEBUG: Number of positives:", n_positives)
    print("DEBUG: Number of negatives to keep (ratio={}): {}".format(negative_ratio, n_negatives_to_keep))
    
    selected_negative_indices = np.random.choice(negative_indices, size=n_negatives_to_keep, replace=False)
    print("DEBUG: Selected negatives indices sample:", selected_negative_indices[:10])
    
    # Combine positive and downsampled negative indices, then shuffle.
    balanced_indices = np.concatenate([positive_indices, selected_negative_indices])
    print("DEBUG: Combined balanced indices count before shuffling:", len(balanced_indices))
    
    np.random.shuffle(balanced_indices)
    print("DEBUG: Balanced indices after shuffling (first 10):", balanced_indices[:10])
    
    return balanced_indices

def stratified_split(balanced_indices, labels, seq_len, train_frac=0.6, val_frac=0.2, threshold=0.5):
    """
    Stratified split of balanced_indices based on the sample label, where the label for a sample
    at index i is actually at labels[i + seq_len].
    """
    # Use the label at i + seq_len for each sample index i.
    pos_mask = labels[balanced_indices + seq_len] > threshold
    pos_indices = balanced_indices[pos_mask]
    neg_indices = balanced_indices[~pos_mask]

    # Shuffle each category.
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # Compute split sizes for each category.
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    n_pos_train = int(train_frac * n_pos)
    n_neg_train = int(train_frac * n_neg)

    n_pos_val = int(val_frac * n_pos)
    n_neg_val = int(val_frac * n_neg)

    # Split positives.
    train_pos = pos_indices[:n_pos_train]
    val_pos = pos_indices[n_pos_train:n_pos_train+n_pos_val]
    test_pos = pos_indices[n_pos_train+n_pos_val:]

    # Split negatives.
    train_neg = neg_indices[:n_neg_train]
    val_neg = neg_indices[n_neg_train:n_neg_train+n_neg_val]
    test_neg = neg_indices[n_neg_train+n_neg_val:]

    # Combine and shuffle.
    train_idx = np.concatenate([train_pos, train_neg])
    val_idx = np.concatenate([val_pos, val_neg])
    test_idx = np.concatenate([test_pos, test_neg])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return train_idx, val_idx, test_idx

class HandoverDataModule(pl.LightningDataModule):
    def __init__(self, config, run_id,
                 external_scaler_feature: MinMaxScaler | None = None,
                 external_scaler_label  : MinMaxScaler | None = None,
                 make_splits: bool = True):
        super().__init__()
        # file_path, batch_size, seq_len, pred_len, negative_ratio=1.0, balance_data=True
        self.dataset = config['dataset']
        self.batch_size = config['batch_size']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.negative_ratio = config['negative_ratio']
        self.balance_data = config['balance_data']  # New parameter
        self.input_size = None
        self.scaler_feature = external_scaler_feature or MinMaxScaler()
        self.scaler_label = external_scaler_label or MinMaxScaler()

        self.external_scaler_feature = external_scaler_feature
        self.external_scaler_label = external_scaler_label

        self.make_splits = make_splits
        self.data_read = False  # flag to ensure file is processed only once
        self.run_id = run_id

        if self.dataset == 'Fjord5G-4312':
            self.file_path = './data/CAU-4312-Fjord5G-handovers-SA-only-processed.parquet.gzip'
            self.inputs = ['speedkmh', 'lRsrp', 'lSinr', 'lCqi'] # GPS speed + signal strength
            self.outputs = ['Handover']
        else:
            print("Unknown dataset")

    def prepare_data(self):
        # This method is called only once on one process.
        if not self.data_read:
            df = pd.read_parquet(self.file_path)
            print(df.info())
            print(df.head())
            print(df.describe(include='all'))
            print(df['Handover'].value_counts())
            print('self.inputs: {}'.format(self.inputs))
            print('self.outputs: {}'.format(self.outputs))

            self.features = df[self.inputs].values.astype(np.float32)
            print('self.features.shape: {}'.format(self.features.shape))
            print('self.features.dtype: {}'.format(self.features.dtype))
            print('self.features[:5]: {}'.format(self.features[:5]))

            self.labels = df[self.outputs].values.reshape(-1, 1).astype(np.float32)

            print('self.features.shape: {}'.format(self.features.shape))
            print('self.labels.shape: {}'.format(self.labels.shape))
            print('self.features.dtype: {}'.format(self.features.dtype))
            print('self.labels.dtype: {}'.format(self.labels.dtype))

            self.input_size = len(self.inputs)
            self.data_read = True

    def setup(self, stage=None):
        if not hasattr(self, 'train_data'):
            if self.balance_data:
                # Use downsampling to balance the dataset
                balanced_indices_all = compute_balanced_indices(self.labels, self.seq_len, self.negative_ratio)
            else:
                # Disable balancing: use all valid starting indices
                balanced_indices_all = np.arange(0, len(self.labels) - self.seq_len)
            
            if self.make_splits:
                # Split indices (you can still perform a stratified split to maintain temporal structure)
                train_indices, val_indices, test_indices = stratified_split(
                    balanced_indices_all, self.labels.squeeze(), self.seq_len, train_frac=0.98, val_frac=0.01, threshold=0.5
                )
            else:
                test_indices = np.arange(0, len(self.labels) - self.seq_len)            

            if self.make_splits and self.external_scaler_feature is None:
                # Scale the data: fit scalers on the training portion only.
                train_features_raw = self.features[train_indices]
                train_labels_raw = self.labels[train_indices]
                # training run – fit the scalers

                print('train_features_raw.shape: {}'.format(train_features_raw.shape))
                print('train_features_raw.dtype: {}'.format(train_features_raw.dtype))

                print('train_labels_raw.shape: {}'.format(train_labels_raw.shape))
                print('train_labels_raw.dtype: {}'.format(train_labels_raw.dtype))

                self.scaler_feature.fit(train_features_raw)
                self.scaler_label.fit(train_labels_raw)
            else:
                # inference run – don’t touch the scalers
                assert self.external_scaler_feature is not None, \
                    "You forgot to provide a fitted feature scaler."
                assert self.external_scaler_label is not None, \
                    "You forgot to provide a fitted label scaler."
            
            # Transform the entire dataset.
            features_scaled = self.scaler_feature.transform(self.features)
            labels_scaled = self.scaler_label.transform(self.labels)

            self.scaler_dir = f"scaler-save/prediction_handover/{self.run_id}"
            print(f"Scaler directory: {self.scaler_dir}")
            os.makedirs(self.scaler_dir, exist_ok=True)

            # save scalers for inference
            joblib.dump(self.scaler_feature, f"{self.scaler_dir}/scaler_feature.gz")
            joblib.dump(self.scaler_label,   f"{self.scaler_dir}/scaler_label.gz")
            
            # save scalers for inference
            joblib.dump(self.scaler_feature, f'scaler-save/prediction_handover/scaler_feature_{self.dataset}.gz')
            joblib.dump(self.scaler_label, f'scaler-save/prediction_handover/scaler_label_{self.dataset}.gz')
            
            # Create datasets using the scaled arrays and the precomputed indices.
            if self.make_splits:
                self.train_data = BalancedHandoverPredictionDataset(
                    features_scaled, labels_scaled, self.seq_len, self.pred_len, train_indices)
                self.val_data = BalancedHandoverPredictionDataset(
                    features_scaled, labels_scaled, self.seq_len, self.pred_len, val_indices)
            self.test_data = BalancedHandoverPredictionDataset(
                features_scaled, labels_scaled, self.seq_len, self.pred_len, test_indices)

            # Optionally, update pos_weight.
            # If balancing is disabled, you might want to compute the actual ratio from the training set.
            if self.balance_data:
                self.pos_weight = torch.tensor((1.0 * self.negative_ratio), dtype=torch.float32)
            else:
                if self.make_splits:
                    # Calculate pos_weight based on actual class frequencies.
                    # For example, if y_train contains binary labels:
                    y_train = self.labels[train_indices + self.seq_len]
                    n_pos = np.sum(y_train > 0.5)
                    n_neg = np.sum(y_train <= 0.5)
                    ratio = n_neg / n_pos if n_pos > 0 else 1.0
                    self.pos_weight = torch.tensor(ratio, dtype=torch.float32)
                else:
                    # Not sure if this is correct
                    self.pos_weight = torch.tensor((1.0 * self.negative_ratio), dtype=torch.float32)


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()