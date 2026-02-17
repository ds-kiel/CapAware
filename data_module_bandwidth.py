import os
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset subclass for time series data.
    Transforms inputs into sequence data using a rolling window.
    Produces batches of shape (batch_size, seq_len, n_features), suitable for RNNs.
    """
    def __init__(self, 
                 inputs: np.ndarray,
                 handovers: np.ndarray,
                 labels: np.ndarray,
                 seq_len: int,
                 pred_len: int):
        # Validate inputs
        assert inputs.dtype in (np.float32, np.float16), "Inputs must be float32 or float16."
        assert labels.dtype in (np.float32, np.float16), "Labels must be float32 or float16."
        assert handovers.dtype in (np.float32, np.float16), "Handovers must be float32 or float16."
        assert len(inputs) == len(labels) == len(handovers), "Inputs, labels and handovers must have the same length."

        # Store data as tensors on CPU (can be moved to GPU later)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.handovers = torch.tensor(handovers, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Precompute the effective length for faster `__len__` calls
        self.effective_length = len(self.inputs) - self.seq_len - self.pred_len + 1
        if self.effective_length <= 0:
            raise ValueError("Dataset length must be greater than zero. Check seq_len and pred_len.")

    def __len__(self):
        return self.effective_length

    def __getitem__(self, index):
        seq_start = index
        seq_end = index + self.seq_len
        pred_end = seq_end + self.pred_len

        sequence_inputs = self.inputs[seq_start:seq_end]
        sequence_handovers = self.handovers[seq_start:seq_end]
        sequence_labels = self.labels[seq_end:pred_end]
        return sequence_inputs, sequence_handovers, sequence_labels

class BandwidthDataModule(pl.LightningDataModule):

    def __init__(self, config, run_id):
        super().__init__()
        self.seq_len = config['seq_len']
        self.batch_size = config['batch_size']
        self.dataset = config['dataset']
        self.num_workers = config['num_workers']
        self.prefetch_factor = config['prefetch_factor']
        self.persistent_workers = config['persistent_workers']
        self.pred_len = config['pred_len']

        self.train_p = config['train_p']
        self.val_p = config['val_p']
        self.test_p = config['test_p']
        self.len_of_inputs = None
        self.len_of_labels = None
        self.len_of_handover = None

        if config['scaler'] == 'MinMaxScaler':
            self.scaler_input = MinMaxScaler()
            self.scaler_label = MinMaxScaler()
            self.scaler_handover = MinMaxScaler()
        else:
            self.scaler_input = StandardScaler()
            self.scaler_label = StandardScaler()
            self.scaler_handover = StandardScaler()

        self.data_read = False
        self.run_id = run_id

        if self.dataset == "Fjord5G-4329-uplink":
            self.file_path = './data/CAU-4329-processed.parquet.gzip'
        elif self.dataset == "SURE-uplink":
            self.file_path = './data/SURE-uplink-Downtown-processed.parquet.gzip'
        elif self.dataset == "UplinkNet-uplink":
            self.file_path = './data/UplinkNet-processed.parquet.gzip'
        else:
            print("Unknown dataset")

    def prepare_data(self):
        # Load raw data without scaling to avoid data leakage.
        if not self.data_read:
            df = pd.read_parquet(self.file_path)
            print(df.info())
            print(df.head())

            if self.dataset == 'Fjord5G-4329-uplink':
                inputs = ['SINR', 'CQI', 'RSRP', 'Band_n3', 'Band_n78'] # core + categorical
                handovers = ['Probabilities']
                labels = ['Tx-BW']
            elif self.dataset == 'SURE-uplink':
                inputs = ['5G_TBS', '5G_RB', '5G_RSRP', '5G_PUSCH_POWER',
                          '4G_TBS', '4G_RB', '4G_RSRP', '4G_PUSCH_POWER']
                labels = ['AGG_TBS']
            elif self.dataset == 'UplinkNet-uplink':
                inputs = ['NR_CSI_RSRP', 
                          'NR_CSI_SINR', 
                          'MHz_15',
                          'MHz_40',
                          'MHz_60',
                          'MHz_100']
                labels = ['NR_Physical_Throughput_UL']

            print('inputs: {}'.format(inputs))
            print('labels: {}'.format(labels))
            print('handovers: {}'.format(handovers))
            self.len_of_inputs = len(inputs)
            self.len_of_labels = len(labels)
            self.len_of_handover = len(handovers)

            self.inputs = df[inputs].values.astype(np.float32)
            if len(labels) > 1:
                print('Multivariate label')
                self.labels = df[labels].values.astype(np.float32)
            elif len(labels) == 1:
                print('Univariate label')
                self.labels = df[labels].values.reshape(-1, 1).astype(np.float32)

            print('Univariate handover')
            self.handovers = df[handovers].values.reshape(-1, 1).astype(np.float32)

            self.data_read = True

    def setup(self, stage=None):
        # Split the raw data into train, validation, and test sets,
        # then fit scalers on the training set and transform all splits.
        if not hasattr(self, 'train_data'):
            total_samples = self.inputs.shape[0]
            # Compute split indices using rounding
            split_idx = [round(self.train_p * total_samples), round((self.train_p + self.val_p) * total_samples)]
            
            # Use np.split to divide inputs and labels
            train_inputs, val_inputs, test_inputs = np.split(self.inputs, split_idx)
            train_labels, val_labels, test_labels = np.split(self.labels, split_idx)
            train_handovers, val_handovers, test_handovers = np.split(self.handovers, split_idx)
            
            # Fit scalers on training data only
            self.scaler_input.fit(train_inputs)
            self.scaler_label.fit(train_labels)
            self.scaler_handover.fit(train_handovers)
            
            # Transform all splits using the scalers fitted on the training set
            train_inputs = self.scaler_input.transform(train_inputs)
            train_labels = self.scaler_label.transform(train_labels)
            train_handovers = self.scaler_handover.transform(train_handovers)

            val_inputs = self.scaler_input.transform(val_inputs)
            val_labels = self.scaler_label.transform(val_labels)
            val_handovers = self.scaler_handover.transform(val_handovers)
            
            test_inputs = self.scaler_input.transform(test_inputs)
            test_labels = self.scaler_label.transform(test_labels)
            test_handovers = self.scaler_handover.transform(test_handovers)

            self.scaler_dir = f"scaler-save/prediction_bandwidth/{self.run_id}"
            print(f"Scaler directory: {self.scaler_dir}")
            os.makedirs(self.scaler_dir, exist_ok=True)

            # save scalers for inference
            joblib.dump(self.scaler_input, f"{self.scaler_dir}/scaler_input.gz")
            joblib.dump(self.scaler_label,   f"{self.scaler_dir}/scaler_label.gz")
            joblib.dump(self.scaler_handover, f"{self.scaler_dir}/scaler_handover.gz")

            # save scalers for inference
            joblib.dump(self.scaler_input, f'scaler-save/prediction_bandwidth/scaler_input_{self.dataset}.gz')
            joblib.dump(self.scaler_label, f'scaler-save/prediction_bandwidth/scaler_label_{self.dataset}.gz')
            joblib.dump(self.scaler_handover, f'scaler-save/prediction_bandwidth/scaler_handover_{self.dataset}.gz')

            # Create the datasets from the scaled splits
            self.train_data = TimeSeriesDataset(train_inputs, train_handovers, train_labels, self.seq_len, self.pred_len)
            self.val_data = TimeSeriesDataset(val_inputs, val_handovers, val_labels, self.seq_len, self.pred_len)
            self.test_data = TimeSeriesDataset(test_inputs, test_handovers, test_labels, self.seq_len, self.pred_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers, 
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers, 
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def inverse_transform_input(self, data):
        return self.scaler_input.inverse_transform(data)
    
    def inverse_transform_label(self, data):
        return self.scaler_label.inverse_transform(data)