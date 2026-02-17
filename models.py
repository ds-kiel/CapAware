import torch
from torch import nn
import torchmetrics
import lightning.pytorch as pl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import math

import seaborn as sns
from torchmetrics import Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix

import loss
import utility

#
# Bandwidth Prediction Models
#
class CapAwareBandwidthPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_type = config['model_type']
        self.input_size = config["input_size"]
        self.out_features = config["out_features"]
        self.pred_len = config["pred_len"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.batch_first = config["batch_first"]
        self.dropout_rnn = config["dropout_rnn"]
        self.dropout_linear = config["dropout_linear"]
        self.learning_rate = config["learning_rate"]
        self.bidirectional = config['bidirectional']
        self.optimizer = config['optimizer']
        self.lr_scheduler = config['lr_scheduler']
        self.num_linear_layers = config['num_linear_layers']
        self.seq_len = config['seq_len']
        self.fused = config['fused']
        self.use_handover = config['use_handover']

        if self.use_handover:
            print('Using handover')
        else:
            print('Not using handover')

        self.penalty_over = config['penalty_over']
        self.penalty_mild = config['penalty_mild']
        self.penalty_deep = config['penalty_deep']
        self.underutil_threshold = config['underutil_threshold']
        self.exponent_over = config['exponent_over']
        self.soft_factor = config['soft_factor']

        self.val_inputs = []
        self.val_labels = []
        self.val_predictions = []

        self.test_inputs = []
        self.test_labels = []
        self.test_predictions = []

        criterions = {
            'MSELoss': nn.MSELoss(),
            'L1Loss': nn.L1Loss(),
            'SmoothL1Loss': nn.SmoothL1Loss(),
            # 0.45 comes from PERCEIVE paper
            'QuantileLoss': loss.QuantileLoss(quantile=0.45),
            'ARULoss': loss.ARULoss(
                self.penalty_over, 
                self.penalty_mild, 
                self.penalty_deep, 
                self.underutil_threshold),
            'HybridARULoss': loss.HybridARULoss(
                self.penalty_over,
                self.penalty_mild,
                self.penalty_deep,
                self.underutil_threshold,
                self.exponent_over), 
            'ARULossHO': loss.ARULossHO(
                self.penalty_over,
                self.penalty_mild,
                self.penalty_deep,
                self.underutil_threshold,
                self.exponent_over,
                self.soft_factor), 
        }
        self.criterion = criterions.get(config["criterion"], nn.MSELoss())

        activations = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Softplus': nn.Softplus(),
            'ELU': nn.ELU(),
        }
        self.activation = activations.get(config["activation"], nn.ReLU())

        self.MAE = torchmetrics.MeanAbsoluteError()
        self.MSE = torchmetrics.MeanSquaredError()
        self.RMSE = torchmetrics.MeanSquaredError(squared=False)
        self.MAPE = torchmetrics.MeanAbsolutePercentageError()
        self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()

        if self.model_type == 'LSTM':
            self.model = nn.LSTM(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                batch_first=self.batch_first,
                dropout=self.dropout_rnn,
                bidirectional=self.bidirectional)

        elif self.model_type == 'GRU':
            self.model = nn.GRU(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                batch_first=self.batch_first,
                dropout=self.dropout_rnn,
                bidirectional=self.bidirectional)

        if self.bidirectional:
            self.in_features = self.hidden_size*2
        else:
            self.in_features = self.hidden_size

        linear_layers = []
        if self.num_linear_layers > 1:
            # Add hidden linear layers with activation (and dropout, if enabled)
            for _ in range(self.num_linear_layers - 1):
                linear_layers.append(nn.Linear(self.in_features, self.in_features))
                linear_layers.append(self.activation)
                if self.dropout_linear:
                    linear_layers.append(nn.Dropout(self.dropout_linear))
            # Final layer: no activation afterward
            linear_layers.append(nn.Linear(self.in_features, self.out_features * self.pred_len))
        else:
            # Only one linear layer: final output mapping
            linear_layers.append(nn.Linear(self.in_features, self.out_features * self.pred_len))
        self.linear = nn.Sequential(*linear_layers)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizers = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW
        }
        optimizer = optimizers.get(self.optimizer, optimizers['AdamW'])(
            self.parameters(), lr=self.learning_rate, fused=self.fused
        )

        schedulers = {
            'StepLR': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9),
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        }
        scheduler = schedulers.get(
            self.lr_scheduler, schedulers['ReduceLROnPlateau'])

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def forward(self, input):
        # input is size (batch_size, seq_len, num_features)
        prediction, _ = self.model(input)
        # prediction is size (batch_size, seq_len, hidden_size)

        # passing through stacked linear layers
        # only last output of LSTM
        prediction_out = self.linear(prediction[:, -1])

        # Reshape back to [batch_size, pred_len, out_features]
        prediction_out = prediction_out.view(-1, self.pred_len, self.out_features)
        
        return prediction_out

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        
        predictions = self(inputs)
        
        if self.use_handover:
            loss = self.criterion(predictions, labels, handovers)
        else:
            loss = self.criterion(predictions, labels)

        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        
        predictions = self(inputs)

        if self.use_handover:
            loss = self.criterion(predictions, labels, handovers)
        else:
            loss = self.criterion(predictions, labels)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        self.val_inputs.append(inputs.detach())
        self.val_labels.append(labels.detach())
        self.val_predictions.append(predictions.detach())

        metrics = {
            "val_loss_MAE": self.MAE(predictions, labels),
            "val_loss_MSE": self.MSE(predictions, labels),
            "val_loss_RMSE": self.RMSE(predictions, labels),
            "val_loss_MAPE": self.MAPE(predictions, labels),
            "val_loss_sMAPE": self.sMAPE(predictions, labels),
        }
        self.log_dict(metrics)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        
        predictions = self(inputs)

        if self.use_handover:
            loss = self.criterion(predictions, labels, handovers)
        else:
            loss = self.criterion(predictions, labels)
        
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        
        self.test_inputs.append(inputs.detach()) # type: ignore
        self.test_labels.append(labels.detach()) # type: ignore
        self.test_predictions.append(predictions.detach()) # type: ignore

        metrics = {
            "test_loss_MAE": self.MAE(predictions, labels),
            "test_loss_MSE": self.MSE(predictions, labels),
            "test_loss_RMSE": self.RMSE(predictions, labels),
            "test_loss_MAPE": self.MAPE(predictions, labels),
            "test_loss_sMAPE": self.sMAPE(predictions, labels),
        }
        self.log_dict(metrics)

        return loss

    def on_test_epoch_end(self):
        inputs = torch.cat(self.test_inputs).cpu().numpy() # type: ignore
        y_pred = torch.cat(self.test_predictions).cpu().numpy() # type: ignore
        y_true = torch.cat(self.test_labels).cpu().numpy() # type: ignore

        self.test_inputs = inputs
        self.test_labels = y_true
        self.test_predictions = y_pred

        print('inputs.shape: {}, y_pred.shape: {}, y_true.shape: {}'.format(
            inputs.shape, y_pred.shape, y_true.shape))

        # Convert to DataFrame for evaluation
        df = pd.DataFrame({
            'Predictions': y_pred.ravel(),
            'Labels': y_true.ravel()
        })
        print(df.info())
        print(df.describe())
        # Evaluate model metrics
        # Assuming evaluate is a module with a function to evaluate metrics
        eval_metrics = utility.evaluate_model_metrics(df, prediction_col='Predictions', label_col='Labels')
        self.log_dict(eval_metrics)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Predictions"], color="red", label="Predictions")
        ax_abs.plot(df.index, df["Labels"], color="blue", label="Labels")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Predictions vs Labels")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Predictions vs Labels - Testing": wandb.Image(fig_abs)
        })
        plt.close()

        # Calculate errors
        df["Error"] = df["Predictions"] - df["Labels"]

        # Separate positive and negative errors
        df["Overprediction"] = df["Error"].apply(lambda x: x if x > 0 else 0)
        df["Underprediction"] = df["Error"].apply(lambda x: x if x < 0 else 0)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Overprediction"], color="red", label="Overprediction (+)")
        ax_abs.plot(df.index, df["Underprediction"], color="blue", label="Underprediction (-)")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Prediction Errors: Overprediction & Underprediction")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Prediction Errors: Overprediction & Underprediction - Testing": wandb.Image(fig_abs)
        })
        plt.close()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        prediction = self(inputs)
        return prediction
    
class Perceive(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_type = config['model_type']
        self.input_size = config["input_size"]
        self.out_features = config["out_features"]
        self.pred_len = config["pred_len"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.batch_first = config["batch_first"]
        self.dropout_rnn = config["dropout_rnn"]
        self.dropout_linear = config["dropout_linear"]
        self.learning_rate = config["learning_rate"]
        self.optimizer = config['optimizer']
        self.lr_scheduler = config['lr_scheduler']
        self.seq_len = config['seq_len']
        self.fused = config['fused']

        self.test_inputs = []
        self.test_labels = []
        self.test_predictions = []

        self.criterion = loss.QuantileLoss(quantile=0.45)
        self.activation = nn.ELU()

        self.MAE = torchmetrics.MeanAbsoluteError()
        self.MSE = torchmetrics.MeanSquaredError()
        self.RMSE = torchmetrics.MeanSquaredError(squared=False)
        self.MAPE = torchmetrics.MeanAbsolutePercentageError()
        self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()

        self.model1 = nn.LSTM(
                input_size=self.input_size,
                hidden_size=150,
                num_layers=1,
                batch_first=self.batch_first)
        
        self.model2 = nn.LSTM(
                input_size=150,
                hidden_size=100,
                num_layers=1,
                batch_first=self.batch_first)

        linear_layers = []
        linear_layers.append(nn.Dropout(p=0.5))
        linear_layers.append(nn.Linear(100, 100))
        linear_layers.append(self.activation)
        linear_layers.append(nn.Dropout(p=0.5))
        linear_layers.append(nn.Linear(100, 1))
        linear_layers.append(self.activation)
        self.linear = nn.Sequential(*linear_layers)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, fused=self.fused)

        schedulers = {
            'StepLR': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9),
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        }
        scheduler = schedulers.get(
            self.lr_scheduler, schedulers['ReduceLROnPlateau'])

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def forward(self, input):
        prediction1, _ = self.model1(input)    
        activated_pred_1 = self.activation(prediction1)

        prediction2, _ = self.model2(activated_pred_1)
        activated_pred2 = self.activation(prediction2)
                
        prediction_out = self.linear(activated_pred2[:, -1])
        
        # Reshape back to [batch_size, pred_len, out_features]
        prediction_out = prediction_out.view(-1, self.pred_len, self.out_features)

        return prediction_out

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch

        prediction = self(inputs)

        loss = self.criterion(prediction, labels)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch

        prediction = self(inputs)

        loss = self.criterion(prediction, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        metrics = {
            "val_loss_MAE": self.MAE(prediction, labels),
            "val_loss_MSE": self.MSE(prediction, labels),
            "val_loss_RMSE": self.RMSE(prediction, labels),
            "val_loss_MAPE": self.MAPE(prediction, labels),
            "val_loss_sMAPE": self.sMAPE(prediction, labels),
        }
        self.log_dict(metrics)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch

        prediction = self(inputs)

        loss = self.criterion(prediction, labels)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.test_step_outputs.append(loss)

        self.test_inputs.append(inputs.detach()) # type: ignore
        self.test_labels.append(labels.detach()) # type: ignore
        self.test_predictions.append(prediction.detach()) # type: ignore

        metrics = {
            "test_loss_MAE": self.MAE(prediction, labels),
            "test_loss_MSE": self.MSE(prediction, labels),
            "test_loss_RMSE": self.RMSE(prediction, labels),
            "test_loss_MAPE": self.MAPE(prediction, labels),
            "test_loss_sMAPE": self.sMAPE(prediction, labels),
        }
        self.log_dict(metrics)

        return loss

    def on_test_epoch_end(self):
        inputs = torch.cat(self.test_inputs).cpu().numpy() # type: ignore
        y_pred = torch.cat(self.test_predictions).cpu().numpy() # type: ignore
        y_true = torch.cat(self.test_labels).cpu().numpy() # type: ignore

        self.test_inputs = inputs
        self.test_labels = y_true
        self.test_predictions = y_pred

        print('inputs.shape: {}, y_pred.shape: {}, y_true.shape: {}'.format(
            inputs.shape, y_pred.shape, y_true.shape))

        # Convert to DataFrame for evaluation
        df = pd.DataFrame({
            'Predictions': y_pred.ravel(),
            'Labels': y_true.ravel()
        })
        print(df.info())
        print(df.describe())
        # Evaluate model metrics
        # Assuming evaluate is a module with a function to evaluate metrics
        eval_metrics = utility.evaluate_model_metrics(df, prediction_col='Predictions', label_col='Labels')
        self.log_dict(eval_metrics)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Predictions"], color="red", label="Predictions")
        ax_abs.plot(df.index, df["Labels"], color="blue", label="Labels")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Predictions vs Labels")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Predictions vs Labels - Testing": wandb.Image(fig_abs)
        })
        plt.close()

        # Calculate errors
        df["Error"] = df["Predictions"] - df["Labels"]

        # Separate positive and negative errors
        df["Overprediction"] = df["Error"].apply(lambda x: x if x > 0 else 0)
        df["Underprediction"] = df["Error"].apply(lambda x: x if x < 0 else 0)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Overprediction"], color="red", label="Overprediction (+)")
        ax_abs.plot(df.index, df["Underprediction"], color="blue", label="Underprediction (-)")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Prediction Errors: Overprediction & Underprediction")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Prediction Errors: Overprediction & Underprediction - Testing": wandb.Image(fig_abs)
        })
        plt.close()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        prediction = self(inputs)
        return prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):                     # x : (B,T,d_model)
        return x + self.pe[:, : x.size(1)] # type: ignore

class SURE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = config["input_size"]

        self.seq_len:      int = 50
        self.pred_len:      int = 1
        self.out_features:  int = 1
        self.d_model = 128
        self.n_heads = 8
        self.d_ff = 256
        self.dropout = 0.03
        self.lr = 0.00025

        # linear projection from raw features to model dim
        self.embed = nn.Linear(self.input_size, self.d_model)

        # positional encoding
        self.posenc = PositionalEncoding(self.d_model, max_len=self.seq_len)

        # single Transformer encoder block
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )

        # global average pooling over time
        self.pool = nn.AdaptiveAvgPool1d(1)

        # head – single linear layer
        self.fc = nn.Linear(
            self.d_model,                  # 128
            self.out_features * self.pred_len  # 1 · 1 = 1
        )

        self.criterion = nn.MSELoss()

        self.MAE = torchmetrics.MeanAbsoluteError()
        self.MSE = torchmetrics.MeanSquaredError()
        self.RMSE = torchmetrics.MeanSquaredError(squared=False)
        self.MAPE = torchmetrics.MeanAbsolutePercentageError()
        self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()

        self.val_inputs = []
        self.val_labels = []
        self.val_predictions = []

        self.test_inputs = []
        self.test_labels = []
        self.test_predictions = []

    def forward(self, x):                     # x:(B,T,F)
        x = self.embed(x)                     # (B,T,d_model)
        x = self.posenc(x)                    # +PE
        x = self.encoder(x)                   # (B,T,d_model)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)   # (B,d_model)
        x = self.fc(x)                        # (B, pred_len*out_features)
        return x.view(-1,
                      self.pred_len,
                      self.out_features)
    
    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        y_hat = self(inputs)
        loss  = self.criterion(y_hat, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        y_hat = self(inputs)
        loss  = self.criterion(y_hat, labels)

        self.log("val_loss", loss, prog_bar=True)

        self.val_inputs.append(inputs.detach())
        self.val_labels.append(labels.detach())
        self.val_predictions.append(y_hat.detach())

        metrics = {
            "val_loss_MAE": self.MAE(y_hat, labels),
            "val_loss_MSE": self.MSE(y_hat, labels),
            "val_loss_RMSE": self.RMSE(y_hat, labels),
            "val_loss_MAPE": self.MAPE(y_hat, labels),
            "val_loss_sMAPE": self.sMAPE(y_hat, labels),
        }
        self.log_dict(metrics)
        
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        y_hat = self(inputs)
        loss  = self.criterion(y_hat, labels)
        
        self.log("test_loss", loss, prog_bar=True)

        self.test_inputs.append(inputs.detach()) # type: ignore
        self.test_labels.append(labels.detach()) # type: ignore
        self.test_predictions.append(y_hat.detach()) # type: ignore

        metrics = {
            "test_loss_MAE": self.MAE(y_hat, labels),
            "test_loss_MSE": self.MSE(y_hat, labels),
            "test_loss_RMSE": self.RMSE(y_hat, labels),
            "test_loss_MAPE": self.MAPE(y_hat, labels),
            "test_loss_sMAPE": self.sMAPE(y_hat, labels),
        }
        self.log_dict(metrics)
        return loss

    def on_test_epoch_end(self):
        inputs = torch.cat(self.test_inputs).cpu().numpy() # type: ignore
        y_pred = torch.cat(self.test_predictions).cpu().numpy() # type: ignore
        y_true = torch.cat(self.test_labels).cpu().numpy() # type: ignore

        self.test_inputs = inputs
        self.test_labels = y_true
        self.test_predictions = y_pred

        print('inputs.shape: {}, y_pred.shape: {}, y_true.shape: {}'.format(
            inputs.shape, y_pred.shape, y_true.shape))

        # Convert to DataFrame for evaluation
        df = pd.DataFrame({
            'Predictions': y_pred.ravel(),
            'Labels': y_true.ravel()
        })
        print(df.info())
        print(df.describe())
        # Evaluate model metrics
        # Assuming evaluate is a module with a function to evaluate metrics
        eval_metrics = utility.evaluate_model_metrics(df, prediction_col='Predictions', label_col='Labels')
        self.log_dict(eval_metrics)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Predictions"], color="red", label="Predictions")
        ax_abs.plot(df.index, df["Labels"], color="blue", label="Labels")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Predictions vs Labels")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Predictions vs Labels - Testing": wandb.Image(fig_abs)
        })
        plt.close()

        # Calculate errors
        df["Error"] = df["Predictions"] - df["Labels"]

        # Separate positive and negative errors
        df["Overprediction"] = df["Error"].apply(lambda x: x if x > 0 else 0)
        df["Underprediction"] = df["Error"].apply(lambda x: x if x < 0 else 0)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Overprediction"], color="red", label="Overprediction (+)")
        ax_abs.plot(df.index, df["Underprediction"], color="blue", label="Underprediction (-)")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Prediction Errors: Overprediction & Underprediction")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Prediction Errors: Overprediction & Underprediction - Testing": wandb.Image(fig_abs)
        })
        plt.close()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, handovers, labels = batch
        prediction = self(inputs)
        return prediction

class UplinkNet(pl.LightningModule):
    def __init__(self, config):
        
        super().__init__()
        self.save_hyperparameters()

        self.input_size = config["input_size"]

        self.seq_len:      int = 5
        self.pred_len:      int = 1    # forecasting horizon
        self.out_features:  int = 1    # y dimension
        self.hidden_size:   int = 16   # LSTM hidden dim
        self.conv_filters:  int = 8    # Conv1D out channels
        self.dropout_ratio:       float = 0.03
        self.lr:            float = 1e-4

        self.conv   = nn.Conv1d(self.input_size, self.conv_filters, kernel_size=1)

        self.lstm   = nn.LSTM(
            input_size = self.conv_filters,
            hidden_size = self.hidden_size,
            num_layers  = 1,
            batch_first = True)

        self.dropout = nn.Dropout(self.dropout_ratio)

        # two time-distributed dense layers
        self.fc_t1 = nn.Linear(self.hidden_size, 8)
        self.fc_t2 = nn.Linear(8, self.out_features * self.pred_len)

        self.criterion = nn.MSELoss()

        self.MAE = torchmetrics.MeanAbsoluteError()
        self.MSE = torchmetrics.MeanSquaredError()
        self.RMSE = torchmetrics.MeanSquaredError(squared=False)
        self.MAPE = torchmetrics.MeanAbsolutePercentageError()
        self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()

        self.val_inputs = []
        self.val_labels = []
        self.val_predictions = []

        self.test_inputs = []
        self.test_labels = []
        self.test_predictions = []

    def forward(self, x):
        # Input x shape: (Batch, Seq_Len, Features) -> (B, T, F)

        # 1. Permute for Conv1d 
        # Conv1d expects (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)  
        x = self.conv(x)
        # Output x shape: (Batch, Conv_Filters, Seq_Len)

        # 2. Permute back for LSTM
        # LSTM expects (Batch, Seq_Len, Features)
        # We swap the last two dimensions to put Time back in the middle.
        x = x.permute(0, 2, 1)
        # Output x shape: (Batch, Seq_Len, Conv_Filters)

        # 3. Apply LSTM correctly over time
        x, _ = self.lstm(x)
        # Output x shape: (Batch, Seq_Len, Hidden_Size)

        # 4. Take the last time step
        x = x[:, -1, :] 
        # Output x shape: (Batch, Hidden_Size)

        # 5. Final Layers
        x = self.dropout(x)
        x = torch.relu(self.fc_t1(x))
        x = self.fc_t2(x)
        
        return x.view(-1, self.pred_len, self.out_features)

    # def forward(self, x):                 # x: (B, T, F)
    #     x = x.permute(0, 2, 1)            # (B, F, T)
    #     x = self.conv(x)                  # (B, conv_filters, T)
    #     x = x.flatten(1)                  # (B, conv_filters*T)
    #     x = x.unsqueeze(1)                # (B, 1, conv_filters*T)
    #     x, _ = self.lstm(x)               # (B, 1, hidden)
    #     x = self.dropout(x.squeeze(1))    # (B, hidden)
    #     x = torch.relu(self.fc_t1(x))     # (B, 8)
    #     x = self.fc_t2(x)                 # (B, pred_len*out_features)
    #     return x.view(-1, self.pred_len, self.out_features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        x, handover, y = batch
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, handover, y = batch
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)

        self.val_inputs.append(x.detach())
        self.val_labels.append(y.detach())
        self.val_predictions.append(y_hat.detach())

        metrics = {
            "val_loss_MAE": self.MAE(y_hat, y),
            "val_loss_MSE": self.MSE(y_hat, y),
            "val_loss_RMSE": self.RMSE(y_hat, y),
            "val_loss_MAPE": self.MAPE(y_hat, y),
            "val_loss_sMAPE": self.sMAPE(y_hat, y),
        }
        self.log_dict(metrics)
        
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, handover, y = batch
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)
        
        self.log("test_loss", loss, prog_bar=True)

        self.test_inputs.append(x.detach()) # type: ignore
        self.test_labels.append(y.detach()) # type: ignore
        self.test_predictions.append(y_hat.detach()) # type: ignore

        metrics = {
            "test_loss_MAE": self.MAE(y_hat, y),
            "test_loss_MSE": self.MSE(y_hat, y),
            "test_loss_RMSE": self.RMSE(y_hat, y),
            "test_loss_MAPE": self.MAPE(y_hat, y),
            "test_loss_sMAPE": self.sMAPE(y_hat, y),
        }
        self.log_dict(metrics)
        return loss

    def on_test_epoch_end(self):
        inputs = torch.cat(self.test_inputs).cpu().numpy() # type: ignore
        y_pred = torch.cat(self.test_predictions).cpu().numpy() # type: ignore
        y_true = torch.cat(self.test_labels).cpu().numpy() # type: ignore

        self.test_inputs = inputs
        self.test_labels = y_true
        self.test_predictions = y_pred

        print('inputs.shape: {}, y_pred.shape: {}, y_true.shape: {}'.format(
            inputs.shape, y_pred.shape, y_true.shape))

        # Convert to DataFrame for evaluation
        df = pd.DataFrame({
            'Predictions': y_pred.ravel(),
            'Labels': y_true.ravel()
        })
        print(df.info())
        print(df.describe())
        # Evaluate model metrics
        # Assuming evaluate is a module with a function to evaluate metrics
        eval_metrics = utility.evaluate_model_metrics(df, prediction_col='Predictions', label_col='Labels')
        self.log_dict(eval_metrics)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Predictions"], color="red", label="Predictions")
        ax_abs.plot(df.index, df["Labels"], color="blue", label="Labels")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Predictions vs Labels")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Predictions vs Labels - Testing": wandb.Image(fig_abs)
        })
        plt.close()

        # Calculate errors
        df["Error"] = df["Predictions"] - df["Labels"]

        # Separate positive and negative errors
        df["Overprediction"] = df["Error"].apply(lambda x: x if x > 0 else 0)
        df["Underprediction"] = df["Error"].apply(lambda x: x if x < 0 else 0)

        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(10, 5))
        ax_abs.plot(df.index, df["Overprediction"], color="red", label="Overprediction (+)")
        ax_abs.plot(df.index, df["Underprediction"], color="blue", label="Underprediction (-)")
        ax_abs.set_xlabel("Time Steps")
        ax_abs.set_ylabel("Prediction (Not Mbps)")
        ax_abs.set_title("Prediction Errors: Overprediction & Underprediction")
        ax_abs.legend()
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({ # type: ignore
            "Prediction Errors: Overprediction & Underprediction - Testing": wandb.Image(fig_abs)
        })
        plt.close()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, _, _ = batch
        prediction = self(inputs)
        return prediction


#
# Handover Prediction Models
#
class CapAwareHandoverPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        input_size=config['input_size']
        hidden_size=config['hidden_size']
        num_layers=config['num_layers']
        dropout=config['dropout']
        learning_rate=config['learning_rate']
        pred_len=config['pred_len']
        pos_weight=None
        threshold=config['threshold']
        dataset=config['dataset']
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, pred_len)
        self.sigmoid = nn.Sigmoid()

        # Use weighted loss function
        if pos_weight is not None:
            print(f'Using weighted loss function with pos_weight={pos_weight}')
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print('Using unweighted loss function')
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.dataset = dataset

        # Initialize binary metrics with an appropriate threshold
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

        # Initialize test metrics
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")

        # Storage for predictions and labels during validation
        self.val_preds = []
        self.val_labels = []

        # Storage for predictions and labels during test
        self.test_preds = []
        self.test_labels = []

        # Storage for predictions and labels during prediction
        self.predictions = []
        self.probabilities = []
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        # Convert logits to probabilities and then to binary predictions
        probabilities = self.sigmoid(y_hat)
        predicted_labels = (probabilities > self.threshold).int()
        y_int = y.int()  # ensure targets are integers

        # Update metrics
        self.precision(predicted_labels, y_int)
        self.recall(predicted_labels, y_int)
        self.f1(predicted_labels, y_int)

        # Accumulate predictions and labels for the confusion matrix
        self.val_preds.append(predicted_labels.detach())
        self.val_labels.append(y_int.detach())

        acc = (predicted_labels == y_int).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute final metrics for the epoch
        precision_value = self.precision.compute()
        recall_value = self.recall.compute()
        f1_value = self.f1.compute()

        # Log the metrics
        self.log("val_precision", precision_value, prog_bar=True)
        self.log("val_recall", recall_value, prog_bar=True)
        self.log("val_f1", f1_value, prog_bar=True)

        # Combine accumulated predictions and labels
        preds = torch.cat(self.val_preds).cpu().numpy().ravel()
        labels = torch.cat(self.val_labels).cpu().numpy().ravel()
        
        # Compute confusion matrix (absolute counts)
        cm = confusion_matrix(labels, preds)
        
        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_abs)
        ax_abs.set_xlabel('Predicted Labels')
        ax_abs.set_ylabel('True Labels')
        ax_abs.set_title('Confusion Matrix (Absolute Counts) - Validation')
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Absolute - Validation": wandb.Image(fig_abs),
            "global_step": self.current_epoch
        })
        plt.close(fig_abs)
        
        # Normalize confusion matrix by row (true labels) to get percentages
        #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Normalize confusion matrix by row (true labels) to get percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.true_divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
            cm_norm[~np.isfinite(cm_norm)] = 0  # replace NaN and Inf with 0
        
        # Plot confusion matrix with percentage values
        fig_perc, ax_perc = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap="Blues", ax=ax_perc)
        ax_perc.set_xlabel('Predicted Labels')
        ax_perc.set_ylabel('True Labels')
        ax_perc.set_title('Confusion Matrix (Row-wise Percentage) - Validation')
        
        # Log percentage confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Percentage - Validation": wandb.Image(fig_perc),
            "global_step": self.current_epoch
        })
        plt.close(fig_perc)
        
        # Reset metrics and accumulated predictions/labels for the next epoch
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        probabilities = self.sigmoid(y_hat)
        predicted_labels = (probabilities > self.threshold).int()
        y_int = y.int()
        
        # Update test metrics
        self.test_precision(predicted_labels, y_int)
        self.test_recall(predicted_labels, y_int)
        self.test_f1(predicted_labels, y_int)

        # Accumulate predictions and labels for the confusion matrix
        self.test_preds.append(predicted_labels.detach())
        self.test_labels.append(y_int.detach())
        
        acc = (predicted_labels == y_int).float().mean()
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        test_precision_value = self.test_precision.compute()
        test_recall_value = self.test_recall.compute()
        test_f1_value = self.test_f1.compute()

        self.log("test_precision", test_precision_value, prog_bar=True)
        self.log("test_recall", test_recall_value, prog_bar=True)
        self.log("test_f1", test_f1_value, prog_bar=True)

        # Combine accumulated predictions and labels
        preds = torch.cat(self.test_preds).cpu().numpy().ravel()
        labels = torch.cat(self.test_labels).cpu().numpy().ravel()

        self.test_preds = preds
        self.test_labels = labels

        # Compute confusion matrix (absolute counts)
        cm = confusion_matrix(labels, preds)
        
        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_abs)
        ax_abs.set_xlabel('Predicted Labels')
        ax_abs.set_ylabel('True Labels')
        ax_abs.set_title('Confusion Matrix (Absolute Counts) - Testing')
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Absolute - Testing": wandb.Image(fig_abs),
            "global_step": self.current_epoch
        })
        plt.close(fig_abs)
        
        # Normalize confusion matrix by row (true labels) to get percentages
        #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Normalize confusion matrix by row (true labels) to get percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.true_divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
            cm_norm[~np.isfinite(cm_norm)] = 0  # replace NaN and Inf with 0
        
        # Plot confusion matrix with percentage values
        fig_perc, ax_perc = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap="Blues", ax=ax_perc)
        ax_perc.set_xlabel('Predicted Labels')
        ax_perc.set_ylabel('True Labels')
        ax_perc.set_title('Confusion Matrix (Row-wise Percentage) - Testing')
        
        # Log percentage confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Percentage - Testing": wandb.Image(fig_perc),
            "global_step": self.current_epoch
        })
        plt.close(fig_perc)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self.forward(x)
        # Compute probabilities by applying sigmoid
        probabilities = self.sigmoid(logits)
        # Convert probabilities to binary predictions using a threshold
        predictions = (probabilities > self.threshold).int()

        self.predictions.append(predictions.detach())
        self.probabilities.append(probabilities.detach())
        
        # {"predictions": tensor([[1, 0], [0, 1]]),
        #  "probabilities": tensor([[0.75, 0.20], [0.35, 0.65]])}
        return {"predictions": predictions, "probabilities": probabilities}
    
    def on_predict_epoch_end(self):
        print("on_predict_epoch_end")

        # Combine accumulated predictions and labels
        predictions = torch.cat(self.predictions).cpu().numpy().ravel()
        probabilities = torch.cat(self.probabilities).cpu().numpy().ravel()

        self.predictions = predictions
        self.probabilities = probabilities
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, fused=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
class LSTMBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_out, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
    
class RSRPHandoverPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        dropout=0.1
        learning_rate=config['learning_rate']
        pred_len=config['pred_len']
        pos_weight=None
        threshold=config['threshold']
        self.save_hyperparameters()

        self.feature_extractor = nn.Sequential(
            LSTMBlock(1,   120),
            nn.Dropout(p=dropout),
            LSTMBlock(120, 50),
            LSTMBlock(50,  50),
            LSTMBlock(50,  50),
        )
        
        self.fc = nn.Linear(50, pred_len) # type: ignore
        self.sigmoid = nn.Sigmoid()

        # Use weighted loss function
        if pos_weight is not None:
            print(f'Using weighted loss function with pos_weight={pos_weight}')
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # type: ignore
        else:
            print('Using unweighted loss function')
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.learning_rate = learning_rate
        self.threshold = threshold

        # Initialize binary metrics with an appropriate threshold
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

        # Initialize test metrics
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")

        # Storage for predictions and labels during validation
        self.val_preds = []
        self.val_labels = []

        # Storage for predictions and labels during test
        self.test_preds = []
        self.test_labels = []

        # Storage for predictions and labels during prediction
        self.predictions = []
        self.probabilities = []
    
    def forward(self, x):
        lstm_out = self.feature_extractor(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        # Convert logits to probabilities and then to binary predictions
        probabilities = self.sigmoid(y_hat)
        predicted_labels = (probabilities > self.threshold).int()
        y_int = y.int()  # ensure targets are integers

        # Update metrics
        self.precision(predicted_labels, y_int)
        self.recall(predicted_labels, y_int)
        self.f1(predicted_labels, y_int)

        # Accumulate predictions and labels for the confusion matrix
        self.val_preds.append(predicted_labels.detach())
        self.val_labels.append(y_int.detach())

        acc = (predicted_labels == y_int).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute final metrics for the epoch
        precision_value = self.precision.compute()
        recall_value = self.recall.compute()
        f1_value = self.f1.compute()

        # Log the metrics
        self.log("val_precision", precision_value, prog_bar=True)
        self.log("val_recall", recall_value, prog_bar=True)
        self.log("val_f1", f1_value, prog_bar=True)

        # Combine accumulated predictions and labels
        preds = torch.cat(self.val_preds).cpu().numpy().ravel()
        labels = torch.cat(self.val_labels).cpu().numpy().ravel()
        
        # Compute confusion matrix (absolute counts)
        cm = confusion_matrix(labels, preds)
        
        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_abs)
        ax_abs.set_xlabel('Predicted Labels')
        ax_abs.set_ylabel('True Labels')
        ax_abs.set_title('Confusion Matrix (Absolute Counts) - Validation')
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Absolute - Validation": wandb.Image(fig_abs),
            "global_step": self.current_epoch
        })
        plt.close(fig_abs)
        
        # Normalize confusion matrix by row (true labels) to get percentages
        #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Normalize confusion matrix by row (true labels) to get percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.true_divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
            cm_norm[~np.isfinite(cm_norm)] = 0  # replace NaN and Inf with 0
        
        # Plot confusion matrix with percentage values
        fig_perc, ax_perc = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap="Blues", ax=ax_perc)
        ax_perc.set_xlabel('Predicted Labels')
        ax_perc.set_ylabel('True Labels')
        ax_perc.set_title('Confusion Matrix (Row-wise Percentage) - Validation')
        
        # Log percentage confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Percentage - Validation": wandb.Image(fig_perc),
            "global_step": self.current_epoch
        })
        plt.close(fig_perc)
        
        # Reset metrics and accumulated predictions/labels for the next epoch
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        probabilities = self.sigmoid(y_hat)
        predicted_labels = (probabilities > self.threshold).int()
        y_int = y.int()
        
        # Update test metrics
        self.test_precision(predicted_labels, y_int)
        self.test_recall(predicted_labels, y_int)
        self.test_f1(predicted_labels, y_int)

        # Accumulate predictions and labels for the confusion matrix
        self.test_preds.append(predicted_labels.detach())
        self.test_labels.append(y_int.detach())
        
        acc = (predicted_labels == y_int).float().mean()
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        test_precision_value = self.test_precision.compute()
        test_recall_value = self.test_recall.compute()
        test_f1_value = self.test_f1.compute()

        self.log("test_precision", test_precision_value, prog_bar=True)
        self.log("test_recall", test_recall_value, prog_bar=True)
        self.log("test_f1", test_f1_value, prog_bar=True)

        # Combine accumulated predictions and labels
        preds = torch.cat(self.test_preds).cpu().numpy().ravel()
        labels = torch.cat(self.test_labels).cpu().numpy().ravel()

        self.test_preds = preds
        self.test_labels = labels
        
        # Compute confusion matrix (absolute counts)
        cm = confusion_matrix(labels, preds)
        
        # Plot confusion matrix with absolute counts
        fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_abs)
        ax_abs.set_xlabel('Predicted Labels')
        ax_abs.set_ylabel('True Labels')
        ax_abs.set_title('Confusion Matrix (Absolute Counts) - Testing')
        
        # Log absolute confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Absolute - Testing": wandb.Image(fig_abs),
            "global_step": self.current_epoch
        })
        plt.close(fig_abs)
        
        # Normalize confusion matrix by row (true labels) to get percentages
        #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Normalize confusion matrix by row (true labels) to get percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.true_divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
            cm_norm[~np.isfinite(cm_norm)] = 0  # replace NaN and Inf with 0
        
        # Plot confusion matrix with percentage values
        fig_perc, ax_perc = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap="Blues", ax=ax_perc)
        ax_perc.set_xlabel('Predicted Labels')
        ax_perc.set_ylabel('True Labels')
        ax_perc.set_title('Confusion Matrix (Row-wise Percentage) - Testing')
        
        # Log percentage confusion matrix figure to Weights and Biases
        self.logger.experiment.log({
            "Confusion Matrix Percentage - Testing": wandb.Image(fig_perc),
            "global_step": self.current_epoch
        })
        plt.close(fig_perc)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self.forward(x)
        # Compute probabilities by applying sigmoid
        probabilities = self.sigmoid(logits)
        # Convert probabilities to binary predictions using a threshold
        predictions = (probabilities > self.threshold).int()

        self.predictions.append(predictions.detach())
        self.probabilities.append(probabilities.detach())

        # {"predictions": tensor([[1, 0], [0, 1]]),
        #  "probabilities": tensor([[0.75, 0.20], [0.35, 0.65]])}
        return {"predictions": predictions, "probabilities": probabilities}
    
    def on_predict_epoch_end(self):
        print("on_predict_epoch_end")

        # Combine accumulated predictions and labels
        predictions = torch.cat(self.predictions).cpu().numpy().ravel()
        probabilities = torch.cat(self.probabilities).cpu().numpy().ravel()

        self.predictions = predictions
        self.probabilities = probabilities
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, fused=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }