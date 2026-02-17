import os
import time
import torch
import argparse

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import wandb

from models import CapAwareBandwidthPredictor, Perceive, SURE, UplinkNet
from data_module_bandwidth import BandwidthDataModule
from utility import print_environment
    
def train(config):
    
    pl.seed_everything(42, workers=True)
    wandb_logger = WandbLogger(
        project='Uplink-Bandwidth-Prediction', 
        save_code=True)

    run_id = wandb_logger.experiment.id
    
    if config['eval_model'] == 'Perceive1000':
        config['seq_len'] = 50
    elif config['eval_model'] == 'Perceive300':
        config['seq_len'] = 15
    elif config['eval_model'] == 'Perceive100':
        config['seq_len'] = 5
    elif config['eval_model'] == 'UplinkNet':
        config['seq_len'] = 5
    elif config['eval_model'] == 'SURE':
        config['batch_size'] = 128

    if config['criterion'] == 'ARULossHO':
        config['use_handover'] = True
        print('Using handover due to criterion: {}'.format(config['criterion']))
    else:    
        config['use_handover'] = False
        print('Not using handover due to criterion: {}'.format(config['criterion']))

    data = BandwidthDataModule(config, run_id)
    data.prepare_data()

    config['input_size'] = data.len_of_inputs
    config['out_features'] = data.len_of_labels
    print('len_of_inputs: {}'.format(data.len_of_inputs))
    print('len_of_labels: {}'.format(data.len_of_labels))

    if config['eval_model'] == 'CapAware':
        model = CapAwareBandwidthPredictor(config)
    elif config['eval_model'] == 'Perceive1000' or config['eval_model'] == 'Perceive300' or config['eval_model'] == 'Perceive100':
        model = Perceive(config)
    elif config["eval_model"] == "SURE":
        model = SURE(config)
    elif config['eval_model'] == 'UplinkNet':
        model = UplinkNet(config)
    else:
        print('Unknown model')
        raise ValueError('Unknown model')

    callbacks = []
    callbacks.append(EarlyStopping(
        monitor='val_loss', 
        patience=5,
        verbose=True, 
        strict=True))
    callbacks.append(ModelCheckpoint(
        filename='{epoch}-{step}-{val_loss:.3f}', 
        monitor='val_loss', 
        mode='min',
        verbose=True))
    
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        #precision=32, # 64, 32, 16-mixed, 'bf16-mixed'
        profiler=None, # None, "pytorch", "simple", 'advanced'
        deterministic=True)

    start = time.time()
    trainer.fit(model, data)
    print('Training time: {} s'.format(time.time()-start))

    # save checkpoint
    ckpt_path = f"CapAwareBandwidthPredictor-{config['eval_model']}.ckpt"
    trainer.save_checkpoint(ckpt_path)

    if config['eval_model'] == 'CapAware':
        model = CapAwareBandwidthPredictor.load_from_checkpoint(ckpt_path)
    elif config['eval_model'] == 'Perceive1000' or config['eval_model'] == 'Perceive300' or config['eval_model'] == 'Perceive100':
        model = Perceive.load_from_checkpoint(ckpt_path)
    elif config["eval_model"] == "SURE":
        model = SURE.load_from_checkpoint(ckpt_path)
    elif config['eval_model'] == 'UplinkNet':
        model = UplinkNet.load_from_checkpoint(ckpt_path)
    else:
        print('Unknown model')
        raise ValueError('Unknown model')

    start = time.time()
    trainer.test(model, data)
    print('Testing time: {} s'.format(time.time()-start))

    os.makedirs("predictions/", exist_ok=True)
    inputs = model.test_inputs
    labels = model.test_labels
    preds = model.test_predictions
    print('type labels: {}'.format(type(labels)))
    print('len labels: {}'.format(len(labels)))
    print('type preds: {}'.format(type(preds)))
    print('len preds: {}'.format(len(preds)))
    np.save(f"predictions/bandwidth-inputs-{config['dataset']}-{config['max_epochs']}.npy", inputs) # type: ignore
    np.save(f"predictions/bandwidth-labels-{config['dataset']}-{config['max_epochs']}.npy", labels) # type: ignore
    np.save(f"predictions/bandwidth-preds-{config['dataset']}-{config['max_epochs']}.npy", preds) # type: ignore

    start = time.time()
    predictions = trainer.predict(model, data)
    print('Prediction time: {} s'.format(time.time()-start))
    
def main():
    wandb.init(project='Uplink-Bandwidth-Prediction')
    train(config)
    wandb.finish()

def main_sweep():
    wandb.init(config=config, allow_val_change=True)
    #wandb.init(project='Uplink-Bandwidth-Prediction')

    print('wandb.config')
    print(wandb.config)
    config['eval_model'] = wandb.config.eval_model
    config['out_features'] = wandb.config.out_features
    config['hidden_size'] = wandb.config.hidden_size
    config['num_layers'] = wandb.config.num_layers
    config['num_linear_layers'] = wandb.config.num_linear_layers
    config['optimizer'] = wandb.config.optimizer
    config['dropout_rnn'] = wandb.config.dropout_rnn
    config['dropout_linear'] = wandb.config.dropout_linear
    config['learning_rate'] = wandb.config.learning_rate
    config['criterion'] = wandb.config.criterion
    
    config['penalty_over'] = wandb.config.penalty_over
    config['penalty_mild'] = wandb.config.penalty_mild
    config['penalty_deep'] = wandb.config.penalty_deep
    config['underutil_threshold'] = wandb.config.underutil_threshold
    config['exponent_over'] = wandb.config.exponent_over
    config['soft_factor'] = wandb.config.soft_factor
    
    config['activation'] = wandb.config.activation
    config['scaler'] = wandb.config.scaler
    config['bidirectional'] = wandb.config.bidirectional
    config['lr_scheduler'] = wandb.config.lr_scheduler
    config['seq_len'] = wandb.config.seq_len
    config['full_out'] = wandb.config.full_out
    print('assigned wandb.config')

    train(config)
    wandb.finish()

sweep_configuration = {
    'method': 'grid', # grid, random, bayes
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'},
    # 'early_terminate': {
    #     'type': 'hyperband',
    #     'min_iter': 3},
    'parameters': {
        'eval_model': {'values': ['CapAware', 'Perceive1000', 'Perceive300', 'Perceive100', 'SURE', 'UplinkNet']},
        'out_features': {'values': [1]},
        'hidden_size': {'values': [64]}, # , 64, 128, 256
        'num_layers': {'values': [3]}, # 3, 4, 5
        'num_linear_layers': {'values': [2]}, # 1, 2, 3
        'dropout_rnn': {'values': [0.1]},
        'dropout_linear': {'values': [0.1]},
        'learning_rate': {'values': [0.001]}, # 0.0001, 
        'criterion': {'values': ['MSELoss']}, # 'MSELoss', 'QuantileLoss', 'HybridARULoss', 'ARULossHO'

        'penalty_over': {'values': [5]}, # 4
        'penalty_mild': {'values': [0.4]}, # 0.5
        'penalty_deep': {'values': [0.8]},
        'underutil_threshold': {'values': [0.90]},
        'exponent_over': {'values': [2.0]},
        'soft_factor': {'values': [0.25]},

        'activation': {'values': ['ReLU']}, # , 'Softplus', 'ELU'
        'scaler': {'values': ['MinMaxScaler']}, # 'StandardScaler', 
        'bidirectional': {'values': [False]}, # True, 
        'optimizer': {'values': ['Adam']}, # 'Adam', 'AdamW'
        'lr_scheduler': {'values': ['ReduceLROnPlateau']}, #, 'StepLR' 
        'seq_len': {'values': [15]}, # , 32, 64
        'full_out': {'values': [False]} #True, 
    },
}

config = dict(
    # Setting common hyperparameters for model
    model_type='LSTM', # GRU, LSTM
    hidden_size=64,
    num_layers=3,
    num_linear_layers=2,
    dropout_rnn=0.1, # 0.1
    dropout_linear=0.1, # 0.1
    learning_rate=0.001,
    criterion='ARULossHO', # MSELoss, L1Loss, QuantileLoss, ARULoss, HybridARULoss

    penalty_over=4.0,
    penalty_mild=0.4,
    penalty_deep=0.8,
    underutil_threshold=0.90,
    exponent_over=2.0,
    soft_factor=0.25,

    activation='ReLU', # ReLU, Softplus, GELU, LeakyReLU, ELU
    scaler='MinMaxScaler', # MinMaxScaler, StandardScaleR
    optimizer='Adam', # Adam or AdamW
    lr_scheduler='ReduceLROnPlateau', # StepLR or ReduceLROnPlateau
    bidirectional=False,
    batch_first=True,

    # Datamodule Parameters
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    PARQUET = False, # True, False
    # 6-2-2 Train-Val-Test Split 
    train_p=0.6,
    val_p=0.2,
    test_p=0.2,

    # Directories
    working_dir='./',
    log_dir="./data/logs/",
    plot_dir="./data/plots/",
    outputs_dir="./data/outputs/",
    checkpoints_dir="./data/checkpoints/",
    ray_dir='./data/ray_results',
    results_dir='./data/ray_tune_results/',
    model_save='./data/model_save',
    # Logging
    logging_name='base',
    experiment_name='1000-epochs-grid-batch-size',
    # Trainer
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='prediction',
        description='Training a Bandwidth Prediction model')

    parser.add_argument('-eval_model', type=str, default='CapAware') # CapAware, Perceive1000, Perceive300, Perceive100, SURE, UplinkNet
    parser.add_argument('-model_type', type=str, default='LSTM')
    parser.add_argument('-pred_len', type=int, default=1) # Number of steps to make predictions
    parser.add_argument('-seq_len', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-max_epochs', type=int, default=1) # 1, 1000
    parser.add_argument('-inverse', type=bool, default=True)
    parser.add_argument('-dataset', type=str, default='Fjord5G-4329-uplink') # Fjord5G-4329-uplink, SURE-uplink, UplinkNet-uplink
    parser.add_argument('-gpu', type=int, default=0) # 0, 1
    parser.add_argument('-use_handover', type=bool, default=True) # Handover-aware loss or not

    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

    if torch.cuda.is_available():
        config['fused'] = True
    else:
        config['fused'] = False

    if args.eval_model == 'Perceive1000':
        args.seq_len = 50
    elif args.eval_model == 'Perceive300':
        args.seq_len = 15
    elif args.eval_model == 'Perceive100':
        args.seq_len = 5

    config['eval_model'] = args.eval_model
    config['model_type'] = args.model_type
    config['pred_len'] = args.pred_len
    config['seq_len'] = args.seq_len
    config['batch_size'] = args.batch_size
    config['max_epochs'] = args.max_epochs
    config['inverse'] = args.inverse
    config['dataset'] = args.dataset
    config['use_handover'] = args.use_handover

    print_environment()
    print(config)
    main()

    # used for W&B Sweeps
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Uplink-Bandwidth-Prediction')
    # print(sweep_id)
    # wandb.agent(sweep_id, function=main_sweep)