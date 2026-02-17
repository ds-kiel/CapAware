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

from models import CapAwareHandoverPredictor, RSRPHandoverPredictor
from data_module_handover import HandoverDataModule
from utility import print_environment

def train(config):

    pl.seed_everything(42, workers=True)
    wandb_logger = WandbLogger(
        project='Handover-Prediction', 
        save_code=True)
    
    run_id = wandb_logger.experiment.id

    data_module = HandoverDataModule(config, run_id)
    data_module.prepare_data()
    data_module.setup()

    config['input_size'] = data_module.input_size
    config['pred_len'] = data_module.pred_len
    print('input_size: {}'.format(data_module.input_size))
    print('pred_len: {}'.format(data_module.pred_len))

    if config['model'] == 'CapAwareHandoverPredictor':
        model = CapAwareHandoverPredictor(config)
    elif config['model'] == 'RSRPHandoverPredictor':
        model = RSRPHandoverPredictor(config)
    else:
        print('Unknown model: {}'.format(config['model']))
        return

    callbacks = []
    callbacks.append(EarlyStopping(
        monitor='val_loss', 
        patience=10,
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
    trainer.fit(model, data_module)
    print('Training time: {} s'.format(time.time()-start))

    # save checkpoint
    ckpt_path = "CapAwareHandoverPredictor.ckpt"
    trainer.save_checkpoint(ckpt_path)

    if config['model'] == 'CapAwareHandoverPredictor':
        model = CapAwareHandoverPredictor.load_from_checkpoint(ckpt_path)
        
    elif config['model'] == 'RSRPHandoverPredictor':
        model = RSRPHandoverPredictor.load_from_checkpoint(ckpt_path)
    else:
        print('Unknown model')
        raise ValueError('Unknown model')

    start = time.time()
    trainer.test(model, data_module)
    print('Testing time: {} s'.format(time.time()-start))

    os.makedirs("predictions/", exist_ok=True)
    labels = model.test_labels
    preds = model.test_preds
    print('type labels: {}'.format(type(labels)))
    print('len labels: {}'.format(len(labels)))
    print('type preds: {}'.format(type(preds)))
    print('len preds: {}'.format(len(preds)))
    np.save(f"predictions/handover-labels-{config['dataset']}-{config['max_epochs']}.npy", labels) # type: ignore
    np.save(f"predictions/handover-preds-{config['dataset']}-{config['max_epochs']}.npy", preds) # type: ignore

    # possibility to load a different module for prediction
    data_module_predict = HandoverDataModule(
        config_predict,
        run_id,
        external_scaler_feature = data_module.scaler_feature,
        external_scaler_label   = data_module.scaler_label,
        make_splits             = False          # use the whole file
    )

    # start = time.time()
    # trainer.test(model, data_module)
    # print('Dataset2 - Testing time: {} s'.format(time.time()-start))

    # labels = model.test_labels
    # preds = model.test_preds
    # print('Dataset2 - type labels: {}'.format(type(labels)))
    # print('Dataset2 - len labels: {}'.format(len(labels)))
    # print('Dataset2 - type preds: {}'.format(type(preds)))
    # print('Dataset2 - len preds: {}'.format(len(preds)))

    start = time.time()
    predictions = trainer.predict(model, data_module_predict)
    print('Prediction time: {} s'.format(time.time()-start))

    # print(type(predictions))
    # print(len(predictions))
    # print(len(predictions[0]))
    # print((predictions[0]))

    preds = model.predictions
    probs = model.probabilities
    print('Dataset2 - type preds: {}'.format(type(preds)))
    print('Dataset2 - len preds: {}'.format(len(preds)))
    print('Dataset2 - type probs: {}'.format(type(probs)))
    print('Dataset2 - len probs: {}'.format(len(probs)))

    print('predictions.shape: {}, probabilities.shape: {}'.format(preds.shape, probs.shape))
    print('predictions[0]: {}, probabilities[0]: {}'.format(preds[0], probs[0]))

    # Convert to DataFrame for evaluation
    df = pd.DataFrame({
        'Predictions': preds,
        'Probabilities': probs
    })
    print(df.info())
    print(df.describe())
    print(df['Predictions'].value_counts())
    #df.to_parquet(f'{self.dataset}-handover-preds-and-probs.parquet.gzip', compression='gzip')

    # test on predict dataset but model is trained with train config
    np.save(f"predictions/handover-predict-preds-{config_predict['dataset']}-{config['max_epochs']}.npy", preds) # type: ignore
    np.save(f"predictions/handover-predict-probs-{config_predict['dataset']}-{config['max_epochs']}.npy", probs) # type: ignore


def main():
    wandb.init(project='Handover-Prediction')
    train(config)
    wandb.finish()

def main_sweep():
    wandb.init(config=config, allow_val_change=True)
    #wandb.init(project='Handover-Prediction')

    print('wandb.config')
    print(wandb.config)
    #config['pred_len'] = wandb.config.pred_len
    config['hidden_size'] = wandb.config.hidden_size
    config['num_layers'] = wandb.config.num_layers
    # config['num_linear_layers'] = wandb.config.num_linear_layers
    # config['optimizer'] = wandb.config.optimizer
    # config['dropout_rnn'] = wandb.config.dropout_rnn
    # config['dropout_linear'] = wandb.config.dropout_linear
    # config['learning_rate'] = wandb.config.learning_rate
    # config['criterion'] = wandb.config.criterion
    # config['activation'] = wandb.config.activation
    # config['scaler'] = wandb.config.scaler
    # config['bidirectional'] = wandb.config.bidirectional
    # config['lr_scheduler'] = wandb.config.lr_scheduler
    config['seq_len'] = wandb.config.seq_len
    #config['full_out'] = wandb.config.full_out
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
        'hidden_size': {'values': [8, 16]}, # , 64, 128, 256
        'num_layers': {'values': [3, 4, 5]}, # 3, 4, 5
        'seq_len': {'values': [32, 64, 128]}, # , 32, 64
    },
}

config = dict(
    model='CapAwareHandoverPredictor', # CapAwareHandoverPredictor, RSRPHandoverPredictor
    batch_size=32,
    seq_len=16,
    pred_len=1,
    input_size=None,
    hidden_size=16,
    num_layers=4,
    dropout=0.1,
    learning_rate=0.001,
    threshold=0.5,
    max_epochs=1,
    negative_ratio=1.0,
    balance_data=True,  # Set to False to disable balancing
    dataset='Fjord5G-4312',
)

config_predict = dict(
    model='CapAwareHandoverPredictor', # CapAwareHandoverPredictor, RSRPHandoverPredictor
    batch_size=8192,
    seq_len=16,
    pred_len=1,
    input_size=None,
    hidden_size=16,
    num_layers=4,
    dropout=0.1,
    learning_rate=0.001,
    threshold=0.5,
    max_epochs=1,
    negative_ratio=1.0,
    balance_data=False,  # Set to False to disable balancing
    dataset='Fjord5G-4312', # you can use other routers from Fjord5G dataset
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='prediction',
        description='Training a Handover Prediction model')

    parser.add_argument('-max_epochs', type=int, default=1)
    parser.add_argument('-gpu', type=int, default=0) # 0, 1
    
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

    config['max_epochs'] = args.max_epochs
    config_predict['max_epochs'] = args.max_epochs

    print_environment()
    print(config)
    main()

    # used for W&B Sweeps
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Handover-Prediction')
    # print('sweep_id: {}'.format(sweep_id))
    # wandb.agent(sweep_id, function=main_sweep)