import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import yaml
from torch.utils.tensorboard import SummaryWriter

from rattlinbog.config import TrainCNN
from rattlinbog.estimators.wetland_classifier import WetlandClassifier
from rattlinbog.filesystem import retrieve_params_df, retrieve_sample_df
from rattlinbog.io_xarray.concatenate import concatenate_training_datasets, concatenate_indices_dataset
from rattlinbog.persist.serialize_best_scoring_nn_model import SerializeBestScoringNNModel
from rattlinbog.pipeline.factory_functions import make_validation_log_cfg
from rattlinbog.pipeline.train import train
from rattlinbog.th_extensions.nn.unet import UNet
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY

OPEN_WATER_MEAN_BSC = -18.85


def main(config: TrainCNN) -> None:
    train_mosaic, valid_mosaic = retrieve_train_and_valid_mosaics(config)

    sample_df = retrieve_sample_df(config.samples_selection)
    sample_train_df = sample_df[sample_df['tile_name'] != 'E060N012T3']

    sample_mosaic = concatenate_indices_dataset(*sample_train_df['filepath'])
    dataset_type = config.parameter_selection.parameter_type
    gt_type = extract_gt_type(config)
    unet = make_unet_for(dataset_type)

    fit_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    train_writer = SummaryWriter(f"/data/scaled-runs/{dataset_type}/{gt_type}/{fit_time}/train")
    valid_writer = SummaryWriter(f"/data/scaled-runs/{dataset_type}/{gt_type}/{fit_time}/valid")
    model_out = Path(f'/data/wetland/models/{dataset_type}/{gt_type}/{fit_time}')
    model_out.mkdir(parents=True, exist_ok=True)
    model_sink = SerializeBestScoringNNModel(model_out, score='F1')

    log_cfg = make_validation_log_cfg(valid_mosaic, train_writer, valid_writer, 10, 20, model_sink)

    estimator = WetlandClassifier(unet, batch_size=16, log_cfg=log_cfg)
    train(estimator, train_mosaic, sample_mosaic, 96000)


def extract_gt_type(config):
    tokens = config.parameter_selection.var_name.split('-')
    gt_type = '-'.join(tokens[tokens.index('MASK') + 1:])
    return gt_type


def retrieve_train_and_valid_mosaics(config: TrainCNN):
    params_df = retrieve_params_df(config.parameter_selection)
    train_df = params_df[params_df['tile_name'] != 'E060N012T3']
    valid_df = params_df[params_df['tile_name'] == 'E060N012T3']
    train_mosaic = concatenate_training_datasets(*train_df['filepath'])
    valid_mosaic = concatenate_training_datasets(*valid_df['filepath'])
    valid_mosaic = valid_mosaic.sel(y=slice(1300000, 1200000), x=slice(6200000, 6300000))

    dataset_type = config.parameter_selection.parameter_type
    if dataset_type == 'hparam':
        train_mosaic = _preprocess_hparam(train_mosaic)
        valid_mosaic = _preprocess_hparam(valid_mosaic)
    elif dataset_type == 'mmean':
        train_mosaic = _preprocess_mmean(train_mosaic)
        valid_mosaic = _preprocess_mmean(valid_mosaic)

    valid_mosaic = valid_mosaic.persist()
    return train_mosaic, valid_mosaic


def _preprocess_hparam(mosaic):
    mosaic = mosaic.sel(parameter=['SIG0-HPAR-PHS', 'SIG0-HPAR-AMP', 'SIG0-HPAR-M0'])
    mosaic[PARAMS_KEY] = mosaic[PARAMS_KEY].map_blocks(preprocess_rgb_comp, template=mosaic[PARAMS_KEY])
    return mosaic


def preprocess_rgb_comp(x):
    o = x.copy()
    o[0, ...] = normalize(x[0], -np.pi, np.pi)
    o[1, ...] = normalize(np.clip((10 ** (x[1] / 10.0)), 1, 1.5), 1.0, 1.5)
    o[2, ...] = normalize(np.clip(x[2], -25, -8), -25, -8)
    return o


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _preprocess_mmean(mosaic):
    mosaic[PARAMS_KEY] = mosaic[PARAMS_KEY].map_blocks(preprocess_sig0, template=mosaic[PARAMS_KEY])
    return mosaic


def preprocess_sig0(x):
    return x.fillna(OPEN_WATER_MEAN_BSC)


def make_unet_for(dataset_type: str):
    if dataset_type == 'hparam':
        return UNet(3, [128, 256, 512, 1024], 1).to(device=th.device('cuda'))
    elif dataset_type == 'mmean':
        return UNet(24, [128, 256, 512, 1024], 1).to(device=th.device('cuda'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN based on input dataset type")
    parser.add_argument("config", help="Config file", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    main(cfg)
