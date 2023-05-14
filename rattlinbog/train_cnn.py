import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from pandas import DataFrame
from torch.utils.tensorboard import SummaryWriter

from rattlinbog.estimators.wetland_classifier import WetlandClassifier
from rattlinbog.io_xarray.concatenate import concatenate_training_datasets, concatenate_indices_dataset
from rattlinbog.persist.serialize_best_scoring_nn_model import SerializeBestScoringNNModel
from rattlinbog.pipeline.factory_functions import make_validation_log_cfg
from rattlinbog.pipeline.train import train
from rattlinbog.th_extensions.nn.unet import UNet
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY

DATA_ROOT = Path("/data/wetland/")


def main(dataset_type: str) -> None:
    train_mosaic, valid_mosaic = retrieve_train_and_valid_mosaics(dataset_type)

    sample_df = retrieve_sample_df()
    sample_train_df = sample_df[sample_df['tile_name'] != 'E060N012T3']

    sample_mosaic = concatenate_indices_dataset(*sample_train_df['filepath'])
    unet = make_unet_for(dataset_type)

    fit_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    train_writer = SummaryWriter(f"/data/scaled-runs/{dataset_type}/{fit_time}/train")
    valid_writer = SummaryWriter(f"/data/scaled-runs/{dataset_type}/{fit_time}/valid")
    model_out = Path(f'/data/wetland/models/{dataset_type}/{fit_time}')
    model_out.mkdir(parents=True, exist_ok=True)
    model_sink = SerializeBestScoringNNModel(model_out, score='F1')

    log_cfg = make_validation_log_cfg(valid_mosaic, train_writer, valid_writer, 10, 20, model_sink)

    estimator = WetlandClassifier(unet, batch_size=16, log_cfg=log_cfg)
    train(estimator, train_mosaic, sample_mosaic, 96000)


def retrieve_train_and_valid_mosaics(dataset_type):
    params_df = retrieve_params_df(dataset_type)
    train_df = params_df[params_df['tile_name'] != 'E060N012T3']
    valid_df = params_df[params_df['tile_name'] == 'E060N012T3']
    train_mosaic = concatenate_training_datasets(*train_df['filepath'])
    valid_mosaic = concatenate_training_datasets(*valid_df['filepath'])
    valid_mosaic = valid_mosaic.sel(y=slice(1300000, 1200000), x=slice(6200000, 6300000))

    if dataset_type == 'hparam':
        train_mosaic = _preprocess_hparam(train_mosaic)
        valid_mosaic = _preprocess_hparam(valid_mosaic)

    valid_mosaic = valid_mosaic.persist()
    return train_mosaic, valid_mosaic


def retrieve_params_df(dataset_type: str) -> DataFrame:
    params_df = gather_files(DATA_ROOT, yeoda_naming_convention, [
        re.compile(dataset_type),
        re.compile('V1M0R1'),
        re.compile('EQUI7_EU020M'),
        re.compile('E\d\d\dN\d\d\dT3')
    ])
    return params_df


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


def retrieve_sample_df():
    return gather_files(DATA_ROOT, yeoda_naming_convention, [
        re.compile('samples'),
        re.compile('V1M0R1'),
        re.compile('EQUI7_EU020M'),
        re.compile('E\d\d\dN\d\d\dT3')
    ])


def make_unet_for(dataset_type: str):
    if dataset_type == 'hparam':
        return UNet(3, [128, 256, 512, 1024], 1).to(device=th.device('cuda'))
    elif dataset_type == 'mmean':
        return UNet(24, [128, 256, 512, 1024], 1).to(device=th.device('cuda'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN based on input dataset type")
    parser.add_argument("dataset", help="Dataset type (hparam or mmean)", type=str)
    args = parser.parse_args()
    main(args.dataset)
