from rattlinbog.estimators.base import ValidationLogging, ImageLogging, LogConfig
from rattlinbog.evaluate.image_producer_from_data_array import ImageProducerFromDataArray
from rattlinbog.evaluate.validator_of_dataset import ValidatorOfDataset
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY


def make_validation_log_cfg(valid_ds, train_log, valid_log, score_freq, image_freq):
    score_cfg = None
    if score_freq > 0:
        score_cfg = ValidationLogging(score_freq, valid_log, ValidatorOfDataset(valid_ds))

    image_cfg = None
    if image_freq > 0:
        image_producer = ImageProducerFromDataArray(valid_ds[PARAMS_KEY])
        image_cfg = ImageLogging(image_freq, valid_log, image_producer)

    return LogConfig(train_log, score_cfg, image_cfg)
