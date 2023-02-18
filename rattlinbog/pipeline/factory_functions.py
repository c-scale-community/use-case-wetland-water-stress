from rattlinbog.estimators.base import ValidationLogging, LogConfig
from rattlinbog.evaluate.validation_source_from_dataset import ValidationSourceFromDataset


def make_validation_log_cfg(valid_ds, train_log, valid_log, score_freq, image_freq):
    return LogConfig(train_log, ValidationLogging(valid_log, ValidationSourceFromDataset(valid_ds),
                                                  score_freq, image_freq))
