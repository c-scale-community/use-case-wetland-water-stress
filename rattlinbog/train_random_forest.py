import argparse
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import xarray as xr
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from rattlinbog.loaders import DATE_FORMAT


def train_random_forest(data_zarr: Path, dst: Path, n_iter=10):
    dataset = xr.open_zarr(data_zarr)
    mask = dataset['mask'].fillna(0)
    indices_0 = np.nonzero(np.logical_not(mask)).values
    indices_1 = np.nonzero(mask).values

    choices_0 = np.arange(indices_0.shape[1])
    np.random.shuffle(choices_0)
    choices_1 = np.arange(indices_1.shape[1])
    np.random.shuffle(choices_1)
    if indices_0.shape[1] > indices_1.shape[1]:
        choices_0 = choices_0[:indices_1.shape[1]]
    else:
        choices_1 = choices_1[:indices_0.shape[1]]

    balanced_sample_indices = np.concatenate([indices_0[:, choices_0], indices_1[:, choices_1]], axis=1)

    samples = dataset['params'].isel(y=xr.DataArray(balanced_sample_indices[0], dims="samples"),
                                     x=xr.DataArray(balanced_sample_indices[1], dims="samples"))
    labels = mask.isel(y=xr.DataArray(balanced_sample_indices[0], dims="samples"),
                       x=xr.DataArray(balanced_sample_indices[1], dims="samples"))

    param_space = {
        'bootstrap': [True, False],
        'n_estimators': [10, 100, 200, 400, 600, 800, 1000],
        'criterion': ["gini", "entropy", "log_loss"],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ["sqrt", "log2", None],
    }

    clf = RandomForestClassifier()
    search = RandomizedSearchCV(clf, param_space, cv=3, n_iter=n_iter, n_jobs=-1)
    fitted = search.fit(samples.T, labels)
    DataFrame(fitted.cv_results_).to_csv(dst / f"{datetime.now().strftime(DATE_FORMAT)}_cv_results.csv")
    print(fitted.best_params_)

    estimator = fitted.best_estimator_
    dst_estimator = dst / f"{datetime.now().strftime(DATE_FORMAT)}_best_model.joblib"
    logging.info(f"storing best estimator at: {dst_estimator}")
    joblib.dump(estimator, dst_estimator)

    def estimate(a, model):
        p = model.predict(a.reshape(-1, a.shape[-1]))
        return p.reshape(a.shape[:2])

    estimated = xr.apply_ufunc(estimate, dataset['params'], input_core_dims=[['parameter']],
                               kwargs=dict(model=fitted),
                               dask='parallelized', output_dtypes=np.float32)
    estimated.load(scheduler='processes').plot.imshow()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train random a random forest algorithm on the given data")
    parser.add_argument("data_zarr", help="Path to zarr archive containing the data", type=Path)
    parser.add_argument("dst_root", help="Path to store the best model and CV statistics", type=Path)
    parser.add_argument("n_iter", help="Number of random search iterations", type=int)
    args = parser.parse_args()
    train_random_forest(args.data_zarr, args.dst_root, args.n_iter)
