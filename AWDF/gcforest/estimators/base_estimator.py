# -*- coding:utf-8 -*-
"""
Description: A python 3.5 implementation of gcForest modification: "A new Adaptive Weighted Deep Forest".
             It is based on the original implementation of gcForest: https://github.com/kingfengji/gcForest

ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn) and repository author.
ATTN2: This package was originally developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn) and then modified by repository author. The readme file and demo roughly explains how to use the codes.
"""
import os, os.path as osp
import numpy as np

from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path

LOGGER = get_logger("gcforest.estimators.base_estimator")

def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

class BaseClassifierWrapper(object):
    def __init__(self, name, est_class, est_args):
        """
        name: str)
            Used for debug and as the filename this model may be saved in the disk
        """
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.cache_suffix = ".pkl"
        self.est = None

    def _init_estimator(self):
        """
        You can re-implement this function when inherient this class
        """
        est = self.est_class(**self.est_args)
        return est

    def fit(self, X, y, cache_dir=None, sample_weight=None):
        """
        cache_dir(str):
            if not None
                then if there is something in cache_dir, dont have fit the thing all over again
                otherwise, fit it and save to model cache

        sample_weight:
            Sample weights.
        """
        LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if self._is_cache_exists(cache_path):
            LOGGER.info("Find estimator from {} . skip process".format(cache_path))
            return
        est = self._init_estimator()
        self._fit(est, X, y, sample_weight=sample_weight)
        if cache_path is not None:
            # saved in disk
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path);
            self._save_model_to_disk(est, cache_path)
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
        return y_proba

    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _is_cache_exists(self, cache_path):
        return cache_path is not None and osp.exists(cache_path)

    def _batch_predict_proba(self, est, X, batch_size):
        LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            LOGGER.info("[progress][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            y_cur = self._predict_proba(est, X[j:j+batch_size])
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
            y_pred_proba[j:j+batch_size,:] = y_cur
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _load_model_from_disk(self, cache_path):
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        raise NotImplementedError()

    def _default_predict_batch_size(self, est, X):
        """
        You can re-implement this function when inherient this class

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba without baches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _fit(self, est, X, y, sample_weight=None):
        est.fit(X, y, sample_weight=sample_weight)

    def _predict_proba(self, est, X):
        return est.predict_proba(X)


class BaseBaggingClassifierWrapper(object):
    """BaseBaggingClassifierWrapper is the same to BaseClassifierWrapper,
    except that the fit method uses sample_weight for bootstrapping.
    """
    def __init__(self, name, est_class, est_args):
        """
        name: str)
            Used for debug and as the filename this model may be saved in the disk
        """
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.cache_suffix = ".pkl"
        self.est = None

    def _init_estimator(self):
        """
        You can re-implement this function when inherient this class
        """
        est = self.est_class(**self.est_args)
        return est

    def fit(self, X, y, cache_dir=None, sample_weight=None):
        """
        cache_dir(str):
            if not None
                then if there is something in cache_dir, dont have fit the thing all over again
                otherwise, fit it and save to model cache

        sample_weight:
            Sample weights.
        """
        LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if self._is_cache_exists(cache_path):
            LOGGER.info("Find estimator from {} . skip process".format(cache_path))
            return
        est = self._init_estimator()
        self._fit(est, X, y, sample_weight=sample_weight)
        if cache_path is not None:
            # saved in disk
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path);
            self._save_model_to_disk(est, cache_path)
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
        return y_proba

    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _is_cache_exists(self, cache_path):
        return cache_path is not None and osp.exists(cache_path)

    def _batch_predict_proba(self, est, X, batch_size):
        LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            LOGGER.info("[progress][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            y_cur = self._predict_proba(est, X[j:j+batch_size])
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
            y_pred_proba[j:j+batch_size,:] = y_cur
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _load_model_from_disk(self, cache_path):
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        raise NotImplementedError()

    def _default_predict_batch_size(self, est, X):
        """
        You can re-implement this function when inherient this class

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba without baches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _fit(self, est, X, y, sample_weight=None):
        # est.fit(X, y, sample_weight=sample_weight)
        # Bootstrap X and y according to sample_weight
        weights = np.array(sample_weight.copy()) / sum(sample_weight)
        ind = np.random.choice(range(X.shape[0]), X.shape[0], p=weights)
        res_X, res_y = X[ind], y[ind]
        # for classification task y should contain at least 1 sample per class:
        # append randomly selected sample for each class that is missing
        print("res_X shape before: {}".format(res_X.shape))
        res_y_unique = np.unique(res_y)
        y_unique = np.unique(y)
        y_missing = set(y_unique) - set(res_y_unique)
        for c in y_missing:
            ind_of_c = np.where(y == c)[0].flatten()
            ind_size = len(ind_of_c)
            eps = 1e-6
            tmp_weights = (weights[ind_of_c] + eps) / (sum(weights[ind_of_c]) + eps * ind_size)
            ind = np.random.choice(ind_of_c, 1, p=tmp_weights)[0]
            # ind = ind_of_c[0]
            res_X = np.vstack((res_X, X[ind]))
            res_y = np.array(list(res_y) + [y[ind]])

        print("res_X shape after: {}".format(res_X.shape))
        LOGGER.info("Fit: weighted bootstrap")
        est.fit(res_X, res_y)

    def _predict_proba(self, est, X):
        return est.predict_proba(X)
