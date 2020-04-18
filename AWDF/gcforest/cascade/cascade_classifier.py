# -*- coding:utf-8 -*-
"""
Description: A python 3.5 implementation of gcForest modification: "A new Adaptive Weighted Deep Forest".
             It is based on the original implementation of gcForest: https://github.com/kingfengji/gcForest

ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn) and repository author.
ATTN2: This package was originally developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn) and then modified by repository author. The readme file and demo roughly explains how to use the codes.
"""
import numpy as np
import os
import os.path as osp
import pickle

from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb

LOGGER = get_logger('gcforest.cascade.cascade_classifier')


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


class CascadeClassifier(object):
    def __init__(self, ca_config):
        """
        Parameters (ca_config)
        ----------
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in early_stopping_rounds, the cascade level will stop automatically growing
        max_layers: int
            maximum number of cascade layers allowed for exepriments, 0 means use Early Stoping to automatically find the layer number
        n_classes: int
            Number of classes
        est_configs:
            List of CVEstimator's config
        look_indexs_cycle (list 2d): default=None
            specification for layer i, look for the array in look_indexs_cycle[i % len(look_indexs_cycle)]
            defalut = None <=> [range(n_groups)]
            .e.g.
                look_indexs_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3; layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermidiate results in data_save_dir
            if data_save_rounds = 0, then no savings for intermidiate results
        """
        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        # self.data_save_dir = self.get_value("data_save_dir", None, basestring)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.opt_layer_num = -1
        self.stack_all_probas = self.get_value("stack_all_probas", False, bool)

        self.probability_based_weighting = self.get_value("probability_based_weighting", False, bool) # True
        if self.probability_based_weighting:
            LOGGER.info("Using probability based weighting (pass weights to the next layer on fit)")
            weighting_function = self.get_value("weighting_function", "linear", str)
            if weighting_function == "linear":
                def linear_wf(corr_proba, proba, y_train, x_train=None):
                    return 1 - corr_proba

                self.weighting_function = linear_wf
            elif weighting_function == "l2":
                def l2_wf(corr_proba, proba, y_train, x_train=None):
                    train_proba = np.array([[1 if j == y_train[i] else 0 for j in range(proba.shape[1])] for i in range(proba.shape[0])])
                    return np.linalg.norm(proba - train_proba, axis=1)

                self.weighting_function = l2_wf
            elif weighting_function == "1-w^1/2":
                def sqrt_wf(corr_proba, proba, y_train, x_train=None):
                    return 1 - corr_proba ** (1 / 2)

                self.weighting_function = sqrt_wf
            elif weighting_function == "1-w2":
                def sqr_wf(corr_proba, proba, y_train, x_train=None):
                    return 1 - corr_proba ** 2

                self.weighting_function = sqr_wf
            elif weighting_function == "contrastive":
                def contrastive_wf(corr_proba, proba, y_train, x_train=None):
                    def geo_mean(a):
                        return a.prod()**(1.0/len(a))

                    def geo_mean_overflow(iterable):
                        a = np.log(iterable)
                        return np.exp(a.sum()/len(a))

                    def d(a, b):
                        return np.linalg.norm(b - a)

                    def d_vec(a, b):
                        return np.linalg.norm(b - a, axis=1)

                    tau = 0.75
                    alpha = 0.9
                    n_elems = y_train.shape[0]

                    # prepare z_ij
                    z_i = np.zeros((n_elems, n_elems))
                    for i in range(n_elems):
                        z_i[i] = (y_train == y_train[i]).astype('int')

                    # prepare d_ij
                    d_i = np.zeros((n_elems, n_elems))
                    for i in range(n_elems):
                        d_i[i] = d_vec(proba[i], proba) * alpha + (1 - alpha) * d_vec(x_train[i], x_train)

                    d_i_min = np.min(d_i)
                    d_i_max = np.max(d_i)
                    d_i -= d_i_min
                    d_i /= d_i_max - d_i_min

                    # tau = np.max(d_ij) / 2
                    loss_i = (1 - z_i) * d_i + z_i * np.maximum(tau - d_i, 0)

                    np.fill_diagonal(loss_i, 0) # remove diagonal elements
                    weights = np.mean(loss_i, axis=1)
                    print(np.max(weights))
                    print(np.min(weights))
                    
                    weights_exp = np.exp(weights)
                    weights = weights_exp / np.sum(weights_exp, axis=0)

                    # np.save("test_weights.npy", weights)
                    # np.save("test_proba.npy", proba)
                    # np.save("test_xtrain.npy", x_train)
                    # np.save("test_ytrain.npy", y_train)

                    return weights

                def sparse_contrastive_wf(corr_proba, proba, y_train, x_train=None, parts=100):
                    print("Randomly permute indices")
                    indices = np.random.permutation(len(corr_proba))
                    n_elems = y_train.shape[0]
                    weights = np.zeros((n_elems,))
                    offset = 0
                    n_elems_in_part = n_elems // parts
                    print("Start weights generation")
                    for i in range(parts):
                        end = min(n_elems, offset + n_elems_in_part)
                        weights[offset:end] = contrastive_wf(corr_proba[offset:end],
                                                             proba[offset:end],
                                                             y_train[offset:end],
                                                             x_train[offset:end])
                        offset += n_elems_in_part
                        print("  Temporary weights generated ({})".format(i))
                    print("All weights have been generated")
                    return weights

                self.weighting_function = sparse_contrastive_wf
            else:
                raise Exception("Incorrect weighting function")

        self.screening_probability = self.get_value("screening_probability", None, float)
        if self.screening_probability:
            LOGGER.info("Using confidence screening")
        # LOGGER.info("\n" + json.dumps(ca_config, sort_keys=True, indent=4, separators=(',', ':')))

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="cascade")

    def _set_estimator(self, li, ei, est):
        if li not in self.estimator2d:
            self.estimator2d[li] = {}
        self.estimator2d[li][ei] = est

    def _get_estimator(self, li, ei):
        return self.estimator2d.get(li, {}).get(ei, None)

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def _check_look_indexs_cycle(self, X_groups, is_fit):
        # check look_indexs_cycle
        n_groups = len(X_groups)
        if is_fit and self.look_indexs_cycle is None:
            look_indexs_cycle = [list(range(n_groups))]
        else:
            look_indexs_cycle = self.look_indexs_cycle
            for look_indexs in look_indexs_cycle:
                if np.max(look_indexs) >= n_groups or np.min(look_indexs) < 0 or len(look_indexs) == 0:
                    raise ValueError("look_indexs doesn't match n_groups!!! look_indexs={}, n_groups={}".format(
                        look_indexs, n_groups))
        if is_fit:
            self.look_indexs_cycle = look_indexs_cycle
        return look_indexs_cycle

    def _check_group_dims(self, X_groups, is_fit):
        if is_fit:
            group_starts, group_ends, group_dims = [], [], []
        else:
            group_starts, group_ends, group_dims = self.group_starts, self.group_ends, self.group_dims
        n_datas = X_groups[0].shape[0]
        X = np.zeros((n_datas, 0), dtype=X_groups[0].dtype)
        for i, X_group in enumerate(X_groups):
            assert(X_group.shape[0] == n_datas)
            X_group = X_group.reshape(n_datas, -1)
            if is_fit:
                group_dims.append( X_group.shape[1] )
                group_starts.append(0 if i == 0 else group_ends[i - 1])
                group_ends.append(group_starts[i] + group_dims[i])
            else:
                assert(X_group.shape[1] == group_dims[i])
            X = np.hstack((X, X_group))
        if is_fit:
            self.group_starts, self.group_ends, self.group_dims = group_starts, group_ends, group_dims
        return group_starts, group_ends, group_dims, X

    def probability_to_sample_weight(self, proba, y_train, x_train):
        if not self.probability_based_weighting:
            return np.ones((proba.shape[0],), dtype=np.float32)

        print("Proba shape: {}".format(proba.shape))
        print("y_train shape: {}".format(y_train.shape))
        corr_proba = np.array([proba[i, y_train[i]] for i in range(len(y_train))])
        weights = self.weighting_function(corr_proba, proba, y_train, x_train)
        print("weights shape: {}".format(weights.shape))
        return weights.copy()

    def fit_transform(self, X_groups_train, y_train_orig, X_groups_test, y_test, stop_by_test=False, train_config=None):
        """
        fit until the accuracy converges in early_stop_rounds
        stop_by_test: (bool)
            When X_test, y_test is validation data that used for determine the opt_layer_id,
            use this option
        """
        if train_config is None:
            from ..config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir

        y_train = y_train_orig.copy()

        is_eval_test = "test" in train_config.phases
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        if is_eval_test and not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_train.shape={},y_train.shape={},X_groups_test.shape={},y_test.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape,
            [xt.shape for xt in X_groups_test] if is_eval_test else "no_test", y_test.shape if is_eval_test else "no_test"))

        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_train, True)
        if is_eval_test:
            self._check_look_indexs_cycle(X_groups_test, False)

        # check groups dimension
        group_starts, group_ends, group_dims, X_train = self._check_group_dims(X_groups_train, True)
        if is_eval_test:
            _, _, _, X_test = self._check_group_dims(X_groups_test, False)
        else:
            X_test = np.zeros((0, X_train.shape[1]))
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("group_starts={}".format(group_starts))
        LOGGER.info("group_ends={}".format(group_ends))
        LOGGER.info("X_train.shape={},X_test.shape={}".format(X_train.shape, X_test.shape))

        n_trains = X_groups_train[0].shape[0]
        n_tests = X_groups_test[0].shape[0] if is_eval_test else 0

        n_classes = self.n_classes
        assert n_classes == len(np.unique(y_train)), "n_classes({}) != len(unique(y)) {}".format(n_classes, np.unique(y_train))
        train_acc_list = []
        test_acc_list = []
        # X_train, y_train, X_test, y_test
        opt_datas = [None, None, None, None]
        try:
            # probability of each cascades's estimators
            X_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            X_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
            X_cur_train, X_cur_test = None, None
            cur_train_sample_weight = None
            layer_id = 0

            X_all_proba_train = np.zeros((n_trains, 0), dtype=np.float32)
            X_all_proba_test  = np.zeros((n_tests, 0), dtype=np.float32)

            while 1:
                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                # Copy previous cascades's probability into current X_cur
                if layer_id == 0:
                    # first layer not have probability distribution
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                    X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)

                    cur_train_sample_weight = np.ones((n_trains,), dtype=np.float32)
                else:
                    if self.stack_all_probas:
                        X_all_proba_train = np.hstack((X_all_proba_train, X_proba_train))
                        X_all_proba_test = np.hstack((X_all_proba_test, X_proba_test))

                        X_cur_train = X_all_proba_train.copy()
                        X_cur_test = X_all_proba_test.copy()
                    else:
                        X_cur_train = X_proba_train.copy()
                        X_cur_test = X_proba_test.copy()

                    cur_train_sample_weight = self.probability_to_sample_weight(X_proba_train.copy(), y_train, X_train)
                # Stack data that current layer needs in to X_cur
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
                for _i, i in enumerate(look_indexs):
                    X_cur_train = np.hstack((X_cur_train, X_train[:, group_starts[i]:group_ends[i]]))
                    X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}, X_cur_test.shape={}".format(
                    layer_id, look_indexs, X_cur_train.shape, X_cur_test.shape))
                # Fit on X_cur, predict to update X_proba
                y_train_proba_li = np.zeros((n_trains, n_classes))
                y_test_proba_li = np.zeros((n_tests, n_classes))
                for ei, est_config in enumerate(self.est_configs):
                    est = self._init_estimators(layer_id, ei)
                    # fit_trainsform
                    test_sets = [("test", X_cur_test, y_test)] if n_tests > 0 else None
                    y_probas = est.fit_transform(X_cur_train, y_train, y_train,
                            test_sets=test_sets, eval_metrics=self.eval_metrics,
                            keep_model_in_mem=train_config.keep_model_in_mem,
                            sample_weight=cur_train_sample_weight)

                    # Hack: if there is not enough output classes extend output vector with zeros
                    if y_probas[0].shape[1] < n_classes:
                        y_probas[0] = np.hstack((y_probas[0], np.zeros((y_probas[0].shape[0], n_classes - y_probas[0].shape[1]))))

                    # train
                    X_proba_train[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[0]
                    # LOGGER.info("Conf screening: y_train_proba_li shape = {}".format(y_train_proba_li.shape))
                    y_train_proba_li += y_probas[0]
                    # test
                    if n_tests > 0:
                        X_proba_test[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[1]
                        y_test_proba_li += y_probas[1]
                    if train_config.keep_model_in_mem:
                        self._set_estimator(layer_id, ei, est)
                y_train_proba_li /= len(self.est_configs) # output y probability of the layer

                # if probability is very high, exclude these samples from training set
                if self.screening_probability != None:
                    necessity_matrix = np.max(y_train_proba_li, axis=1) < self.screening_probability # / y_train_proba_li.shape[1]
                    # X_cur_train = X_cur_train[necessity_matrix, :]
                    LOGGER.info("Conf screening: max = {}".format(np.max(np.max(y_train_proba_li, axis=1))))
                    LOGGER.info("Conf screening: right = {}".format(self.screening_probability))
                    LOGGER.info("Conf screening: matrix = {}".format(np.sum(necessity_matrix)))
                    LOGGER.info("Conf screening: before shape = {}".format(X_proba_train.shape))
                    X_proba_train = X_proba_train[necessity_matrix, :]
                    X_train = X_train[necessity_matrix, :]
                    X_cur_train = X_cur_train[necessity_matrix, :]
                    y_train = y_train[necessity_matrix]
                    y_train_proba_li = y_train_proba_li[necessity_matrix] # for accuracy computation
                    n_trains = y_train.shape[0]
                    LOGGER.info("Conf screening: after shape = {}".format(X_proba_train.shape))

                    if n_trains == 0:
                        opt_datas = [X_proba_train, y_train, X_proba_test if n_tests > 0 else None, y_test]
                        return layer_id, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]

                train_avg_acc = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1), 'layer_{} - train.classifier_average'.format(layer_id))
                train_acc_list.append(train_avg_acc)
                if n_tests > 0:
                    y_test_proba_li /= len(self.est_configs)
                    test_avg_acc = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1), 'layer_{} - test.classifier_average'.format(layer_id))
                    test_acc_list.append(test_avg_acc)
                else:
                    test_acc_list.append(0.0)

                opt_layer_id = get_opt_layer_id(test_acc_list if stop_by_test else train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [X_proba_train, y_train, X_proba_test if n_tests > 0 else None, y_test]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                    # log and save final result (opt layer)
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                        opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
                    if data_save_dir is not None:
                        self.save_data( data_save_dir, opt_layer_id, *opt_datas)
                    # remove unused model
                    if train_config.keep_model_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            for ei, est_config in enumerate(self.est_configs):
                                self._set_estimator(li, ei, None)
                    self.opt_layer_num = opt_layer_id + 1
                    return opt_layer_id, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(data_save_dir, layer_id, *opt_datas)
                # inc layer_id
                layer_id += 1
            LOGGER.info("[Result][Reach Max Layer] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
            if data_save_dir is not None:
                self.save_data(data_save_dir, self.max_layers - 1, *opt_datas)
            self.opt_layer_num = self.max_layers
            return self.max_layers, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
        except KeyboardInterrupt:
            pass

    def transform(self, X_groups_test):
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_test.shape={}".format([xt.shape for xt in X_groups_test]))
        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_test, False)
        # check group_dims
        group_starts, group_ends, group_dims, X_test = self._check_group_dims(X_groups_test, False)
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("X_test.shape={}".format(X_test.shape))

        n_tests = X_groups_test[0].shape[0]
        n_classes = self.n_classes

        # probability of each cascades's estimators
        X_proba_test = np.zeros((X_test.shape[0], n_classes * self.n_estimators_1), dtype=np.float32)
        X_cur_test = None

        result = np.zeros((X_test.shape[0], n_classes * self.n_estimators_1), dtype=np.float32)
        necessity_matrix = np.ones((X_test.shape[0],), dtype=bool)
        small_necessity_matrix = necessity_matrix.copy()

        X_all_proba_test = np.zeros((n_tests, 0), dtype=np.float32)

        for layer_id in range(self.opt_layer_num):
            # Copy previous cascades's probability into current X_cur
            if layer_id == 0:
                # first layer not have probability distribution
                X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
            else:
                if self.stack_all_probas:
                    X_all_proba_test = np.hstack((X_all_proba_test, X_proba_test))
                    X_cur_test = X_all_proba_test.copy()
                else:
                    X_cur_test = X_proba_test.copy()
            # Stack data that current layer needs in to X_cur
            look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
            for _i, i in enumerate(look_indexs):
                X_cur_test = np.hstack((X_cur_test[small_necessity_matrix], X_test[necessity_matrix, group_starts[i]:group_ends[i]]))
                #X_test[:, group_starts[i]:group_ends[i]]))
            LOGGER.info("[layer={}] look_indexs={}, X_cur_test.shape={}".format(
                layer_id, look_indexs, X_cur_test.shape))
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                if est is None:
                    raise ValueError("model (li={}, ei={}) not present, maybe you should set keep_model_in_mem to True".format(
                        layer_id, ei))
                y_probas = est.predict_proba(X_cur_test)
                # X_proba_test[:, ei * n_classes:ei * n_classes + n_classes] = y_probas
                X_proba_test[necessity_matrix, ei * n_classes:ei * n_classes + n_classes] = y_probas

            result[necessity_matrix] = X_proba_test[necessity_matrix].copy()
            if self.screening_probability != None:
                LOGGER.info("Conf screening test: X shape = {}".format(X_proba_test.shape))
                result[necessity_matrix] = X_proba_test[necessity_matrix].copy()

                tmp_y_proba = X_proba_test.reshape((X_proba_test.shape[0], self.n_estimators_1, self.n_classes))
                tmp_y_proba = tmp_y_proba.mean(axis=1)
                necessity_matrix[necessity_matrix] = np.max(tmp_y_proba[necessity_matrix], axis=1) < self.screening_probability
                small_necessity_matrix = np.max(tmp_y_proba, axis=1) < self.screening_probability

        return result # X_proba_test

    def predict_proba(self, X):
        # n x (n_est*n_classes)
        y_proba = self.transform(X)
        # n x n_est x n_classes
        y_proba = y_proba.reshape((y_proba.shape[0], self.n_estimators_1, self.n_classes))
        y_proba = y_proba.mean(axis=1)
        return y_proba

    def save_data(self, data_save_dir, layer_id, X_train, y_train, X_test, y_test):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train} if pi == 0 else {"X": X_test, "y": y_test}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
