#
# Inspired by https://arxiv.org/abs/1702.08835
# This code is the modification of https://github.com/STO-OTZ/my_gcForest/ gcForest implementation
#
import itertools
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from utils import create_logger

import cvxpy as cp
from mlflow import log_metric
from minimization_utils import project_onto_simplex



class MGCForest():
    """
    Multi-Grained Cascade Forest

    @param estimators_config    A dictionary containing the configurations for the estimators of
                                the estimators of the MultiGrainedScanners and the CascadeForest.
    @param stride_ratios        A list of stride ratios for each MultiGrainedScanner instance.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.

    Example:

    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }]
    },
    """
    def __init__(
        self,
        estimators_config,
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
        folds=3,
        verbose=False
    ):
        self.mgs_instances = [
            MultiGrainedScanner(
                estimators_config['mgs'],
                stride_ratio=stride_ratio,
                folds=folds,
                verbose=verbose,
            )
            for stride_ratio in stride_ratios
        ]
        self.stride_ratios = stride_ratios

        self.c_forest = CascadeForest(estimators_config['cascade'], verbose=verbose)

    def fit(self, X, y, transfer_X=None):
        if transfer_X is None:
            return self._fit_no_transfer(X, y)
        else:
            return self._fit_transfer(X, y, transfer_X)

    def _fit_transfer(self, X, y, transfer_X):
        scanned_X = np.hstack([
            mgs.scan(X, y)
            for mgs in self.mgs_instances
        ])

        scanned_transfer_X = np.hstack([
            mgs.scan(transfer_X)
            for mgs in self.mgs_instances
        ])

        self.c_forest.fit(scanned_X, y, scanned_transfer_X)

    def _fit_no_transfer(self, X, y):
        scanned_X = np.hstack([
            mgs.scan(X, y)
            for mgs in self.mgs_instances
        ])

        self.c_forest.fit(scanned_X, y)

    def predict(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.predict(scanned_X)

    def __repr__(self):
        return '<MGCForest {}>'.format(self.stride_ratios)


class MultiGrainedScanner():
    """
    Multi-Grained Scanner

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the MultiGrainedScanner.
    @param stride_ratio         The stride ratio to use for slicing the input.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """
    def __init__(
        self, estimators_config, stride_ratio=0.25, folds=3, verbose=False
    ):
        self.estimators_config = estimators_config
        self.stride_ratio = stride_ratio
        self.folds = folds

        self.estimators = [
            estimator_config['estimator_class'](**estimator_config['estimator_params'])
            for estimator_config in self.estimators_config
        ]

        self.logger = create_logger(self, verbose)

    def slices(self, X, y=None):
        """
        Given an input X with dimention N, this generates ndarrays with all the instances
        values for each window. The window shape depends on the stride_ratio attribute of
        the instance.

        For example, if the input has shape (10, 400), and the stride_ratio is 0.25, then this
        will generate 301 windows with shape (10, 100)
        """
        self.logger.debug('Slicing X with shape {}'.format(X.shape))

        n_samples = X.shape[0]
        sample_shape = X[0].shape
        window_shape = [
            max(1, int(s * self.stride_ratio)) if i < 2 else s
            for i, s in enumerate(sample_shape)
        ]

        #
        # Generates all the windows slices for X.
        # For each axis generates an array showing how the window moves on that axis.
        #
        slices = [
            [slice(i, i + window_axis) for i in range(sample_axis - window_axis + 1)]
            for sample_axis, window_axis in zip(sample_shape, window_shape)
        ]
        total_windows = np.prod([len(s) for s in slices])

        self.logger.info('Window shape: {} Total windows: {}'.format(window_shape, total_windows))

        #
        # For each window slices, return the same slice for all the samples in X.
        # For example, if for the first window we have the slices [slice(0, 10), slice(0, 10)],
        # this generates the following slice on X:
        #   X[:, 0:10, 0:10] == X[(slice(None, slice(0, 10), slice(0, 10))]
        #
        # Since this generates on each iteration a window for all the samples, we insert the new
        # windows so that for each sample the windows are consecutive. This is done with the
        # ordering_range magic variable.
        #
        windows_slices_list = None
        ordering_range = np.arange(n_samples) + 1

        for i, axis_slices in enumerate(itertools.product(*slices)):
            if windows_slices_list is None:
                windows_slices_list = X[(slice(None),) + axis_slices]
            else:
                windows_slices_list = np.insert(
                    windows_slices_list,
                    ordering_range * i,
                    X[(slice(None),) + axis_slices],
                    axis=0,
                )

        #
        # Converts any sample with dimention higher or equal than 2 to just one dimention
        #
        windows_slices = \
            windows_slices_list.reshape([windows_slices_list.shape[0], np.prod(window_shape)])

        #
        # If the y parameter is not None, returns the y value for each generated window
        #
        if y is not None:
            y = np.repeat(y, total_windows)

        return windows_slices, y

    def scan(self, X, y=None):
        """
        Slice the input and for each window creates the estimators and save the estimators in
        self.window_estimators. Then for each window, fit the estimators with the data of all
        the samples values on that window and perform a cross_val_predict and get the predictions.
        """
        self.logger.info('Scanning and fitting for X ({}) and y ({}) started'.format(
            X.shape, None if y is None else y.shape
        ))
        self.n_classes = np.unique(y).size

        #
        # Create the estimators
        #
        sliced_X, sliced_y = self.slices(X, y)
        self.logger.debug('Slicing turned X ({}) to sliced_X ({})'.format(X.shape, sliced_X.shape))

        predictions = None
        for estimator_index, estimator in enumerate(self.estimators):
            prediction = None

            if y is None:
                self.logger.debug('Prediction with estimator #{}'.format(estimator_index))
                prediction = estimator.predict_proba(sliced_X)
            else:
                self.logger.debug(
                    'Fitting estimator #{} ({})'.format(estimator_index, estimator.__class__)
                )
                estimator.fit(sliced_X, sliced_y)

                #
                # Gets a prediction of sliced_X with shape (len(newX), n_classes).
                # The method `predict_proba` returns a vector of size n_classes.
                #
                if estimator.oob_score:
                    self.logger.debug('Using OOB decision function with estimator #{} ({})'.format(
                        estimator_index, estimator.__class__
                    ))
                    prediction = estimator.oob_decision_function_
                else:
                    self.logger.debug('Cross-validation with estimator #{} ({})'.format(
                        estimator_index, estimator.__class__
                    ))
                    prediction = cross_val_predict(
                        estimator,
                        sliced_X,
                        sliced_y,
                        cv=self.folds,
                        method='predict_proba',
                        n_jobs=-1,
                    )

            prediction = prediction.reshape((X.shape[0], -1))

            if predictions is None:
                predictions = prediction
            else:
                predictions = np.hstack([predictions, prediction])

        self.logger.info('Finished scan X ({}) and got predictions with shape {}'.format(
            X.shape, predictions.shape
        ))
        return predictions

    def __repr__(self):
        return '<MultiGrainedScanner stride_ratio={}>'.format(self.stride_ratio)


class CascadeForest():
    """
    CascadeForest

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the CascadeForest.
    @param folds                The number of k-folds to use.
    @param min_layers           The minimum number of layers.
    @param max_layers           The maximum number of layers.
    @param reg_lambda           Regularization lambda for weights computation.
    @param method               Weights generation method for Transfer learning.
    @param replace_features     Concatenate probabilities with replacement of previous probabilities, or not.
    @param weights_for          Weights for: 'all' - X & probas, 'X' or 'probas'.
    @param weights_method       Weighting method: 'l2' - fair least squares, 'dist' - inverse distance softmax.
    @param bootstrap_size       Number of samples to bootstrap. If false, it is equal to number of rows in X.
    @param separate_weights     Separate weights computation using predicted class labels, or not.
    @param verbose              Adds verbosity.
    """
    def __init__(self, estimators_config, folds=3, min_layers=None, max_layers=None, reg_lambda=0.1,
                 method='no_weights', replace_features=False, weights_for='all',
                 weights_method='l2', bootstrap_size=None,
                 separate_weights=False, verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds

        self.logger = create_logger(self, verbose)
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.reg_lambda = reg_lambda
        self.method = method
        self.replace_features = replace_features
        self.weights_for = weights_for
        self.weights_method = weights_method
        self.bootstrap_size = bootstrap_size
        self.separate_weights = separate_weights

    def fit(self, X, y, transfer_X=None, transfer_y=None):
        if transfer_X is None:
            return self._fit_no_transfer(X, y)
        else:
            return self._fit_transfer(X, y, transfer_X, transfer_y)

    def _stack_features(self, X, predictions):
        if not self.replace_features:
            return np.hstack([X] + predictions)
        elif self.replace_features == 'ignore':
            return X
        else:
            return np.hstack([X[:, :self.initial_features_num]] + predictions)

    def _bootstrap(self, X, y, sample_weights, size=None):
        samples_num = X.shape[0] if size is None else size
        ind = np.random.choice(X.shape[0], samples_num, p=sample_weights)
        bootstrap_X, bootstrap_y = X[ind], y[ind]

        y_unique = np.unique(y)
        bootstrap_y_unique = np.unique(bootstrap_y)
        y_missing = set(y_unique) - set(bootstrap_y_unique)
        for c in y_missing:
            print(f"Class {c} is missing in the dataset")
            ind_of_c = np.where(y == c)[0].flatten()
            ind_size = len(ind_of_c)
            eps = 1e-6
            tmp_weights = (sample_weights[ind_of_c] + eps) / (sum(sample_weights[ind_of_c]) + eps * ind_size)
            ind = np.random.choice(ind_of_c, 1, p=tmp_weights)[0]
            # ind = ind_of_c[0]
            bootstrap_X = np.vstack((bootstrap_X, X[ind]))
            bootstrap_y = np.array(list(bootstrap_y) + [y[ind]])
        
        return bootstrap_X, bootstrap_y

    def _fit_transfer(self, X, y, transfer_X, transfer_y=None):
        self.logger.info('Transfer-Learning Cascade fitting for X ({}) and y ({}), transfer_X ({}) started'.format(
            X.shape, y.shape, transfer_X.shape
        ))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None
        self.initial_features_num = X.shape[1]

        sample_weights = np.ones((X.shape[0],)) / X.shape[0]

        while True:
            self.logger.info('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]

            predictions = []
            transfer_predictions = []
            for estimator in estimators:
                self.logger.debug('Fitting X ({}) and y ({}) with estimator {}'.format(
                    X.shape, y.shape, estimator
                ))
                # estimator.fit(X, y, sample_weight=sample_weights) 

                # Bootstrap X, y instead of using sample weights
                if self.method == 'bootstrap_replacement_inv_weights':
                    tmp_w = (1 - sample_weights)
                    bootstrap_X, bootstrap_y = self._bootstrap(X, y, tmp_w, size=self.bootstrap_size)
                    estimator.fit(bootstrap_X, bootstrap_y)
                elif self.method == 'bootstrap_replacement':
                    bootstrap_X, bootstrap_y = self._bootstrap(X, y, sample_weights, size=self.bootstrap_size)
                    estimator.fit(bootstrap_X, bootstrap_y)
                elif self.method == 'sample_weight':
                    estimator.fit(X, y, sample_weight=sample_weights)
                elif self.method == 'sample_weight_inv':
                    tmp_w = (1 - sample_weights)
                    estimator.fit(X, y, sample_weight=tmp_w)
                else:
                    estimator.fit(X, y)

                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                # prediction = cross_val_predict(
                #     estimator,
                #     X,
                #     y,
                #     cv=self.folds,
                #     method='predict_proba',
                #     n_jobs=-1,
                #     fit_params={
                #         'sample_weight': sample_weights,
                #     },
                # )
                prediction = estimator.predict_proba(X)

                predictions.append(prediction)

                transfer_pred = estimator.predict_proba(transfer_X)
                transfer_predictions.append(transfer_pred) # TODO: check if it is correct

                # print("Shapes are:")
                # print(prediction.shape)
                # print(transfer_pred.shape)

            self.logger.info('Level {}:: got all predictions'.format(self.level + 1))

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = self._stack_features(X, predictions)

            # Stack horizontally the predictions of unlabeled X (transfer_X)
            transfer_X = self._stack_features(transfer_X, transfer_predictions)

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            #
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )

            # Get class labels estimation for weights separation
            transfer_y_prediction = self.classes.take(
                np.array(transfer_predictions).mean(axis=0).argmax(axis=1)
            )

            if not (transfer_y is None):                
                transfer_score = accuracy_score(transfer_y, transfer_y_prediction)
                self.logger.info('Level {}:: got transfer accuracy {}'.format(self.level + 1, transfer_score))
                log_metric('acc_transfer_tmp', transfer_score)


            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            log_metric('acc_tmp', score)
            if not (self.min_layers is None) and self.level < self.min_layers:
                self.max_score = None
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
                # Compute weights:
                # sample_weights = self._compute_weights(X, transfer_X, method='l2')
                if self.weights_for == 'X':
                    weights_X = X[:, :self.initial_features_num]
                    weights_transfer_X = transfer_X[:, :self.initial_features_num]
                elif self.weights_for == 'proba':
                    weights_X = X[:, self.initial_features_num:]
                    weights_transfer_X = transfer_X[:, self.initial_features_num:]
                else:
                    weights_X = X
                    weights_transfer_X = transfer_X

                if self.method != 'no_weights':
                    if self.separate_weights:
                        train_classes = y if self.separate_weights != 'pred' else y_prediction
                        sample_weights = self._compute_separated_weights(weights_X,
                                                                         weights_transfer_X,
                                                                         train_classes,
                                                                         transfer_y_prediction,
                                                                         method=self.weights_method)
                    else:    
                        sample_weights = self._compute_weights(weights_X, weights_transfer_X, method=self.weights_method)
            else:
                break
            if not (self.max_layers is None) and self.level > self.max_layers:
                break


    def _compute_separated_weights(self, X, target_X, y, pred_target_y, method=None):
        weights = np.zeros((X.shape[0],))
        for c in self.classes:
            x_ind = np.where(y == c)[0]
            target_ind = np.where(pred_target_y == c)[0]
            w_c = self._compute_weights(X[x_ind], target_X[target_ind], method=method)
            weights[x_ind] = w_c
        weights /= np.sum(weights)
        return weights

            
    def _compute_weights(self, X, target_X, method='l2'):
        """
        Compute weights minimizing Maximum Mean Discrepancy.

        @param X            Samples matrix.
        @param target_X     Target samples matrix.
        @param method       Method of weights computation.
                            Preferable options:
                                'l2' - convex optimization with unary simplex restrictions;
                                'proj_l2_nonorm' - solving system and projecting onto unary simplex,
                                                   pure solution.
        """
        if method == 'proj_l2' or method == 'proj_l2_nonorm':
            #
            # At first calculate unrestricted weights: (X.T)^-1
            # Then project answer onto Unit simplex
            # 
            target_center = np.mean(target_X, axis=0) # * X.shape[0]
            # Solve the system
            w = np.linalg.lstsq(X.T, target_center.T, rcond='warn')[0]
            weights = project_onto_simplex(w, normalize=True if method == 'proj_l2' else False)
            print(f"Weights sum: {np.sum(weights)}")
            weights /= np.sum(weights)
            return weights

        #
        # Pure solution, which make unrestricted weights
        #
        # Compute target center multiplied by number of source rows
        # target_center = np.mean(target_X, axis=0) * X.shape[0]
        # Solve the system
        # print("X^T shape: ({}), target_center^T shape: ({})".format(X.T.shape, target_center.T.shape))
        # w = np.linalg.lstsq(X.T, target_center.T, rcond='warn')[0]
        # print(w)
        # return w.T
        if method == 'dist' or method == 'dist2':
            print("Using distance weighting")
            target_center = np.mean(target_X, axis=0)
            residuals = X - target_center
            norm = np.linalg.norm(residuals, axis=1)
            print(f"Max norm: {np.max(norm)}")
            if method == 'dist':
                weights = np.max(norm) - norm # inverse weights
            elif method == 'dist2':
                small_eps = 1e-9
                weights = 1.0 / (norm + small_eps)
            weights = np.exp(weights) # softmax
            print(f"Weights sum: {np.sum(weights)}")
            weights /= np.sum(weights)
            return weights

        # Compute target center multiplied by number of source rows
        target_center = np.mean(target_X, axis=0) # * X.shape[0]
        # Solve the system
        q = cp.Constant(value=target_center.flatten())
        x_ = cp.Constant(value=X)

        w = cp.Variable(X.shape[0])
        # lam = self.optimization_lambda # 0.001
        # M = len(J)
        M = np.linalg.norm(X) ** 2 # target_X)
        print("M:", M)
        lam = self.reg_lambda # 0.1
        if lam == 0:
            print("No regularization")
            # cp.norm2(cp.matmul(X, beta) - Y)**2
            objective = cp.Minimize(cp.sum_squares(q.T - w.T @ x_)) # cp.Minimize(cp.sum_squares(q - x_ * w))
        else:
            objective = cp.Minimize(cp.sum_squares(q.T - w.T @ x_) / M + lam * cp.norm2(w)) # + lam * cp.norm2(w))
        constraints = [w >= 0, cp.sum_entries(w) == 1] #, w >= self.simplex_lower_boundary]
        prob = cp.Problem(objective, constraints)

        print("Problem is prepared")

        try:
            result = prob.solve()
        except Exception as ex:
            print("Exception occurred: {}".format(ex))
            print("Using SCS solver")
            result = prob.solve(solver=cp.SCS, verbose=False)
        print("Problem status: {}".format(prob.status))
        try:
            weights = w.value.A.flatten()
        except Exception as ex:
            print("Can't compute weights, use uniform distribution")
            weights = np.ones((X.shape[0],)) / X.shape[0]
        print(weights)
        weights[weights < 0] = 0
        weights_sum = np.sum(weights)
        print("Weights sum: {}".format(weights_sum))
        if weights_sum != 1.0: # probably always true
            weights /= weights_sum
        return weights


    def _fit_no_transfer(self, X, y):
        self.logger.info('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None

        while True:
            self.logger.info('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]

            predictions = []
            for estimator in estimators:
                self.logger.debug('Fitting X ({}) and y ({}) with estimator {}'.format(
                    X.shape, y.shape, estimator
                ))
                estimator.fit(X, y)

                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )

                predictions.append(prediction)

            self.logger.info('Level {}:: got all predictions'.format(self.level + 1))

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = np.hstack([X] + predictions)

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            #
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )

            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
            else:
                break

    def predict(self, X):
        for estimators in self.levels:

            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            self.logger.info('Shape of predictions: {} shape of X: {}'.format(
                np.array(predictions).shape, X.shape
            ))
            # X = np.hstack([X] + predictions)
            X = self._stack_features(X, predictions)

        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

    def __repr__(self):
        return '<CascadeForest forests={}>'.format(len(self.estimators_config))
