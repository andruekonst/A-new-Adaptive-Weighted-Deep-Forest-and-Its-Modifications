import numpy as np
import random
import uuid

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from deep_forest import MGCForest
from deep_forest import CascadeForest

from skimage.transform import resize

from mlflow import log_metric, log_param, log_artifact, set_tag
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def load_usps():
    import h5py
    with h5py.File('datasets/usps.h5', 'r') as hf:
        train = hf.get('train')
        X_train = train.get('data')[:]
        y_train = train.get('target')[:]
        test = hf.get('test')
        X_test = test.get('data')[:]
        y_test = test.get('target')[:]
    return X_train, X_test, y_train, y_test

def test_transfer_learning():
    print('Load MNIST data')
    # mnist = fetch_mldata('MNIST original', data_home='./datasets/scikit-learn-datasets')
    # print(mnist.data.shape)
    # print('Data: {}, target: {}'.format(mnist.data.shape, mnist.target.shape))

    # Resample MNIST to USPS resolution (28x28 -> 16x16)
    # print('Resampling images...')
    # mnist_data = np.swapaxes(mnist.data, 0, 1)
    # print(mnist_data.shape)
    # mnist_data = mnist_data.reshape((28, 28, mnist_data.shape[-1]))
    # print(mnist_data.shape)

    # return None
    # resampled_mnist = resize(mnist_data, output_shape=(16, 16), anti_aliasing=True)
    # mnist_target = mnist.target
    # print('Done')

    import h5py
    # with h5py.File('datasets/resampled_mnist.h5', 'w') as hf:
    #     hf.create_dataset('data', data=resampled_mnist)
    #     hf.create_dataset('target', data=mnist.target)
    # print('Dataset has been saved')

    # return None

    print('Load resampled images')
    with h5py.File('datasets/resampled_mnist.h5', 'r') as hf:
        print(list(hf.keys()))
        resampled_mnist = hf.get('data')[:, :, :]
        mnist_target = hf.get('target')[:]

    resampled_mnist = resampled_mnist.reshape((16 * 16, resampled_mnist.shape[-1]))
    resampled_mnist = np.swapaxes(resampled_mnist, 0, 1)
    # print(resampled_mnist)
    print('X shape: ({}), target shape: ({})'.format(resampled_mnist.shape, mnist_target.shape))
    

    X_train, X_test, y_train, y_test = train_test_split(
        resampled_mnist, # mnist.data,
        mnist_target, # mnist.target,
        test_size=0.2,
        random_state=42,
    )

    #
    # Limit the size of the dataset
    #
    train_size = 2000
    X_train = X_train[:train_size] # X_train[:2000]
    y_train = y_train[:train_size] # y_train[:2000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    print('X_train:', X_train.shape, X_train.dtype)
    print('y_train:', y_train.shape, y_train.dtype)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    transfer_test = 'mnist->usps' # 'usps->mnist'

    if transfer_test == 'mnist->mnist':
        X_transfer = X_test.copy()
        X_transfer_test = X_test.copy()
        y_transfer = y_test
        y_transfer_test = y_test
    elif transfer_test == 'mnist->usps':
        X_transfer, X_transfer_test, y_transfer, y_transfer_test = load_usps()
    elif transfer_test == 'usps->mnist':
        X_transfer = X_train[:2000].copy() # X_train[:2000]
        y_transfer = y_train[:2000].copy() # y_train[:2000]
        X_transfer_test = X_test[:2000].copy()
        y_transfer_test = y_test[:2000].copy()

        X_train, X_test, y_train, y_test = load_usps()
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]



    mgc_forest = MGCForest(
        estimators_config={
            'mgs': [{
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 21,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
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
            }, {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 1,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }]
        },
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
    )

    class_weights = {i: 1 for i in range(10)}

    classifiers_bootstrap = True

    estimators_config = [{
        'estimator_class': ExtraTreesClassifier,
        'estimator_params': {
            'n_estimators': 1000, # 1000,
            'min_samples_split': 11,
            'max_features': 1,
            'n_jobs': -1,
            'class_weight': class_weights,
            'bootstrap': classifiers_bootstrap,
        }
    }, {
        'estimator_class': ExtraTreesClassifier,
        'estimator_params': {
            'n_estimators': 1000, # 1000,
            'min_samples_split': 11,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'class_weight': class_weights,
            'bootstrap': classifiers_bootstrap,
        }
    }, {
        'estimator_class': RandomForestClassifier,
        'estimator_params': {
            'n_estimators': 1000, # 1000,
            'min_samples_split': 11,
            'max_features': 1,
            'n_jobs': -1,
            'class_weight': class_weights,
            'bootstrap': classifiers_bootstrap,
        }
    }, {
        'estimator_class': RandomForestClassifier,
        'estimator_params': {
            'n_estimators': 1000, # 1000,
            'min_samples_split': 11,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'class_weight': class_weights,
            'bootstrap': classifiers_bootstrap,
        }
    }]

    forest_structure = 'all'
    if forest_structure == 'one':
        estimators_config = [estimators_config[-1]] # use only one random forest

    set_tag('forest_structure', forest_structure)
    set_tag('classifiers_bootstrap', classifiers_bootstrap)

    extra = False
    n_components = 128

    if extra == 'pca':
        print("Use PCA")
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test  = pca.transform(X_test)
        X_transfer      = pca.transform(X_transfer)
        X_transfer_test = pca.transform(X_transfer_test)
        log_param('n_components', n_components)
    elif extra == 'lda':
        print("Use LDA")
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X_train, y_train)
        X_train = lda.transform(X_train)
        X_test  = lda.transform(X_test)
        X_transfer      = lda.transform(X_transfer)
        X_transfer_test = lda.transform(X_transfer_test)
        log_param('n_components', n_components)

    set_tag('extra', extra)


    method = 'bootstrap_replacement' # 'no_weights' # 'sample_weight' # 'bootstrap_replacement_inv_weights'
    min_layers = 40
    reg_lambda = 0 # 0.001
    replace_features = False
    weights_for = 'all' # 'proba' # 'X' # 'all'
    weights_method = 'l2' # 'proj_l2_nonorm' # 'proj_l2' # 'l2'
    bootstrap_size = train_size * 2 # train_size / 10
    separate_weights = False # True # 'pred'

    log_param('train_size', train_size)
    log_param('weights_method', weights_method)
    log_param('bootstrap_size', bootstrap_size)
    log_param('separate_weights', separate_weights)


    mgc_forest = CascadeForest(estimators_config, min_layers=min_layers, reg_lambda=reg_lambda, method=method,
                               replace_features=replace_features, weights_for=weights_for, weights_method=weights_method,
                               separate_weights=separate_weights)

    mgc_forest.fit(X_train, y_train, X_transfer, y_transfer)




    y_pred = mgc_forest.predict(X_test)

    print('Predict on MNIST')
    print('Prediction shape:', y_pred.shape)
    acc_source = accuracy_score(y_test, y_pred)
    print(
        'Accuracy:', acc_source,
        'F1 score:', f1_score(y_test, y_pred, average='weighted')
    )

    y_transfer_pred = mgc_forest.predict(X_transfer)

    print('Predict on USPS')
    print('Prediction shape:', y_transfer_pred.shape)
    acc_target = accuracy_score(y_transfer, y_transfer_pred)
    print(
        'Accuracy:', acc_target,
        'F1 score:', f1_score(y_transfer, y_transfer_pred, average='weighted')
    )

    y_transfer_test_pred = mgc_forest.predict(X_transfer_test)

    print('Predict on unseen USPS')
    print('Prediction shape:', y_transfer_test_pred.shape)
    acc_target_unseen = accuracy_score(y_transfer_test, y_transfer_test_pred)
    print(
        'Accuracy:', acc_target_unseen,
        'F1 score:', f1_score(y_transfer_test, y_transfer_test_pred, average='weighted')
    )

    log_param('transfer', transfer_test) # 'mnist->usps')
    log_param('method', method)
    log_param('min_layers', min_layers)
    log_param('reg_lambda', reg_lambda)
    log_param('replace_features', replace_features)
    log_param('weights_for', weights_for)

    log_metric('acc_source', acc_source)
    log_metric('acc_target', acc_target)
    log_metric('acc_target_unseen', acc_target_unseen)


def test_transfer_learning_usps_mnist():
    print('Load MNIST data')

    import h5py

    print('Load resampled images')
    with h5py.File('datasets/resampled_mnist.h5', 'r') as hf:
        print(list(hf.keys()))
        resampled_mnist = hf.get('data')[:, :, :]
        mnist_target = hf.get('target')[:]

    resampled_mnist = resampled_mnist.reshape((16 * 16, resampled_mnist.shape[-1]))
    resampled_mnist = np.swapaxes(resampled_mnist, 0, 1)
    # print(resampled_mnist)
    print('X shape: ({}), target shape: ({})'.format(resampled_mnist.shape, mnist_target.shape))
    

    X_transfer, X_transfer_test, y_transfer, y_transfer_test = train_test_split(
        resampled_mnist, # mnist.data,
        mnist_target, # mnist.target,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = load_usps()

    # X_transfer = X_transfer.reshape((len(X_transfer), 16, 16))
    # X_transfer_test = X_transfer_test.reshape((len(X_transfer_test), 16, 16))

    #
    # Limit the size of the dataset
    #
    X_transfer = X_transfer[:2000]
    y_transfer = y_transfer[:2000]
    X_transfer_test = X_transfer_test[:2000]
    y_transfer_test = y_transfer_test[:2000]

    print('X_train:', X_train.shape, X_train.dtype)
    print('y_train:', y_train.shape, y_train.dtype)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)


    mgc_forest = MGCForest(
        estimators_config={
            'mgs': [{
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 21,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
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
            }, {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 1,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 1000,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }]
        },
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
    )

    estimators_config = [{
        'estimator_class': ExtraTreesClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 1,
            'n_jobs': -1,
        }
    }, {
        'estimator_class': ExtraTreesClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 'sqrt',
            'n_jobs': -1,
        }
    }, {
        'estimator_class': RandomForestClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 1,
            'n_jobs': -1,
        }
    }, {
        'estimator_class': RandomForestClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 'sqrt',
            'n_jobs': -1,
        }
    }]


    mgc_forest = CascadeForest(estimators_config, min_layers=6) #, max_layers=2)

    mgc_forest.fit(X_train, y_train, X_transfer)

    y_pred = mgc_forest.predict(X_test)

    print('Predict on USPS')
    print('Prediction shape:', y_pred.shape)
    print(
        'Accuracy:', accuracy_score(y_test, y_pred),
        'F1 score:', f1_score(y_test, y_pred, average='weighted')
    )

    y_transfer_pred = mgc_forest.predict(X_transfer)

    print('Predict on MNIST')
    print('Prediction shape:', y_transfer_pred.shape)
    print(
        'Accuracy:', accuracy_score(y_transfer, y_transfer_pred),
        'F1 score:', f1_score(y_transfer, y_transfer_pred, average='weighted')
    )

    y_transfer_test_pred = mgc_forest.predict(X_transfer_test)

    print('Predict on unseen MNIST')
    print('Prediction shape:', y_transfer_test_pred.shape)
    print(
        'Accuracy:', accuracy_score(y_transfer_test, y_transfer_test_pred),
        'F1 score:', f1_score(y_transfer_test, y_transfer_test_pred, average='weighted')
    )

    log_param('transfer', 'usps->mnist')



def test_simple_transfer_learning():
    estimators_config = [{
        'estimator_class': ExtraTreesClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 1,
            'n_jobs': -1,
        }
    }, {
        'estimator_class': ExtraTreesClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 'sqrt',
            'n_jobs': -1,
        }
    }, {
        'estimator_class': RandomForestClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 1,
            'n_jobs': -1,
        }
    }, {
        'estimator_class': RandomForestClassifier,
        'estimator_params': {
            'n_estimators': 1000,
            'min_samples_split': 11,
            'max_features': 'sqrt',
            'n_jobs': -1,
        }
    }]

    from sklearn.datasets import load_iris

    # load_wine

    iris = load_iris()
    X = iris.data[:, :]
    y = iris.target
    X_train, X_transfer, y_train, y_transfer = train_test_split(X, y, test_size=0.85, random_state=42)

    forest = CascadeForest(estimators_config, folds=3, verbose=True)
    forest.fit(X_train, y_train, transfer_X=X_transfer)
    y_transfer_pred = forest.predict(X_transfer)
    print('Prediction shape:', y_transfer_pred.shape)
    print(
        'Accuracy:', accuracy_score(y_transfer, y_transfer_pred),
        'F1 score:', f1_score(y_transfer, y_transfer_pred, average='weighted')
    )


def check_datasets():
    import h5py
    print('Load resampled images')
    with h5py.File('datasets/resampled_mnist.h5', 'r') as hf:
        print(list(hf.keys()))
        resampled_mnist = hf.get('data')[:, :, :]
        mnist_target = hf.get('target')[:]

    resampled_mnist = resampled_mnist.reshape((16 * 16, resampled_mnist.shape[-1]))
    resampled_mnist = np.swapaxes(resampled_mnist, 0, 1)
    # print(resampled_mnist)
    print('X shape: ({})'.format(resampled_mnist.shape))

    X_transfer, X_transfer_test, y_transfer, y_transfer_test = load_usps()

    import matplotlib.pyplot as plt
    plt.imshow(resampled_mnist[0].reshape((16, 16)))
    #plt.show()
    plt.imshow(X_transfer[0].reshape((16, 16)))
    #plt.show()
    def print_params(im):
        print("Min: {}, max: {}, mean: {}".format(np.min(im), np.max(im), np.mean(im)))
    print_params(resampled_mnist[0])
    print_params(X_transfer[0])
    print_params(mnist_target)
    print_params(y_transfer)



if __name__ == "__main__":
    test_simple_transfer_learning()
    # test_transfer_learning()
    # check_datasets()
    # test_transfer_learning_usps_mnist()