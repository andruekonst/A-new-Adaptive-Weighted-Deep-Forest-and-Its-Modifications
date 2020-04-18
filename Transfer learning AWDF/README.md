# A new Adaptive Weighted Deep Forest for Transfer Learning Implementation

Code is based on the lightweight [gcForest implementation](https://github.com/STO-OTZ/my_gcForest/).

Make sure, you have [CVXPY](https://www.cvxpy.org/install/index.html) installed.

## Example of usage

For this example, we will make synthetic transfer learning case, where Source and Target domains are sampled from the same common dataset.

*This code could be found in `test_deep_forest.py` file in `test_simple_transfer_learning` method.*
```
# Load IRIS dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :]
y = iris.target

# Split data into fake Source and Target
# here X_train refers to Source domain, and X_transfer to Target
X_train, X_transfer, y_train, y_transfer = train_test_split(X, y, test_size=0.85, random_state=42)

# Specify the Deep Forest structure
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

# Initialize the Deep Forest Cascade
forest = CascadeForest(estimators_config, folds=3, verbose=True)
# Fit method on the Source domain in order to cover both Source and Target
# (!) notice, that only X_transfer data is used, no y_trainsfer is needed for the training
forest.fit(X_train, y_train, transfer_X=X_transfer)

# Make predictions on the Target dataset
y_transfer_pred = forest.predict(X_transfer)

print(
    'Accuracy:', accuracy_score(y_transfer, y_transfer_pred),
    'F1 score:', f1_score(y_transfer, y_transfer_pred, average='weighted')
)
```


## Running testing scripts

[MLflow](https://github.com/mlflow/mlflow/) is required to for the results logging.

For the `MNIST -> USPS` transfer learning testing, two files are needed:
1. `datasets/usps.h5` containing `train` and `test` datasets, which represent images of shape 16x16 and corresponding digits classes;
2. `datasets/mnist.h5` containing default dataset, with resampled (to 16x16) images and corresponding digits classes.

To run tests, use `test_transfer_learning()` and `test_transfer_learning_usps_mnist()` methods from `test_deep_forest.py`.