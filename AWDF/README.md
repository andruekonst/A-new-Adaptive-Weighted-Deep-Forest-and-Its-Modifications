# A new Adaptive Weighted Deep Forest Implementation

Code is based on the official [gcForest implementation](https://github.com/kingfengji/gcForest), adapted for Python 3.5+

This package is provided and free for academic usage. You can use it at your own risk. For other purposes, please contact the author of the repository and author of original official implementation (follow the link above for details).

## Usage

Usage is at most the same as of gcForest, instead of the following cascade parameters:
* `stack_all_probas` -- use cumulative probabilities stacking, or stack only last layer output with X to use as the next layer features (bool);
* `probability_based_weighting` -- use probability-based weighting or not (bool);
* `weighting_function` -- name of the weighting function, one of: `"linear", "l2", "1-w^1/2", "1-w2", "contrastive"`;
* `screening_probability` -- screening probability threshold (<= 1.0).

To use the first training strategy, specify `BaggingRandomForestClassifier` or `BaggingExtraTreesClassifier` in `estimators` configuration.
The second training strategy is used by default with standard Scikit-learn estimators.

## Example of usage

```
from gcforest.gcforest import GCForest

config = {
	"random_state": 0,
	"max_layers": 100,
	"early_stopping_rounds", 3,
	"n_classes": len(np.unique(y_train)),
	"estimators": [
		{"n_folds": n_folds, "type": "RandomForestClassifier", "n_estimators": n_estimators, "max_depth": None, "n_jobs": -1},
		{"n_folds": n_folds, "type": "ExtraTreesClassifier", "n_estimators": n_estimators, "max_depth": None, "n_jobs": -1}
	],
	"stack_all_probas": true,
	"probability_based_weighting": true,
	"weighting_function": "l2",
	"screening_probability": 0.95,
}

gc = GCForest(config)
X_train_enc = gc.fit_transform(X_train, y_train)
y_pred = gc.predict(X_test)
```
