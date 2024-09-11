import os
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# define nine outlier detection tools to be compared
from numpy.random import seed
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.utility import precision_n_scores, standardizer
from scipy.io import loadmat

# from pyod.models.feature_bagging import FeatureBagging
from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_consistent_length, column_or_1d

from src.evaluation import evaluate_clf
from src.models import get_normalizing_flow, neg_loglik
from src.plots import clip, plot_points, plot_importances
from src.utils import get_dataset_names, get_models
from sklearn import tree

# Set-up seeds
random_state = np.random.RandomState(40)

# Define variables
SPLIT_RATIO = 0.3
MULTI = 2
SCORING = "f1"
N_FEATURES = 10

# Define paths
DATA_PATH = Path("data")
RESULTS = Path("results")
FIGURES = Path("figures")
LOSS_FIGURES = FIGURES / "loss"
POINTS_FIGURES = FIGURES / "points"
INTERPRETABILITY = Path("interpretability")
IMPORTANCES = INTERPRETABILITY / "importances"
TEXT_REPRESENTATION = INTERPRETABILITY / "text_representation"

# Update directories
IMPORTANCES = IMPORTANCES / f"{SCORING}"

# Create directories
RESULTS.mkdir(parents=True, exist_ok=True)
LOSS_FIGURES.mkdir(parents=True, exist_ok=True)
POINTS_FIGURES.mkdir(parents=True, exist_ok=True)
IMPORTANCES.mkdir(parents=True, exist_ok=True)
TEXT_REPRESENTATION.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Get dataset names
    mat_file_list = get_dataset_names()

    # construct containers for saving results of each dataset
    results_df = pd.DataFrame(
        columns=[
            "dataset",
            "model",
            "roc",
            "prn_n",
            "prc",
            "recall",
            "mcc",
            "f1",
            "time",
        ]
    )

    for mat_file in mat_file_list:
        print(f"\n... Processing {mat_file}, ...\n")

        # load mat file
        math_path = DATA_PATH / mat_file
        mat = loadmat(math_path)

        # get features and labels (outliers)
        X = mat["X"][:, :]
        y = mat["y"].ravel()

        # get outlier ratio
        ndim = X.shape[1]
        outliers_pct = np.count_nonzero(y) / len(y)
        outliers_percentage = round(outliers_pct * 100, ndigits=4)

        print(f"\nDataset Shape: {X.shape}")
        print(f"Outliers Percentage: {outliers_percentage}%\n")

        # 50% data for training and 50% for testing
        print(
            f"\nSplitting data with Training: {100 * (1 - SPLIT_RATIO)}%, Testing: {100 * SPLIT_RATIO}%\n"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=SPLIT_RATIO, random_state=random_state
        )

        # Standardize data
        X_train_original = X_train.copy()
        X_train, X_test = standardizer(X_train, X_test)

        # set up learning rate schedule
        initial_learning_rate = 1e-3
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9
        )

        # get normalizing flow model and transformed distribution
        model, td = get_normalizing_flow(ndim)

        # compile model
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0),
            loss=neg_loglik,
        )

        # fit model
        result = model.fit(x=X_train, y=np.zeros(len(X_train)), epochs=300, verbose=0)

        # plot and save loss
        fig, ax = plt.subplots()
        ax.plot(result.history["loss"])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss for {mat_file}")
        fig.savefig(LOSS_FIGURES / f"{mat_file}_loss.png")

        # create a new dataset MULTI times larger than the original along with its likelihoods
        print(f"\nCreated a new dataset {MULTI} times larger than the original...")
        sample_size = MULTI * len(X_train)
        gen_dataset = td.sample(sample_size, seed=23).numpy()  # it is unlabeled
        likelihoods = td.log_prob(gen_dataset)
        print(f"New dataset shape: {gen_dataset.shape}")

        # get non-outliers of the new dataset
        num_non_outliers = int((1 - outliers_pct) * sample_size)
        ind = np.argpartition(likelihoods, -num_non_outliers)[-num_non_outliers:]
        non_outliers = gen_dataset[ind]

        # get outliers and their likelihoods of the new dataset
        outliers = np.array([o.tolist() for o in gen_dataset if o not in non_outliers])
        outliers_likelihoods = td.log_prob(outliers)

        # concatenate outliers and non-outliers in one array called x_gen and their labels in y_gen
        x_gen = np.concatenate((outliers, non_outliers))
        y_non_outliers = np.zeros(len(non_outliers))
        y_outliers = np.ones(len(outliers))
        y_gen = np.concatenate((y_outliers, y_non_outliers))

        # Cap x_gen based on the limits of X_train
        x_gen = clip(x_gen, X_train)

        # save figures
        plot_points(
            X_train_original,
            X_train,
            X_test,
            x_gen,
            y_gen,
            POINTS_FIGURES / f"{mat_file}.png",
        )

        ##############################
        # Tune Decision Tree with the new dataset (gen_, all_y)
        ##############################

        # define classifier
        start_time = time()
        clf_name = "Decision Tree"
        clf = DecisionTreeClassifier()

        # define the hyperparameter grid to search
        param_grid = {
            "max_depth": [None, 2, 3, 5, 10, 20, 30],
            "min_samples_split": [2, 3, 4, 5, 7, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        }

        # perform a grid search with cross-validation
        print(f"\n... Tuning {clf_name}, ...\n")
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=SCORING)
        grid_search.fit(x_gen, y_gen)

        # get the best hyperparameters
        best_params = grid_search.best_params_

        # create a new classifier with the best hyperparameters
        best_clf = DecisionTreeClassifier(**best_params)

        # fit the classifier on the training data
        best_clf.fit(x_gen, y_gen)

        # Calculate the mean of all_y, preds, and the accuracy score
        preds = best_clf.predict(X_test)
        proba = best_clf.predict_proba(X_test)[:, 1]

        # evaluate and save results
        new_row = evaluate_clf(mat_file, clf_name, y_test, preds, proba, start_time)
        new_row_df = pd.DataFrame(new_row, index=[0])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        # interpret results
        importances = best_clf.feature_importances_
        N = X_train.shape[1]
        feature_names = [f'feature_{i + 1}' for i in range(N)]
        plot_importances(N_FEATURES, importances, feature_names, IMPORTANCES / f"{mat_file}.png")

        # create text representation of the tree and save it
        text_representation = tree.export_text(best_clf, feature_names=feature_names)
        with open(TEXT_REPRESENTATION / f"{mat_file}.txt", "w") as f:
            f.write(text_representation)

        plt.figure(figsize=(20, 20))
        tree.plot_tree(best_clf, feature_names=feature_names, class_names=["non-outlier", "outlier"], filled=True)
        plt.savefig(TEXT_REPRESENTATION / f"{mat_file}.png")

        ##############################
        # Comparison with pyod models on the original dataset (X_train, X_test, y_train, y_test)
        ##############################

        # define pyod models
        model_list = get_models()
        for clf, clf_name in model_list:
            try:
                start_time = time()

                print(f"\n... Fit {clf_name}, ...\n")
                if clf_name == "DeepSVDD":
                    clf = clf(n_features=ndim, contamination=outliers_pct)
                else:
                    clf = clf(contamination=outliers_pct)
                clf.fit(X_train)

                # get the prediction on the test data
                y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
                # TODO: Identify the difference between decision_function and predict_proba and choose one
                y_test_scores = clf.decision_function(X_test)  # outlier scores
                y_proba = clf.predict_proba(X_test)[:, 1]  # outlier probabilities

                # evaluate and save results
                new_row = evaluate_clf(
                    mat_file, clf_name, y_test, y_test_pred, y_proba, start_time
                )
                new_row_df = pd.DataFrame(new_row, index=[0])
                results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            except Exception as e:
                print(f"\n\n\n\n\nError in {clf_name}: {e}\n\n\n\n\n\n")

    # sort by dataset name and roc from highest to lowest
    results_df.sort_values(by=["dataset", "roc"], inplace=True, ascending=[True, False])

    # save results
    results_df.to_csv(RESULTS / f"{SCORING}_results.csv", index=False)
