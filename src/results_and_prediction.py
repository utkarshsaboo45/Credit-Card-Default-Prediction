"""
Plot PR curve and check results on test set

Usage: src/results_and_prediction.py --train_in_path=<train_in_path> --test_in_path=<test_in_path> --preprocessor_in_path=<preprocessor_in_path> --img_out_path=<img_out_path>

Options:
--train_in_path=<train_in_path>                     Path to the training data
--test_in_path=<test_in_path>                       Path to the test data
--preprocessor_in_path=<preprocessor_in_path>       Path to load the preprocessor object
--img_out_path=<img_out_path>                       Path to save images
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from docopt import docopt
import pickle

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from averaging_model import (
    define_models
)


# Credits to Varada K.
def plot_roc_curve(model, X, y, img_out_path, filename="/roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))

    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best");

    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)

    plt.savefig(img_out_path + filename)


# Credits to Varada K.
def plot_PR_curve(
    precision,
    recall,
    close_default,
    img_out_path,
    label="PR curve",
    marker_colour="r",
    marker_label="Default threshold",
    filename="/pr_curve.png"
):
    plt.plot(precision, recall, label=label)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot(
        precision[close_default],
        recall[close_default],
        "o",
        markersize=12,
        label=marker_label,
        c=marker_colour,
    )
    plt.legend(loc="best");

    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)

    plt.savefig(img_out_path + filename)


def get_scores(model, X, y, threshold):
    y_pred = model.predict_proba(X)[:, 1] > threshold
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return {
        "Precision": precision,
        "Recall": recall,
        "f1": f1
    }


def main(train_in_path, test_in_path, preprocessor_in_path, img_out_path):

    train_df = pd.read_csv(train_in_path)
    test_df = pd.read_csv(test_in_path)

    X_train, y_train = train_df.drop(columns=["is_default"]), train_df["is_default"]
    X_test, y_test = test_df.drop(columns=["is_default"]), test_df["is_default"]

    preprocessor = pickle.load(open(preprocessor_in_path, "rb"))

    models = define_models(preprocessor)

    pipe_voting = VotingClassifier(
        list(models.items()), voting="soft"
    )

    pipe_voting.fit(X_train, y_train)

    plot_roc_curve(pipe_voting, X_train, y_train, img_out_path)

    precision_avg, recall_avg, thresholds_avg = precision_recall_curve(
        y_train, pipe_voting.predict_proba(X_train)[:, 1]
    )

    close_default_avg = np.argmin(np.abs(thresholds_avg - 0.5))

    plot_PR_curve(precision_avg, recall_avg, close_default_avg, img_out_path)

    print("Final Scores on test set:")
    print(get_scores(pipe_voting, X_test, y_test, 0.63))
    

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--train_in_path"], opt["--test_in_path"], opt["--preprocessor_in_path"], opt["--img_out_path"])