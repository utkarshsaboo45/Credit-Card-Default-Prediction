import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from docopt import docopt
import pickle

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from averaging_model import (
    define_models,

)

opt = docopt(__doc__)


# Credits to Varada K.
def plot_roc_curve(model, X, y):
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


# Credits to Varada K.
def plot_PR_curve(
    precision,
    recall,
    close_default,
    label="PR curve",
    marker_colour="r",
    marker_label="Default threshold",
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


def main(train_in_path, test_in_path, preprocessor_in_path):

    train_df = pd.read_csv(train_in_path)
    test_df = pd.read_csv(test_in_path)

    X_train, y_train = train_df.drop(columns=["is_default"]), train_df["is_default"]
    X_test, y_test = test_df.drop(columns=["is_default"]), test_df["is_default"]

    preprocessor = pickle.load(open(preprocessor_in_path, "r"))

    models = define_models(preprocessor)

    pipe_voting = VotingClassifier(
        list(models.items()), voting="soft"
    )

    pipe_voting.fit(X_train, y_train)

    plot_roc_curve(pipe_voting, X_train, y_train)

    precision_avg, recall_avg, thresholds_avg = precision_recall_curve(
        y_train, pipe_voting.predict_proba(X_train)[:, 1]
    )

    close_default_avg = np.argmin(np.abs(thresholds_avg - 0.5))

    plot_PR_curve(precision_avg, recall_avg, close_default_avg)

    print("Final Scores on train set:")
    print(get_scores(pipe_voting, X_test, y_test, 0.63))
    

if __name__ == "__main__":
    main(opt["train_in_path"], opt["preprocessor_out_path"], opt["results_out_path"])