"""
Test Logistic Regression with different hyperparameters on our training data

Usage: src/hyperparam_tuning.py --train_in_path=<train_in_path> --preprocessor_in_path=<preprocessor_in_path> --results_out_path=<results_out_path> --model_out_path=<model_out_path>

Options:
--train_in_path=<train_in_path>                     Path to the training data
--preprocessor_in_path=<preprocessor_in_path>       Path to load the preprocessor object
--results_out_path=<results_out_path>               Save path for the scores
--model_out_path=<model_out_path>                   Save path for tuned model file
"""

import numpy as np
import pandas as pd

from docopt import docopt
import os
import pickle

from sklearn.model_selection import (
    RandomizedSearchCV,
)

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

from base_model import (
    define_feature_types,
    get_column_names,
    mean_std_cross_val_scores,
    get_scoring_metrics,
    save_csv
)


def tune_lr(preprocessor, X_train, y_train):#, model_out_path):
    # Logistic Regresion
    print("Tuning Logistic Regression")

    param_lr = {
        "logisticregression__class_weight": ["balanced", None],
        "logisticregression__C": 10.0 ** np.arange(-2, 4)
    }

    pipe_lr = make_pipeline(preprocessor, LogisticRegression(max_iter=10000))

    random_search = RandomizedSearchCV(
        pipe_lr,
        param_lr,
        n_jobs=-1,
        return_train_score=True,
        scoring=get_scoring_metrics(),
        refit="recall"
    )

    random_search.fit(X_train, y_train);

    # print("Dumping Model!")

    # default_model_out_path = model_out_path + "random_search.pkl"

    # try:
    #     pickle.dump(random_search, open(default_model_out_path, 'wb'))
    # except:
    #     os.makedirs(os.path.dirname(default_model_out_path))
    #     pickle.dump(random_search, open(default_model_out_path, 'wb'))
    
    # print("Model dumped!\n\n")

    return random_search

def train_best_lr(preprocessor, random_search, X_train, y_train, model_out_path):

    print("Training Best Logistic Regression")

    pipe_lr_best = make_pipeline(
        preprocessor,
        LogisticRegression(
            class_weight=random_search.best_params_["logisticregression__class_weight"],
            C=random_search.best_params_["logisticregression__C"],
            max_iter=10000
        )
    )
    
    pipe_lr_best.fit(X_train, y_train)

    print("Dumping Model!")

    default_model_out_path = model_out_path + "/pipe_lr_best.pkl"

    try:
        pickle.dump(pipe_lr_best, open(default_model_out_path, 'wb'))
    except:
        os.makedirs(os.path.dirname(default_model_out_path))
        pickle.dump(pipe_lr_best, open(default_model_out_path, 'wb'))
    
    print("Model dumped!\n\n")

    return pipe_lr_best

def save_feature_importances(X_train, pipe_lr_best, preprocessor, results_out_path, out_file_name="/feature_importances.csv"):

    _, pass_through_features, numerical_features = define_feature_types(X_train)    

    importances = pd.DataFrame(
        pipe_lr_best.named_steps["logisticregression"].coef_,
        columns=get_column_names(preprocessor, pass_through_features, numerical_features)
    ).T

    importances["abs_coef"] = np.abs(importances[0])

    importances.sort_values(by="abs_coef", ascending=False, inplace=True)

    save_csv(importances, results_out_path, out_file_name)


def save_results(model, results_out_path, out_file_name="/randomized_search_results.csv"):
    results = pd.DataFrame(model.cv_results_)[[
        "mean_fit_time",
        "param_logisticregression__class_weight",
        "param_logisticregression__C",
        "mean_train_recall",
        "mean_test_recall",
        "mean_train_precision",
        "mean_test_precision",
        "mean_train_f1",
        "mean_test_f1"
    ]].sort_values("mean_test_recall", ascending=False).T

    save_csv(results, results_out_path, out_file_name)

def main(train_in_path, preprocessor_in_path, results_out_path, model_out_path):

    train_df = pd.read_csv(train_in_path)

    X_train, y_train = train_df.drop(columns=["is_default"]), train_df["is_default"]

    print("Loading Pickle File...")
    preprocessor = pickle.load(open(preprocessor_in_path, "rb"))
    print("Loaded Pickle File!")

    random_search = tune_lr(preprocessor, X_train, y_train)#, model_out_path)

    save_results(random_search, results_out_path)

    pipe_lr_best = train_best_lr(preprocessor, random_search, X_train, y_train, model_out_path)

    save_feature_importances(X_train, pipe_lr_best, preprocessor, results_out_path)
    

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--train_in_path"], opt["--preprocessor_in_path"], opt["--results_out_path"], opt["--model_out_path"])