import pandas as pd

from docopt import docopt
import pickle

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from catboost import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier

from base_model import (
    train,
    save_csv,
    mean_std_cross_val_scores,
    get_scoring_metrics
)


opt = docopt(__doc__)


def define_models(preprocessor):

    pipe_lr_balanced = make_pipeline(
        preprocessor,
        LogisticRegression(
            class_weight="balanced",
            C=100.0,
            max_iter=10000
        )
    )

    pipe_nb = make_pipeline(preprocessor, GaussianNB())

    pipe_catboost = make_pipeline(
        preprocessor,
        CatBoostClassifier(
            verbose=0,
            random_state=123,
            auto_class_weights="Balanced"
        )
    )

    pipe_lgbm = make_pipeline(
        preprocessor,
        LGBMClassifier(random_state=123, class_weight="balanced")
    )

    return {
        "Logistic Regression": pipe_lr_balanced,
        "Catboost": pipe_catboost,
        "LGBM": pipe_lgbm,
        "Naive Bayes": pipe_nb
    }

def train_averaging_models(models, results, X_train, y_train):
    # Stacking Classifier
    pipe_stacking = StackingClassifier(
        estimators=list(models.items()),
        final_estimator=LogisticRegression()
    )

    results["stacking"] = mean_std_cross_val_scores(
        pipe_stacking,
        X_train,
        y_train,
        cv=10,
        n_jobs=-1,
        return_train_score=True,
        scoring=get_scoring_metrics()
    )

    # Voting Classifier
    pipe_voting = VotingClassifier(
        list(models.items()), voting="soft"
    )

    results["voting"] = mean_std_cross_val_scores(
        pipe_voting,
        X_train,
        y_train,
        cv=10,
        n_jobs=-1,
        return_train_score=True,
        scoring=get_scoring_metrics()
    )

    return results


def main(train_in_path, preprocessor_in_path, results_out_path, model_out_path):

    train_df = pd.read_csv(train_in_path)

    X_train, y_train = train_df.drop(columns=["is_default"]), train_df["is_default"]

    preprocessor = pickle.load(open(preprocessor_in_path, "r"))

    models = define_models(preprocessor)

    results = {}

    for model_name, model in models.items():
        print("Training", model_name)
        results = train(results, model_name, model, X_train, y_train)
        print(model_name, "done!\n")
    
    results = train_averaging_models(models, results, )
    
    print("Saving results!\n\n")
    save_csv(results, results_out_path, filename="/selected_results.csv")
    print("Results saved!")
    

if __name__ == "__main__":
    main(opt["train_in_path"], opt["preprocessor_out_path"], opt["results_out_path"])