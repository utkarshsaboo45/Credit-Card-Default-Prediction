import numpy as np
import pandas as pd

from docopt import docopt
import os
import pickle

from sklearn.model_selection import cross_validate

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier

opt = docopt(__doc__)

def define_feature_types(X_train):
    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE"
    ]

    pass_through_features = [
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6"
    ]

    numerical_features = list(set(X_train.columns) -
                            set(categorical_features) -
                            set(pass_through_features))

    assert len(numerical_features) + len(categorical_features) + len(pass_through_features) == len(X_train.columns)

    return categorical_features, pass_through_features, numerical_features


def define_column_transformer(X_train,categorical_features, pass_through_features, numerical_features, preprocessor_out_path):
    scalar = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = make_column_transformer(
        (scalar, numerical_features),
        (ohe, categorical_features),
        ("passthrough", pass_through_features)
    )

    preprocessor.fit(X_train);

    print("Dumping Preprocessor!")

    default_preprocessor_out_path = preprocessor_out_path + "preprocessor.pkl"
    try:
        pickle.dump(preprocessor, open(default_preprocessor_out_path, "wb"))
    except:
        os.makedirs(os.path.dirname(default_preprocessor_out_path))
        pickle.dump(preprocessor, open(default_preprocessor_out_path, "wb"))
    
    print("Preprocessor dumped!\n\n")

    return preprocessor

def get_column_names(preprocessor, pass_through_features, numerical_features):
    # Get transformed column names
    return numerical_features + preprocessor.named_transformers_[
        "onehotencoder"
    ].get_feature_names_out().tolist() + pass_through_features

# Credits to Varada K.
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def get_scoring_metrics():
    return [
        "roc_auc",
        "f1",
        "recall",
        "precision"
    ]


def train(results, model_name, model_obj, X_train, y_train):

    results[model_name] = mean_std_cross_val_scores(
        model_obj,
        X_train,
        y_train,
        cv=10,
        n_jobs=-1,
        return_train_score=True,
        scoring=get_scoring_metrics()
    )

    return results

def save_csv(results, results_out_path, filename="/results.csv"):

    if not os.path.exists(results_out_path):
        os.makedirs(results_out_path)

    pd.DataFrame(results).to_csv(results_out_path + filename, index = False, encoding="utf-8")


def main(train_in_path, preprocessor_out_path, results_out_path):
    train_df = pd.read_csv(train_in_path)

    X_train, y_train = train_df.drop(columns=["is_default"]), train_df["is_default"]

    categorical_features, pass_through_features, numerical_features = define_feature_types(X_train)

    preprocessor = define_column_transformer(
        X_train,
        categorical_features,
        pass_through_features,
        numerical_features,
        preprocessor_out_path
    )

    results_base = {}

    models = {
        "Dummy Classifier": make_pipeline(preprocessor, DummyClassifier()),
        "Decision Tree": make_pipeline(preprocessor, DecisionTreeClassifier()),
        "SVC": make_pipeline(preprocessor, SVC()),
        "Logistic Regression": make_pipeline(preprocessor, LogisticRegression(max_iter=10000)),
        "Random Forest": make_pipeline(preprocessor, RandomForestClassifier()),
        "XGBoost": make_pipeline(preprocessor, XGBClassifier(verbosity=0)),
        "LGBM": make_pipeline(preprocessor, LGBMClassifier()),
        "CatBoost": make_pipeline(preprocessor, CatBoostClassifier(verbose=0)),
        "Naive Bayes": make_pipeline(preprocessor, GaussianNB())
    }

    for model_name, model in models.items():
        print("Training", model_name)
        results_base = train(results_base, model_name, model, X_train, y_train)
        print(model_name, "done!\n")

    print("Saving results!\n\n")
    save_csv(results_base, results_out_path, filename="/base_results.csv")
    print("Results saved!")

if __name__ == "__main__":
    main(opt["train_in_path"], opt["preprocessor_out_path"], opt["results_out_path"])