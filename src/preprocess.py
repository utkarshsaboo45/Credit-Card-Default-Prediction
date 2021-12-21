"""
Preprocess raw data, split it in training and test sets and save in csv files

Usage: src/preprocess.py --in_path=<in_path> --train_out_path=<train_out_path> --test_out_path=<test_out_path>

Options:
--in_path=<in_path>                 Path to the raw data
--train_out_path=<train_out_path>   Save path to the train data
--test_out_path=<test_out_path>     Save path to the test data
"""

import pandas as pd

import os

from docopt import docopt

from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def read_data(path):
    # Read CSV from the given path
    df = pd.read_csv(path)
    
    return df


def clean_data(df):
    # Dropping unrelated columns
    df.drop(columns=["ID"], inplace=True)

    # Renaming columns
    df.rename(columns={
        "default payment next month": "is_default",
        "PAY_0":"PAY_1"
    }, inplace=True)

    return df


def create_features(df):
    # Percent Bill Paid
    for i in range(1, 6):
        df[f"percent_paid{i}"] = df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i + 1}"] * 100
        df[f"percent_paid{i}"][df[f"percent_paid{i}"] < 0] = 100 - (df[f"percent_paid{i}"])
        df[f"percent_paid{i}"][df[f"BILL_AMT{i + 1}"] == 0] = 100 + df[f"PAY_AMT{i}"] * 0.01

    # Precent Credit Utilized
    for i in range(1, 7):
        df[f"percent_credit_utilised{i}"] = df[f"BILL_AMT{i}"] / df[f"LIMIT_BAL"] * 100

    # Standard Deviations
    bill_amt_col_names = []
    pay_amt_col_names = []

    for i in range(1, 7):
        bill_amt_col_names.append(f"BILL_AMT{i}")
        pay_amt_col_names.append(f"PAY_AMT{i}")

    df["std_dev_bill"] = df[bill_amt_col_names].std(axis=1)
    df["std_dev_pay"] = df[pay_amt_col_names].std(axis=1)

    return df


def preprocess(df):
    # Change Education categories 0, 5 and 6 to 4
    df["EDUCATION"].replace({5: 4, 6: 4, 0: 4}, inplace=True)

    # Change PAY_X values from -2 to 0
    for i in range(1, 7):
        df[f"PAY_{i}"].replace({-2: 0}, inplace=True)

    return df

def split_data(df):
    # Data Splitting
    return train_test_split(
        df,
        test_size=0.2,
        random_state=123
    )

def save_data(df, path):
    # Export data to csv file
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    df.to_csv(path, index=False)


def main(in_path, train_out_path, test_out_path):
    # Reads the raw data from in_path and 
    # writes the processed data to out_path
    df = read_data(in_path)
    df = clean_data(df)
    df = create_features(df)
    df = preprocess(df)

    train_data, test_data = split_data(df)
    save_data(train_data, train_out_path)
    save_data(test_data, test_out_path)


if __name__ == "__main__":
    main(opt["--in_path"], opt["--train_out_path"], opt["--test_out_path"])