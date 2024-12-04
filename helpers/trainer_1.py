import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from bson import ObjectId
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from db.mongo import histories, samples, users
from helpers.features import compute_features


def data_augumentation(df):
    augmented_data = []

    # Set the δ value
    delta = 0.01

    # Calculate mean and standard deviation for each feature (column)
    means = df.mean()
    std_devs = df.std()

    # Set the augmentation factor (e.g., 2 to double or 3 to triple the data size)
    augmentation_factor = 10  # Adjust as needed

    # Generate augmented data
    augmented_data = []
    for _ in range(augmentation_factor):
        for _, _ in df.iterrows():
            # Generate a synthetic sample within the zone of acceptance for each feature
            synthetic_sample = {}
            for column in df.columns:
                lower_bound = means[column] - std_devs[column] * delta
                upper_bound = means[column] + std_devs[column] * delta
                synthetic_sample[column] = np.random.uniform(lower_bound, upper_bound)
            augmented_data.append(synthetic_sample)

    # Create a DataFrame from the augmented data
    df_augmented = pd.DataFrame(augmented_data)

    # Combine original and augmented data
    df_combined = pd.concat([df, df_augmented], ignore_index=True)
    return df_combined


def process_event_data(input_data):
    rows = []

    for sample in input_data:
        events = sample["events"]
        df = pd.DataFrame(events)

        features = compute_features(df)

        feature_row = {**features}
        rows.append(feature_row)  # Add the row to the list

    result_df = pd.DataFrame(rows)
    return result_df


def train(df):
    # Define features and target
    X = df[
        [
            "error_rate%",
            "capsLock_usage",
            "wpm",
            "negative_ud%",
            "negative_uu%",
            "rsa_ratio",
            "lsa_ratio",
            "mean_hold_time1",
            "mean_hold_time2",
            "mean_f1",
            "mean_f2",
            "mean_f3",
            "mean_f4",
        ]
    ]
    y = df["label"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=142
    )

    model = XGBClassifier(
        colsample_bytree=0.5,
        min_child_weight=5,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        scale_pos_weight=0.7,
    )

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "max_depth": [8, 16, 32, 64],
        "learning_rate": [0.0001, 0.001, 0.01, 0.01],
        "n_estimators": [50, 100, 200],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_resampled, Y_resampled = smote.fit_resample(X_train, y_train)
    print(
        len(X_resampled),
    )
    # Fit GridSearchCV
    grid_search.fit(X_resampled, Y_resampled)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, best_model


data_folder = Path(__file__).resolve().parent.parent / "data"


def generate_model_url(user_id: str):
    return data_folder / f"{user_id}.json"


async def train_model_for_user(user_id: str):
    user = await users.find_one({"_id": ObjectId(user_id)})
    all_users_dataset = pd.read_csv(data_folder / "all_users_dataset.csv")
    cursor = samples.find(
        {"userId": str(user_id), "events": {"$exists": True, "$ne": []}}
    )
    user_samples = await cursor.to_list(length=None)

    legitimate_samples = process_event_data(user_samples)

    imposter_samples = all_users_dataset.copy()
    legitimate_samples["label"] = 1
    imposter_samples["label"] = 0

    final_set = pd.concat([legitimate_samples, imposter_samples], ignore_index=True)

    accuracy, best_model = train(final_set)
    best_model.save_model(generate_model_url(user_id))
    histories.insert_one(
        {"userId": user_id, "accuracy": accuracy, "numberOfSamples": len(user_samples)}
    )

    logging.info(
        f"Train model for user: ${user['email']} - Number of samples: {len(user_samples)} - Accuracy: {accuracy}"
    )


async def predict_user_samples(user_id: str, samples: List[dict]):
    user = await users.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise Exception("User not found")

    classifier = XGBClassifier()
    classifier.load_model(generate_model_url(user_id))
    proccessed_samples = process_event_data(samples)

    y_pred = classifier.predict_proba(proccessed_samples)
    print(y_pred)
    return y_pred.tolist()
