from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from vendors.supabase import supabase
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from helpers.final_features import compute_features


def data_augmentation(
    df,
    factor=5,
    perturbation_range=(-0.001, 0.001),
    augment_columns=None,
):
    augmented_data = []
    augment_columns = augment_columns or df.columns

    # Generate augmented data
    for _ in range(factor):
        for _, row in df.iterrows():
            synthetic_sample = row.copy()  # Start with a copy of the original row

            for column in augment_columns:
                if column in df.columns:
                    # Randomly perturb the column within the specified range
                    perturbation = np.random.uniform(*perturbation_range)
                    synthetic_sample[column] += perturbation

            augmented_data.append(synthetic_sample)

    # Create a DataFrame from the augmented data
    df_augmented = pd.DataFrame(augmented_data)

    # Combine original and augmented data
    df_combined = pd.concat([df, df_augmented], ignore_index=True)
    return df_combined, df_augmented


def remove_zscore_outlier(df, threshold=3):
    z_scores = df.apply(stats.zscore)
    outlier_count = (z_scores.abs() > threshold).sum(axis=1)

    return df[outlier_count <= 2]


# def process_event_data(input_data):
#     rows = []

#     for sample in input_data:
#         events = sample["events"]
#         df = pd.DataFrame(events)

#         features = compute_features(df)

#         feature_row = {**features}
#         rows.append(feature_row)  # Add the row to the list

#     result_df = pd.DataFrame(rows)


#     return result_df
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


def create_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBClassifier(eval_metric="logloss"),
            ),
        ]
    )


data_folder = Path(__file__).resolve().parent.parent / "data"


def generate_model_url(user_id: str):
    return data_folder / f"{user_id}.json"


data_folder = Path(__file__).resolve().parent.parent / "data"


async def train_model_for_user(user):
    response = (
        supabase.table("samples")
        .select("*")
        .eq("user_id", user.id)
        .eq("is_legitimate", True)
        .execute()
    )

    user_samples = response.data

    # Extract features
    legitimate_data = process_event_data(user_samples)
    legitimate_data, _ = data_augmentation(
        legitimate_data,
        perturbation_range=(-0.002, 0.002),
        factor=4,
        augment_columns=[
            "mean_hold_time1",
            "mean_f1",
            "mean_f2",
            "mean_f3",
            "mean_f4",
        ],
    )
    legitimate_data["label"] = 1
    raw_imposter_data = pd.read_csv(data_folder / "all_users_dataset.csv")
    raw_imposter_data["negative_ud"] = raw_imposter_data["negative_ud%"] / 100
    raw_imposter_data["negative_uu"] = raw_imposter_data["negative_uu%"] / 100
    columns_to_transform = [
        "negative_ud%",
        "negative_uu%",
        "mean_hold_time1",
        "mean_hold_time2",
        "mean_f1",
        "mean_f2",
        "mean_f3",
        "mean_f4",
    ]

    # Transform percentages to decimals
    raw_imposter_data[columns_to_transform] = (
        raw_imposter_data[columns_to_transform] / 1000
    )
    raw_imposter_data = raw_imposter_data[
        [
            "capsLock_usage",
            "negative_ud",
            "negative_uu",
            "rsa_ratio",
            "lsa_ratio",
            "mean_hold_time1",
            "mean_f1",
            "mean_f2",
            "mean_f3",
            "mean_f4",
        ]
    ]
    imposter_data = raw_imposter_data
    imposter_data["label"] = 0

    # Choose one user as legitimate, others as imposters
    train_legitimate, test_legitimate = train_test_split(
        legitimate_data, test_size=0.2, random_state=142
    )
    train_imposter, test_imposter = train_test_split(
        imposter_data, test_size=0.2, random_state=142
    )

    # Combine legitimate and imposter data for training and testing
    train_set = pd.concat([train_legitimate, train_imposter])
    test_set = pd.concat([test_legitimate, test_imposter])

    # Shuffle the datasets (optional, for better randomness)
    train_set = train_set.sample(frac=1, random_state=142).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=142).reset_index(drop=True)

    x_train = train_set.drop(columns=["label"])
    x_test = test_set.drop(columns=["label"])

    y_train = train_set["label"]
    y_test = test_set["label"]

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=142, k_neighbors=2)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

    # # Define the parameter grid
    # param_grid = {
    #     'max_depth': [5, 7, 9, 11],
    #     'subsample': [0.6, 0.8, 1.0],
    #     'min_child_weight': [1, 1.2, 1.4],
    # }

    # rf_model = RandomForestClassifier(random_state=142)

    # # Define the parameter grid for Random Forest
    # param_grid = {
    #     'n_estimators': [100, 500, 1000],        # Number of trees in the forest
    #     'max_depth': [3, 7, 12],        # Maximum depth of the tree
    #     'min_samples_split': [2, 5, 10],       # Minimum samples required to split an internal node
    #     'min_samples_leaf': [1, 2, 4],         # Minimum samples required to be at a leaf node
    # }

    # Set up GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=rf_model,
    #     param_grid=param_grid,
    #     scoring='accuracy',
    #     cv=3,
    #     verbose=1,
    #     n_jobs=-1
    # )

    # SVM Model: RBF kernel with moderate C and default gamma
    svm_model = SVC(
        kernel="rbf", C=1, gamma="auto", probability=True
    )  # Reasonable C for moderate regularization

    # KNN Model: 5 neighbors with distance weighting
    knn_model = KNeighborsClassifier(
        n_neighbors=5, weights="distance"
    )  # Use distance weighting for sensitivity

    # Random Forest Model: Moderate depth with 100 estimators
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2, random_state=42
    )

    voting_model = VotingClassifier(
        estimators=[("svm", svm_model), ("knn", knn_model), ("rf", rf_model)],
        voting="soft",
    )

    param_grid = {
        "xgb__n_estimators": [50, 100, 200],
        "xgb__max_depth": [3, 5, 7],
        "xgb__learning_rate": [0.01, 0.1, 0.2],
        "xgb__subsample": [0.8, 1.0],
        "xgb__scale_pos_weight": [25, 50, 75, 99, 100],
    }

    pipeline = create_pipeline()

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )

    # Perform the grid search
    grid_search.fit(x_train_balanced, y_train_balanced)

    # # Best parameters and score
    # print(f"Best Parameters: {grid_search.best_params_}")
    # print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

    # # Evaluate on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
    joblib.dump(best_model, generate_model_url(user.id))

    # Metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    supabase.table("samples").insert(
        {
            "user_id": user.id,
            "accuracy": accuracy_score(y_test, y_pred),
            "num_of_samples": len(user_samples),
        }
    ).execute()


def is_model_existed(user_id: str):
    return Path(generate_model_url(user_id=user_id)).is_file()


async def predict_user_samples(user, samples: List[dict]):
    if user is None:
        raise Exception("User not found")
    # Load the saved Random Forest model
    classifier = joblib.load(generate_model_url(user.id))

    # Process the samples
    processed_samples = process_event_data(samples)

    # Make predictions
    y_pred = classifier.predict(processed_samples)
    print(classifier.predict_proba(processed_samples))

    # Return predictions as a list
    return y_pred.tolist()
