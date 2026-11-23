import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]
TARGET = "Exited"


def prepare_data(csv_path="Churn_Modelling.csv"):
    """Load the CSV, clean it and split it into train and test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = df.drop(columns=COLUMNS_TO_DROP)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=1)


def train_model(X_train, y_train):
    """Train a Random Forest classifier and return it."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print the performance of the model and return the accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy * 100:.2f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, path="classifier.joblib"):
    """Save the trained model to a file with joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path="classifier.joblib"):
    """Load a saved model from a file."""
    return joblib.load(path)
