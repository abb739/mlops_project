import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, get_run_logger, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DATA_FILE = "Churn_Modelling.csv"
MODEL_FILE = "classifier.joblib"


def _run(command):
    """Run a shell command with the current Python and stop if it fails."""
    logger = get_run_logger()
    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)  # nosec B603
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(command)}")


# ---------- TASKS ----------
@task(name="install-dependencies")
def install_dependencies():
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_code():
    _run([sys.executable, "-m", "black", *CODE_FILES])


@task(name="code-quality")
def code_quality():
    _run([sys.executable, "-m", "flake8", "--max-line-length=100", *CODE_FILES])


@task(name="code-security")
def code_security():
    _run([sys.executable, "-m", "bandit", "-q", *CODE_FILES])


@task(name="unit-tests")
def unit_tests():
    _run([sys.executable, "-m", "pytest", "-q", "test_pipeline.py"])


@task(name="prepare-data")
def prepare():
    return prepare_data(DATA_FILE)


@task(name="train-model")
def train(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="save-model")
def save(model):
    save_model(model, MODEL_FILE)


@task(name="load-model")
def load():
    return load_model(MODEL_FILE)


@task(name="evaluate-model")
def evaluate(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@task(name="predict")
def predict(model):
    import pandas as pd

    customer = pd.DataFrame(
        [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]],
        columns=[
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ],
    )
    result = int(model.predict(customer)[0])
    get_run_logger().info("Prediction (1 = leaves, 0 = stays): %s", result)
    return result


# ---------- FLOWS ----------
@flow(name="install", log_prints=True)
def flow_install():
    install_dependencies()


@flow(name="code", log_prints=True)
def flow_code():
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    X_train, X_test, y_train, y_test = prepare()
    model = train(X_train, y_train)
    save(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    X_train, X_test, y_train, y_test = prepare()
    model = load()
    evaluate(model, X_test, y_test)


@flow(name="predict", log_prints=True)
def flow_predict():
    predict(load())


@flow(name="all", log_prints=True)
def flow_all():
    install_dependencies()
    flow_code()
    X_train, X_test, y_train, y_test = prepare()
    model = train(X_train, y_train)
    save(model)
    evaluate(model, X_test, y_test)


FLOWS = {
    "install": flow_install,
    "code": flow_code,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "all": flow_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect pipeline")
    parser.add_argument("--flow", choices=FLOWS.keys(), required=True)
    args = parser.parse_args()
    FLOWS[args.flow]()
