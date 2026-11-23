import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Churn ML pipeline")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        required=True,
        help="which step to run",
    )
    parser.add_argument("--data", default="Churn_Modelling.csv", help="CSV file")
    parser.add_argument("--model-path", default="classifier.joblib", help="model file")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.step == "prepare":
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    elif args.step == "train":
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)

    elif args.step == "evaluate":
        model = load_model(args.model_path)
        evaluate_model(model, X_test, y_test)

    elif args.step == "all":
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, args.model_path)


if __name__ == "__main__":
    main()
