from model_pipeline import prepare_data, train_model


def test_prepare_data_shapes():
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert "Exited" not in X_train.columns


def test_train_model_predicts_0_or_1():
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(X_train, y_train)
    assert set(model.predict(X_test)) <= {0, 1}
