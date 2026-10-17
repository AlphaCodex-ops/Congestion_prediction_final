import numpy as np

from congestion_project.training import ModelTrainer, Predictor


def test_training_persistence_and_prediction_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(160, 3))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = rng.normal(size=(60, 3))
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler().fit(X_train)
    model = ModelTrainer(
        scaler.transform(X_train),
        y_train,
        scaler.transform(X_test),
        y_test,
        {"n_estimators": 30, "random_state": 7},
        ["a", "b", "c"],
    )
    model.train()
    model_path = model.save_model(tmp_path / "model.joblib")
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    predictor = Predictor(model_path, scaler_path)
    probability = predictor.predict(X_test[0])
    assert 0 <= probability <= 1
    assert predictor.predict_batch(X_test).shape == (len(X_test),)
    assert predictor.get_feature_names() == ["a", "b", "c"]
    assert "confusion_matrix" in model.evaluate()
