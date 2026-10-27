import unittest

import numpy as np

from pattern_recognition.data import load_processed_mnist
from pattern_recognition.models.from_scratch.logistic_regression import (
    LogisticRegressionNP,
)
from pattern_recognition.models.from_scratch.mlp import build_mlp, train_model
from pattern_recognition.models.from_scratch.naive_bayes import (
    MultinomialNaiveBayes,
    NaiveBayes,
    gaussian_log_pdf,
    multinomial_log_pdf,
)
from pattern_recognition.models.from_scratch.random_forest import (
    RandomForestFromScratch,
)
from pattern_recognition.models.pytorch.cnn import MNISTCNN


class FromScratchModelSmokeTests(unittest.TestCase):
    def test_naive_bayes_models_fit_and_predict(self):
        X = np.array([[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.0, 0.8]])
        y = np.array([0, 0, 1, 1])

        gaussian = NaiveBayes(gaussian_log_pdf).fit(X, y)
        multinomial = MultinomialNaiveBayes(multinomial_log_pdf).fit(X, y)

        np.testing.assert_array_equal(gaussian.predict(X), y)
        multinomial_predictions = multinomial.predict(X)
        self.assertEqual(multinomial_predictions.shape, y.shape)
        self.assertTrue(np.isin(multinomial_predictions, multinomial.classes).all())

    def test_logistic_regression_fit_and_predict(self):
        X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegressionNP(
            lr=0.2, epochs=100, batch_size=4, l2=0.0, seed=7
        ).fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)

    def test_random_forest_fit_and_predict(self):
        X = np.array([[0.0], [0.1], [0.9], [1.0]])
        y = np.array([0, 0, 1, 1])
        model = RandomForestFromScratch(
            n_estimators=5, max_depth=3, random_state=7
        ).fit(X, y)

        predictions = model.predict(X)
        self.assertEqual(predictions.shape, y.shape)
        self.assertTrue(np.isin(predictions, np.unique(y)).all())

    def test_mlp_manual_training_api_and_cnn_output_shape(self):
        import torch

        X = torch.tensor([[-1.0, -0.5], [-0.5, -1.0], [0.5, 1.0], [1.0, 0.5]])
        y = torch.tensor([0, 0, 1, 1])
        model = build_mlp([2, 4, 2], seed=7)
        was_grad_enabled = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            history = train_model(
                model,
                epochs=2,
                lr=0.1,
                batch_size=4,
                X_tr=X,
                y_tr=y,
                X_va=X,
                y_va=y,
                verbose=False,
            )
        finally:
            torch.set_grad_enabled(was_grad_enabled)

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(MNISTCNN()(torch.zeros(2, 1, 28, 28)).shape, (2, 10))


class DataLoaderTests(unittest.TestCase):
    def test_missing_processed_data_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Run plan/phase0"):
            load_processed_mnist("missing-mnist-data.npz")


if __name__ == "__main__":
    unittest.main()
