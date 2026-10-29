import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pattern_recognition.models.from_scratch.mlp import forward_with_cache
from pattern_recognition.models.pytorch.cnn import predict_cnn


def evaluate_mlp_full(model, X, y):
    t0 = time.time()
    logits, _ = forward_with_cache(model, X, train=False)
    y_pred = logits.argmax(dim=1).numpy()
    t_pred = time.time() - t0
    y_true = y.numpy()
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro")
    r = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, p, r, f1, t_pred, y_pred

def evaluate_cnn_full(model, X, y):
    t0 = time.time()
    y_pred_t, _, _ = predict_cnn(model, X)
    t_pred = time.time() - t0
    y_pred = y_pred_t.numpy()
    y_true = y.numpy()
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro")
    r = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, p, r, f1, t_pred, y_pred
