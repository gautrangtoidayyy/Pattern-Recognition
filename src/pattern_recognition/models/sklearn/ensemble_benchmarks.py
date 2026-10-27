import time

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score


ENSEMBLE_TYPES = (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)

def benchmark(models, X_tr, X_te, y_tr, y_te):
    # Ham chuan hoa benchmark ke thua chinh xac tu Lab 05
    rows = []
    trained_models = {}
    for name, model in models.items():
        print(f"Đang huấn luyện mô hình: {name:<22} ...", end="", flush=True)
        m = clone(model)
        
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = m.predict(X_te)
        t_pred = time.perf_counter() - t0
        
        acc = accuracy_score(y_te, y_pred)
        f1_m = f1_score(y_te, y_pred, average="macro")
        
        row = {
            "Mô hình": name,
            "Loại": "Ensemble" if isinstance(model, ENSEMBLE_TYPES) else "Đơn",
            "Accuracy": acc,
            "Macro F1": f1_m,
            "Train (s)": t_fit,
            "Predict (s)": t_pred
        }
        rows.append(row)
        trained_models[name] = m
        print(f" Hoàn tất! Acc: {acc:.4f} | Train: {t_fit:6.2f}s | Pred: {t_pred:5.3f}s")
        
    df = pd.DataFrame(rows).set_index("Mô hình")
    return df.sort_values("Accuracy", ascending=False).round(4), trained_models
