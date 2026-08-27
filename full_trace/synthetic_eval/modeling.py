import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.intp)
    y_pred = np.asarray(y_pred, dtype=np.intp)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision_good": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_good": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_good": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0,
        "pred_good": int(np.sum(y_pred == 1)),
        "pred_bad": int(np.sum(y_pred == 0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def make_models(seed=42):
    return {
        "rf_200": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    }


def fit_eval_models(train_features, train_labels, eval_sets, seed=42):
    rows = []
    models = make_models(seed=seed)
    for model_name, model in models.items():
        if len(np.unique(train_labels)) < 2:
            continue
        model.fit(train_features, train_labels)
        for eval_name, (X_eval, y_eval) in eval_sets.items():
            pred = model.predict(X_eval)
            row = {"model": model_name, "eval_set": eval_name}
            row.update(metrics_dict(y_eval, pred))
            rows.append(row)
    return rows


def synthetic_train_test_split(features, labels, test_size=0.25, seed=42):
    labels = np.asarray(labels, dtype=np.intp)
    stratify = labels if len(np.unique(labels)) > 1 and min(np.bincount(labels)) >= 2 else None
    return train_test_split(features, labels, test_size=test_size, random_state=seed, stratify=stratify)


def real_vs_synthetic_discriminator(real_features, synth_features, seed=42):
    n = min(len(real_features), len(synth_features))
    if n < 8:
        return {"n_real": int(len(real_features)), "n_synthetic": int(len(synth_features)), "accuracy": np.nan, "balanced_accuracy": np.nan}
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(np.arange(len(real_features)), size=n, replace=False)
    synth_idx = rng.choice(np.arange(len(synth_features)), size=n, replace=False)
    X = np.vstack([real_features[real_idx], synth_features[synth_idx]])
    y = np.concatenate([np.ones(n, dtype=np.intp), np.zeros(n, dtype=np.intp)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "n_real": int(len(real_features)),
        "n_synthetic": int(len(synth_features)),
        "heldout_n": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }

