#!/usr/bin/env python3
"""Train and evaluate several feed-forward neural networks on EchoNext metadata.

Converted from the NN_smalldata.ipynb notebook into a single runnable script.

Usage:
    python nn_smalldata.py --csv echonext_metadata_100k.csv --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_CSV = "echonext_metadata_100k.csv"
DEFAULT_OUTPUT_DIR = "outputs"
N_TRAIN = 8000
N_TEST  = 2000
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
RANDOM_SEED = 92
N_NEURONS = 20
DROPOUT = 0.5

RESULT_FILENAMES = {
    ("3", "all"): "NN_roc_results_3_ALLDATA_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("3", "ECG2"): "NN_roc_results_3_ECG_ECHO_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("3", "ECG"): "NN_roc_results_3_ECG_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("5", "all"): "NN_roc_results_5_ALLDATA_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("5", "ECG2"): "NN_roc_results_5_ECG_ECHO_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("5", "ECG"): "NN_roc_results_5_ECG_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("10", "all"): "NN_roc_results_10_ALLDATA_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("10", "ECG2"): "NN_roc_results_10_ECG_ECHO_20neuron_8000train_2000test_dropout50_RELU.npz",
    ("10", "ECG"): "NN_roc_results_10_ECG_20neuron_8000train_2000test_dropout50_RELU.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to input CSV file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for result .npz files")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Training epochs per model family")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--n-train", type=int, default=N_TRAIN, help="Number of training rows to use")
    parser.add_argument("--n-test", type=int, default=N_TEST, help="Number of test rows to use")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    return parser.parse_args()


class SimpleClassifier3(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, N_NEURONS)
        self.fc2 = nn.Linear(N_NEURONS, N_NEURONS)
        self.fc3 = nn.Linear(N_NEURONS, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.output(x)


class SimpleClassifier5(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, N_NEURONS)
        self.fc2 = nn.Linear(N_NEURONS, N_NEURONS)
        self.fc3 = nn.Linear(N_NEURONS, N_NEURONS)
        self.fc4 = nn.Linear(N_NEURONS, N_NEURONS)
        self.fc5 = nn.Linear(N_NEURONS, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return self.output(x)


class SimpleClassifier10(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, N_NEURONS)] + [nn.Linear(N_NEURONS, N_NEURONS) for _ in range(8)] + [nn.Linear(N_NEURONS, num_classes)]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return self.output(x)


MODEL_SPECS = {
    "3": SimpleClassifier3,
    "5": SimpleClassifier5,
    "10": SimpleClassifier10,
}


FEATURE_SETS = {
    "all": ["sex_OH", "age", "pr_val", "qrs_val", "atr_rate", "vent_rate", "qt_val", "peri_efu_OH", "LVPW", "IVS", "TR_vel", "EF"],
    "ECG2": ["sex_OH", "age", "pr_val", "qrs_val", "atr_rate", "vent_rate", "qt_val", "peri_efu_OH", "LVPW", "IVS"],
    "ECG": ["sex_OH", "age", "pr_val", "qrs_val", "atr_rate", "vent_rate", "qt_val"],
}





def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_labels(series: pd.Series, fill_value: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    values = series.copy()
    if fill_value is not None:
        values = values.fillna(fill_value)
    encoded = encoder.fit_transform(values.astype(str).to_numpy())
    return encoded, encoder.classes_


def to_float_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values.astype(np.float32))


def prepare_features(df: pd.DataFrame) -> dict[str, torch.Tensor | np.ndarray]:
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    imp_pap = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=20)

    sex, sex_classes = encode_labels(df["sex"], fill_value="none")
    age = imp.fit_transform(df[["age_at_ecg"]])
    race, race_classes = encode_labels(df["race_ethnicity"], fill_value="none")

    vent_rate = imp.fit_transform(df[["ventricular_rate"]])
    atr_rate = imp.fit_transform(df[["atrial_rate"]])
    pr_val = imp.fit_transform(df[["pr_interval"]])
    qrs_val = imp.fit_transform(df[["qrs_duration"]])
    qt_val = imp.fit_transform(df[["qt_corrected"]])

    rv_func, rv_func_classes = encode_labels(df["rv_systolic_function_value"], fill_value="none")
    peri_efu, peri_classes = encode_labels(df["pericardial_effusion_value"], fill_value="none")
    tric2, tric_classes = encode_labels(df["tricuspid_regurgitation_value"], fill_value="none")

    ef = imp.fit_transform(df[["lvef_value"]])
    rv_flag, rv_flag_classes = encode_labels(df["rv_systolic_dysfunction_moderate_or_greater_flag"], fill_value="none")

    pap = imp_pap.fit_transform(df[["pasp_value"]])
    ivs = imp.fit_transform(df[["ivs_measurement"]])
    lvpw = imp.fit_transform(df[["lvpw_measurement"]])
    tr_vel = imp.fit_transform(df[["tr_max_velocity_value"]])
    split_id, id_classes = encode_labels(df["split"], fill_value="none")
    shd, shd_classes = encode_labels(df["shd_moderate_or_greater_flag"], fill_value="none")

    features: dict[str, torch.Tensor | np.ndarray] = {
        "sex_classes": sex_classes,
        "race_classes": race_classes,
        "rv_func_classes": rv_func_classes,
        "peri_classes": peri_classes,
        "tric_classes": tric_classes,
        "rv_flag_classes": rv_flag_classes,
        "id_classes": id_classes,
        "shd_classes": shd_classes,
        "age": to_float_tensor(age),
        "vent_rate": to_float_tensor(vent_rate),
        "atr_rate": to_float_tensor(atr_rate),
        "pr_val": to_float_tensor(pr_val),
        "qrs_val": to_float_tensor(qrs_val),
        "qt_val": to_float_tensor(qt_val),
        "EF": to_float_tensor(ef),
        "PAP": to_float_tensor(pap),
        "IVS": to_float_tensor(ivs),
        "LVPW": to_float_tensor(lvpw),
        "TR_vel": to_float_tensor(tr_vel),
        "sex_OH": F.one_hot(torch.tensor(sex, dtype=torch.long), num_classes=max(2, len(sex_classes))).to(torch.float32),
        "race_OH": F.one_hot(torch.tensor(race, dtype=torch.long), num_classes=max(6, len(race_classes))).to(torch.float32),
        "RV_func_OH": F.one_hot(torch.tensor(rv_func, dtype=torch.long), num_classes=max(5, len(rv_func_classes))).to(torch.float32),
        "peri_efu_OH": F.one_hot(torch.tensor(peri_efu, dtype=torch.long), num_classes=max(6, len(peri_classes))).to(torch.float32),
        "tric_OH": F.one_hot(torch.tensor(tric2, dtype=torch.long), num_classes=max(5, len(tric_classes))).to(torch.float32),
        "RVflag_OH": F.one_hot(torch.tensor(rv_flag, dtype=torch.long), num_classes=max(2, len(rv_flag_classes))).to(torch.float32),
        "SHD_OH": F.one_hot(torch.tensor(shd, dtype=torch.long), num_classes=max(2, len(shd_classes))).to(torch.float32),
        "id": split_id,
        "SHD": shd,
    }
    return features


def debug_feature_dtypes(features: dict[str, torch.Tensor | np.ndarray]) -> None:
    vars_to_check = {
        "sex": features["sex_OH"],
        "age": features["age"],
        "race": features["race_OH"],
        "vent_rate": features["vent_rate"],
        "atr_rate": features["atr_rate"],
        "pr_val": features["pr_val"],
        "qrs_val": features["qrs_val"],
        "qt_correct": features["qt_val"],
        "RV_func": features["RV_func_OH"],
        "peri_efu": features["peri_efu_OH"],
        "tric2": features["tric_OH"],
        "EF": features["EF"],
        "PAP": features["PAP"],
        "IVS": features["IVS"],
        "LVPW": features["LVPW"],
        "TR_vel": features["TR_vel"],
        "id": features["id"],
        "SHD": features["SHD"],
    }

    print("Variable dtypes:\n" + "-" * 40)
    for name, var in vars_to_check.items():
        if hasattr(var, "dtype"):
            try:
                unique_shape = np.shape(np.unique(var))
            except Exception:
                unique_shape = "n/a"
            print(f"{name:12s} : {var.dtype} : {unique_shape}")
        else:
            print(f"{name:12s} : {type(var)}")


def build_split_indices(features: dict[str, torch.Tensor | np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_id = np.asarray(features["id"])
    pap = features["PAP"]
    id_train = np.where(split_id == 2)[0]
    id_test = np.where(split_id != 2)[0]
    return id_train, id_test


def concat_feature_set(features: dict[str, torch.Tensor | np.ndarray], names: list[str]) -> torch.Tensor:
    tensors = [features[name] for name in names]
    return torch.cat(tensors, dim=1)


def prepare_train_test_tensors(
    X_full: torch.Tensor,
    y: np.ndarray,
    id_train: np.ndarray,
    id_test: np.ndarray,
    n_train: int,
    n_test: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scaler = StandardScaler()

    X_train = X_full[id_train, :][:n_train]
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = y[id_train][:n_train]

    X_test = X_full[id_test, :][:n_test]
    X_test_scaled = scaler.transform(X_test)
    y_test = y[id_test][:n_test]

    return (
        torch.tensor(X_train_scaled.astype(np.float32)),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test_scaled.astype(np.float32)),
        torch.tensor(y_test, dtype=torch.long),
    )


def train_model_family(
    model_class: type[nn.Module],
    dataloaders: dict[str, DataLoader],
    input_sizes: dict[str, int],
    num_classes: int,
    epochs: int,
) -> dict[str, nn.Module]:
    criterion = nn.CrossEntropyLoss()
    models = {name: model_class(input_sizes[name], num_classes) for name in dataloaders}
    optimizers = {name: optim.Adam(model.parameters(), lr=LEARNING_RATE) for name, model in models.items()}

    for epoch in range(epochs):
        losses = {name: 0.0 for name in dataloaders}
        for name, loader in dataloaders.items():
            model = models[name]
            model.train()
            optimizer = optimizers[name]
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                losses[name] += loss.item()

        if epoch % 5 == 0:
            avg_losses = {name: losses[name] / max(len(loader), 1) for name, loader in dataloaders.items()}
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Lossall: {avg_losses['all']:.4f}, "
                f"LossECG2: {avg_losses['ECG2']:.4f}, "
                f"LossECG: {avg_losses['ECG']:.4f}"
            )

    return models


def predict_fn_single(X_np: np.ndarray, model: nn.Module, device: str = "cpu") -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X_np.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = F.softmax(logits, dim=1)[:, 1]
    return probs.cpu().numpy()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_deterministic_predictions(
    predict_fn: Callable,
    X: np.ndarray,
    y_true: np.ndarray,
    site: str = "probs",
    mode: str = "none",
    n_samples: int = 200,
    bootstrap_batch_size: int | None = None,
    threshold: float = 0.5,
    plot: bool = True,
    fpr_grid: np.ndarray | None = None,
    ci: tuple[float, float] = (2.5, 97.5),
    plot_sample_rocs: int = 40,
    random_state: int = 0,
    mc_dropout_kwargs: dict | None = None,
) -> dict[str, np.ndarray | float | tuple[float, float]]:
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y_true = np.asarray(y_true).astype(int)
    n_obs = X.shape[0]

    if fpr_grid is None:
        fpr_grid = np.linspace(0.0, 1.0, 201)

    def get_probs_for_X(X_in: np.ndarray, **kwargs) -> np.ndarray:
        out = np.asarray(predict_fn(X_in, **kwargs))
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.reshape(-1)
        if site == "probs":
            return out
        if site == "logits":
            return sigmoid(out)
        raise ValueError("site must be 'probs' or 'logits'")

    if mode == "none":
        probs_samples = get_probs_for_X(X)[np.newaxis, :]
    elif mode == "mc_dropout":
        probs_samples = np.stack(
            [get_probs_for_X(X, **(mc_dropout_kwargs or {})) for _ in range(n_samples)],
            axis=0,
        )
    elif mode == "bootstrap":
        batch_size = bootstrap_batch_size or n_obs
        probs_list = []
        for _ in range(n_samples):
            idx = rng.randint(0, n_obs, size=batch_size)
            probs_b = get_probs_for_X(X[idx])
            probs_full = np.full(n_obs, np.mean(probs_b), dtype=float)
            probs_full[idx] = probs_b
            probs_list.append(probs_full)
        probs_samples = np.stack(probs_list, axis=0)
    else:
        raise ValueError("mode must be 'none', 'mc_dropout', or 'bootstrap'")

    s_used = probs_samples.shape[0]
    probs_mean = probs_samples.mean(axis=0)

    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, probs_mean)
    roc_auc_val = auc(roc_fpr, roc_tpr)

    tpr_samples_grid = np.zeros((s_used, len(fpr_grid)))
    auc_samples = np.zeros(s_used)
    for i in range(s_used):
        probs_i = probs_samples[i]
        try:
            fpr_i, tpr_i, _ = roc_curve(y_true, probs_i)
            auc_samples[i] = roc_auc_score(y_true, probs_i)
        except ValueError:
            fpr_i = np.array([0.0, 1.0])
            tpr_i = np.array([0.0, 1.0])
            auc_samples[i] = 0.5
        tpr_samples_grid[i] = np.interp(fpr_grid, fpr_i, tpr_i)

    lower_pct, upper_pct = ci
    tpr_mean = tpr_samples_grid.mean(axis=0)
    tpr_lower = np.percentile(tpr_samples_grid, lower_pct, axis=0)
    tpr_upper = np.percentile(tpr_samples_grid, upper_pct, axis=0)
    auc_mean = float(np.mean(auc_samples))
    auc_ci = (float(np.percentile(auc_samples, lower_pct)), float(np.percentile(auc_samples, upper_pct)))

    y_pred = (probs_mean >= threshold).astype(int)
    confmat = confusion_matrix(y_true, y_pred)

    print(f"ROC AUC (mean probs): {roc_auc_val:.4f}")
    print(f"AUC (samples mean): {auc_mean:.4f}, {lower_pct}/{upper_pct}% CI = ({auc_ci[0]:.4f}, {auc_ci[1]:.4f})")
    print(f"\nClassification report (threshold = {threshold}):\n")
    print(classification_report(y_true, y_pred, digits=4))

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        to_plot = min(plot_sample_rocs, s_used)
        sample_idxs = rng.choice(s_used, size=to_plot, replace=False)
        for si in sample_idxs:
            plt.plot(fpr_grid, tpr_samples_grid[si], color="gray", alpha=0.12, linewidth=0.8)
        plt.plot(fpr_grid, tpr_mean, color="C0", lw=2, label=f"Mean ROC (AUC mean={auc_mean:.3f})")
        plt.fill_between(fpr_grid, tpr_lower, tpr_upper, color="C0", alpha=0.25, label=f"{lower_pct}/{upper_pct}% band")
        plt.plot(roc_fpr, roc_tpr, color="C1", linestyle="--", lw=1.5, label=f"ROC(mean probs) AUC={roc_auc_val:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="k", linewidth=0.8)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC with {mode} band")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=[0, 1])
        disp.plot(ax=plt.gca(), cmap=None, colorbar=True)
        plt.title(f"Confusion matrix (threshold={threshold})")
        plt.tight_layout()
        plt.show()

    return {
        "auc_mean": auc_mean,
        "auc_ci": auc_ci,
        "auc_samples": auc_samples,
        "fpr_grid": fpr_grid,
        "tpr_mean": tpr_mean,
        "tpr_lower": tpr_lower,
        "tpr_upper": tpr_upper,
        "tpr_samples": tpr_samples_grid,
        "probs_mean": probs_mean,
        "y_pred": y_pred,
        "confusion_matrix": confmat,
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "roc_thresholds": roc_thresholds,
        "probs_samples": probs_samples,
    }


def save_results(results: dict, output_path: Path) -> None:
    np.savez_compressed(
        output_path,
        auc_mean=results["auc_mean"],
        auc_ci=np.array(results["auc_ci"]),
        auc_samples=results["auc_samples"],
        fpr_grid=results["fpr_grid"],
        tpr_mean=results["tpr_mean"],
        tpr_lower=results["tpr_lower"],
        tpr_upper=results["tpr_upper"],
        tpr_samples=results["tpr_samples"],
        probs_mean=results["probs_mean"],
        confusion_matrix=results["confusion_matrix"],
        ypred=results["y_pred"],
        roc_fpr=results["roc_fpr"],
        roc_tpr=results["roc_tpr"],
        roc_thresholds=results["roc_thresholds"],
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(df.head())
    print("DataFrame shape:", df.shape)

    features = prepare_features(df)
    debug_feature_dtypes(features)
    print(features["SHD"])

    id_train, id_test = build_split_indices(features)
    y = np.asarray(features["SHD"])  # target used in the notebook

    train_tensors: dict[str, torch.Tensor] = {}
    test_tensors: dict[str, torch.Tensor] = {}

    for feature_name, feature_columns in FEATURE_SETS.items():
        X_full = concat_feature_set(features, feature_columns)
        X_train, y_train, X_test, y_test = prepare_train_test_tensors(
            X_full=X_full,
            y=y,
            id_train=id_train,
            id_test=id_test,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        train_tensors[feature_name] = X_train
        test_tensors[feature_name] = X_test

    y_train_tens = y_train
    y_test_tens = y_test

    dataloaders = {
        name: DataLoader(TensorDataset(train_tensors[name], y_train_tens), batch_size=args.batch_size, shuffle=True)
        for name in FEATURE_SETS
    }
    input_sizes = {name: train_tensors[name].shape[1] for name in FEATURE_SETS}

    trained_models: dict[str, dict[str, nn.Module]] = {}
    for family_name, model_class in MODEL_SPECS.items():
        trained_models[family_name] = train_model_family(
            model_class=model_class,
            dataloaders=dataloaders,
            input_sizes=input_sizes,
            num_classes=2,
            epochs=args.epochs,
        )

    y_test_np = y_test_tens.cpu().numpy()
    print("y_test_np shape:", y_test_np.shape)

    for family_name, family_models in trained_models.items():
        for feature_name, model in family_models.items():
            results = evaluate_deterministic_predictions(
                predict_fn=lambda X_np, m=model, **kwargs: predict_fn_single(X_np, model=m, device="cpu"),
                X=test_tensors[feature_name].cpu().numpy(),
                y_true=y_test_np,
                site="probs",
                mode="none",
                n_samples=1,
                threshold=0.5,
                plot=True,
            )
            save_results(results, output_dir / RESULT_FILENAMES[(family_name, feature_name)])


if __name__ == "__main__":
    main()
