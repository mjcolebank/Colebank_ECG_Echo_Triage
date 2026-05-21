#!/usr/bin/env python3
"""Train and evaluate several Bayesian neural networks on EchoNext metadata with variational inference.

Converted from the BNN_smalldata.ipynb notebook into a single runnable script,
with structure aligned to nn_smalldata.py.

This version uses variational inference with SVI + AutoNormal instead of HMC/NUTS.

Usage:
    python bnn_smalldata_vi.py --csv echonext_metadata_100k.csv --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import torch
import torch.nn.functional as F
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

DEFAULT_CSV = "echonext_metadata_100k.csv"
DEFAULT_OUTPUT_DIR = "outputs"
N_TRAIN = 70000
N_TEST = 20000
SVI_STEPS = 10000
LEARNING_RATE = 1e-3
NUM_PARTICLES = 1
NUM_POSTERIOR_SAMPLES = 2000
RANDOM_SEED = 72
HIDDEN_DIM = 20
PRIOR_SCALE = 0.25
NUM_CLASSES = 2

RESULT_FILENAMES = {
    ("3", "all"): "bnn_vi_roc_results_3_20neuron_full_70000train_20000test_prior25_GELU.npz",
    ("3", "ECG2"): "bnn_vi_roc_results_3_20neuron_ECG_Echo_70000train_20000test_prior25_GELU.npz",
    ("3", "ECG"): "bnn_vi_roc_results_3_20neuron_ECG_only_70000train_20000test_prior25_GELU.npz",
    ("5", "all"): "bnn_vi_roc_results_5_20neuron_full_70000train_20000test_prior25_GELU.npz",
    ("5", "ECG2"): "bnn_vi_roc_results_5_20neuron_ECG_Echo_70000train_20000test_prior25_GELU.npz",
    ("5", "ECG"): "bnn_vi_roc_results_5_20neuron_ECG_only_70000train_20000test_prior25_GELU.npz",
    ("10", "all"): "bnn_vi_roc_results_10_20neuron_full_70000train_20000test_prior25_GELU.npz",
    ("10", "ECG2"): "bnn_vi_roc_results_10_20neuron_ECG_Echo_70000train_20000test_prior25_GELU.npz",
    ("10", "ECG"): "bnn_vi_roc_results_10_20neuron_ECG_only_70000train_20000test_prior25_GELU.npz",
}


FEATURE_SETS = {
    "all": ["sex_OH", "age", "pr_val", "qrs_val", "atr_rate", "vent_rate", "qt_val", "peri_efu_OH", "LVPW", "IVS", "TR_vel", "EF"],
    "ECG2": ["sex_OH", "age", "pr_val", "qrs_val", "atr_rate", "vent_rate", "qt_val", "peri_efu_OH", "LVPW", "IVS"],
    "ECG": ["sex_OH", "age", "pr_val", "qrs_val", "atr_rate", "vent_rate", "qt_val"],
}





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to input CSV file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for result .npz files")
    parser.add_argument("--n-train", type=int, default=N_TRAIN, help="Number of training rows to use")
    parser.add_argument("--n-test", type=int, default=N_TEST, help="Number of test rows to use")
    parser.add_argument("--svi-steps", type=int, default=SVI_STEPS, help="Number of SVI optimization steps")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Adam learning rate for SVI")
    parser.add_argument("--num-particles", type=int, default=NUM_PARTICLES, help="Number of ELBO particles per SVI step")
    parser.add_argument("--posterior-predictive-samples", type=int, default=NUM_POSTERIOR_SAMPLES, help="Samples drawn from the variational posterior for posterior prediction")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Hidden layer width")
    parser.add_argument("--prior-scale", type=float, default=PRIOR_SCALE, help="Normal prior scale for weights and biases")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Show evaluation plots")
    return parser.parse_args()


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

    return {
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
    id_train = np.where(split_id == 2)[0]
    id_val = np.where(split_id == 1)[0]
    id_test = np.where(split_id != 2)[0]
    return id_train, id_val, id_test


def concat_feature_set(features: dict[str, torch.Tensor | np.ndarray], names: list[str]) -> torch.Tensor:
    tensors = [features[name] for name in names]
    return torch.cat(tensors, dim=1)


def prepare_train_test_arrays(
    X_full: torch.Tensor,
    y: np.ndarray,
    id_train: np.ndarray,
    id_test: np.ndarray,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()

    X_train = X_full[id_train, :][:n_train].cpu().numpy()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = y[id_train][:n_train]

    X_test = X_full[id_test, :][:n_test].cpu().numpy()
    X_test_scaled = scaler.transform(X_test)
    y_test = y[id_test][:n_test]

    return (
        X_train_scaled.astype(np.float32),
        y_train.astype(np.int32),
        X_test_scaled.astype(np.float32),
        y_test.astype(np.int32),
    )


def bnn_model_factory(num_hidden_layers: int) -> Callable:
    def model(x: jnp.ndarray, y: jnp.ndarray | None = None, hidden_dim: int = HIDDEN_DIM, prior_scale: float = 1.0, num_classes: int = NUM_CLASSES):
        width_in = x.shape[1]
        hidden = x

        for i in range(num_hidden_layers):
            w = numpyro.sample(f"w{i+1}", dist.Normal(0.0, prior_scale).expand([width_in, hidden_dim]).to_event(2))
            b = numpyro.sample(f"b{i+1}", dist.Normal(0.0, prior_scale).expand([hidden_dim]).to_event(1))
            # hidden = jax.nn.tanh(jnp.dot(hidden, w) + b)
            # hidden = jax.nn.relu(jnp.dot(hidden, w) + b)
            hidden = jax.nn.gelu(jnp.dot(hidden, w) + b)
            width_in = hidden_dim

        w_out = numpyro.sample(f"w{num_hidden_layers+1}", dist.Normal(0.0, prior_scale).expand([width_in, num_classes]).to_event(2))
        b_out = numpyro.sample(f"b{num_hidden_layers+1}", dist.Normal(0.0, prior_scale).expand([num_classes]).to_event(1))
        logits = jnp.dot(hidden, w_out) + b_out

        numpyro.deterministic("logits", logits)

        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        return logits

    return model


MODEL_SPECS = {
    "3": bnn_model_factory(2),
    "5": bnn_model_factory(4),
    "10": bnn_model_factory(9),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def evaluate_posterior_predictions(
    pred_samples: dict,
    y_true: np.ndarray,
    site: str = "logits",
    threshold: float = 0.5,
    plot: bool = True,
    n_posterior_samples: int | None = None,
    fpr_grid: np.ndarray | None = None,
    ci: tuple[float, float] = (2.5, 97.5),
    plot_sample_rocs: int = 40,
    random_state: int = 0,
    positive_class: int = 1,
) -> dict[str, np.ndarray | float | tuple[float, float] | str | int]:
    y_true = np.asarray(y_true)

    if site not in pred_samples:
        raise ValueError(f"site '{site}' not found in pred_samples")

    samples = np.asarray(pred_samples[site])

    if site == "obs":
        if samples.ndim != 2:
            raise ValueError(f"For site='obs', expected samples shape (S, N), got {samples.shape}")
        s_all, n_obs = samples.shape
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
    else:
        if samples.ndim == 2:
            s_all, n_obs = samples.shape
            n_classes = 2
            is_binary = True
        elif samples.ndim == 3 and samples.shape[-1] == 1:
            samples = samples[..., 0]
            s_all, n_obs = samples.shape
            n_classes = 2
            is_binary = True
        elif samples.ndim == 3:
            s_all, n_obs, n_classes = samples.shape
            is_binary = n_classes == 2
        else:
            raise ValueError(f"For logits, expected shape (S,N), (S,N,1), or (S,N,K), got {samples.shape}")

    if len(y_true) != n_obs:
        raise ValueError(f"y_true has length {len(y_true)} but predictions have N={n_obs}")

    if n_posterior_samples is None or n_posterior_samples > s_all:
        n_posterior_samples = s_all

    if n_posterior_samples < s_all:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(s_all, size=n_posterior_samples, replace=False)
        samples_sel = samples[idx]
    else:
        samples_sel = samples

    if site == "obs":
        if is_binary:
            probs_samples = samples_sel.astype(float)
        else:
            probs_samples = np.eye(n_classes)[samples_sel.astype(int)]
    else:
        if samples_sel.ndim == 2:
            probs_samples = sigmoid(samples_sel)
        else:
            probs_samples = softmax(samples_sel, axis=-1)

    lower_pct, upper_pct = ci

    if probs_samples.ndim == 2:
        probs_mean = np.mean(probs_samples, axis=0)
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, probs_mean)
        roc_auc = auc(roc_fpr, roc_tpr)

        if fpr_grid is None:
            fpr_grid = np.linspace(0.0, 1.0, 201)
        fpr_grid = np.asarray(fpr_grid)

        s_used = probs_samples.shape[0]
        tpr_samples_grid = np.zeros((s_used, len(fpr_grid)))
        auc_samples = np.zeros(s_used)

        for i in range(s_used):
            probs = probs_samples[i]
            try:
                fpr_i, tpr_i, _ = roc_curve(y_true, probs)
                auc_samples[i] = roc_auc_score(y_true, probs)
            except ValueError:
                fpr_i = np.array([0.0, 1.0])
                tpr_i = np.array([0.0, 1.0])
                auc_samples[i] = 0.5
            tpr_samples_grid[i] = np.interp(fpr_grid, fpr_i, tpr_i)

        tpr_mean = tpr_samples_grid.mean(axis=0)
        tpr_lower = np.percentile(tpr_samples_grid, lower_pct, axis=0)
        tpr_upper = np.percentile(tpr_samples_grid, upper_pct, axis=0)
        auc_mean = float(np.mean(auc_samples))
        auc_ci = (float(np.percentile(auc_samples, lower_pct)), float(np.percentile(auc_samples, upper_pct)))

        y_pred = (probs_mean >= threshold).astype(int)
        confmat = confusion_matrix(y_true, y_pred)

        print(f"AUC (mean probs ROC): {roc_auc:.4f}")
        print(f"AUC (posterior mean): {auc_mean:.4f}, {lower_pct}/{upper_pct}% CI = ({auc_ci[0]:.4f}, {auc_ci[1]:.4f})")
        print(f"\nClassification report (threshold = {threshold}):\n")
        print(classification_report(y_true, y_pred, digits=4))

        if plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            rng = np.random.RandomState(random_state)
            to_plot = min(plot_sample_rocs, s_used)
            sample_idxs = rng.choice(s_used, size=to_plot, replace=False)
            for si in sample_idxs:
                plt.plot(fpr_grid, tpr_samples_grid[si], color="gray", alpha=0.1, linewidth=0.8)
            plt.plot(fpr_grid, tpr_mean, lw=2, label=f"Mean ROC (AUC mean={auc_mean:.3f})")
            plt.fill_between(fpr_grid, tpr_lower, tpr_upper, alpha=0.25, label=f"{lower_pct}/{upper_pct}% credible band")
            plt.plot(roc_fpr, roc_tpr, linestyle="--", lw=1.5, label=f"ROC(mean probs) AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="k")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve with posterior credible band")
            plt.legend(loc="lower right")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=[0, 1])
            disp.plot(ax=plt.gca(), cmap=None, colorbar=True)
            plt.title(f"Confusion matrix (threshold={threshold})")
            plt.tight_layout()
            plt.show()

        return {
            "problem_type": "binary",
            "y_true": y_true,
            "auc_mean": auc_mean,
            "auc_ci": auc_ci,
            "auc_samples": auc_samples,
            "fpr_grid": fpr_grid,
            "tpr_mean": tpr_mean,
            "tpr_lower": tpr_lower,
            "tpr_upper": tpr_upper,
            "tpr_samples": tpr_samples_grid,
            "probs_mean": probs_mean,
            "probs_samples": probs_samples,
            "y_pred": y_pred,
            "confusion_matrix": confmat,
            "roc_fpr": roc_fpr,
            "roc_tpr": roc_tpr,
            "roc_thresholds": roc_thresholds,
        }

    probs_mean = np.mean(probs_samples, axis=0)
    y_pred = np.argmax(probs_mean, axis=1)
    confmat = confusion_matrix(y_true, y_pred)
    print("\nMulticlass classification report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    try:
        auc_macro_ovr = roc_auc_score(y_true_bin, probs_mean, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro_ovr = np.nan

    if fpr_grid is None:
        fpr_grid = np.linspace(0.0, 1.0, 201)
    fpr_grid = np.asarray(fpr_grid)

    y_true_pos = (y_true == positive_class).astype(int)
    s_used = probs_samples.shape[0]
    tpr_samples_grid = np.zeros((s_used, len(fpr_grid)))
    auc_samples = np.zeros(s_used)

    for i in range(s_used):
        probs_i = probs_samples[i, :, positive_class]
        try:
            fpr_i, tpr_i, _ = roc_curve(y_true_pos, probs_i)
            auc_samples[i] = roc_auc_score(y_true_pos, probs_i)
        except ValueError:
            fpr_i = np.array([0.0, 1.0])
            tpr_i = np.array([0.0, 1.0])
            auc_samples[i] = 0.5
        tpr_samples_grid[i] = np.interp(fpr_grid, fpr_i, tpr_i)

    tpr_mean = tpr_samples_grid.mean(axis=0)
    tpr_lower = np.percentile(tpr_samples_grid, lower_pct, axis=0)
    tpr_upper = np.percentile(tpr_samples_grid, upper_pct, axis=0)
    auc_mean = float(np.mean(auc_samples))
    auc_ci = (float(np.percentile(auc_samples, lower_pct)), float(np.percentile(auc_samples, upper_pct)))
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true_pos, probs_mean[:, positive_class])

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        rng = np.random.RandomState(random_state)
        to_plot = min(plot_sample_rocs, s_used)
        sample_idxs = rng.choice(s_used, size=to_plot, replace=False)
        for si in sample_idxs:
            plt.plot(fpr_grid, tpr_samples_grid[si], color="gray", alpha=0.1, linewidth=0.8)
        plt.plot(fpr_grid, tpr_mean, lw=2, label=f"Class {positive_class} vs rest (AUC mean={auc_mean:.3f})")
        plt.fill_between(fpr_grid, tpr_lower, tpr_upper, alpha=0.25, label=f"{lower_pct}/{upper_pct}% credible band")
        plt.plot(roc_fpr, roc_tpr, linestyle="--", lw=1.5)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="k")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC credible band: class {positive_class} vs rest")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=np.arange(n_classes))
        disp.plot(ax=plt.gca(), cmap=None, colorbar=True)
        plt.title("Confusion matrix")
        plt.tight_layout()
        plt.show()

    return {
        "problem_type": "multiclass",
        "y_true": y_true,
        "n_classes": n_classes,
        "auc_macro_ovr": auc_macro_ovr,
        "auc_mean": auc_mean,
        "auc_ci": auc_ci,
        "auc_samples": auc_samples,
        "fpr_grid": fpr_grid,
        "tpr_mean": tpr_mean,
        "tpr_lower": tpr_lower,
        "tpr_upper": tpr_upper,
        "tpr_samples": tpr_samples_grid,
        "probs_mean": probs_mean,
        "probs_samples": probs_samples,
        "y_pred": y_pred,
        "confusion_matrix": confmat,
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "roc_thresholds": roc_thresholds,
    }


def run_svi_family(
    model_fn: Callable,
    train_arrays: dict[str, np.ndarray],
    y_train: np.ndarray,
    hidden_dim: int,
    prior_scale: float,
    num_classes: int,
    svi_steps: int,
    learning_rate: float,
    num_particles: int,
    posterior_samples: int,
    base_seed: int,
) -> dict[str, dict[str, Any]]:
    """Fit one variational posterior per feature set with SVI + AutoNormal."""
    fitted: dict[str, dict[str, Any]] = {}

    for offset, feature_name in enumerate(FEATURE_SETS):
        print(f"\nRunning SVI for family={model_fn.__name__}, features={feature_name}")
        rng_key = jax.random.PRNGKey(base_seed + offset)
        rng_fit, rng_post = jax.random.split(rng_key, 2)

        guide = AutoNormal(model_fn)
        optimizer = Adam(learning_rate)
        svi = SVI(model_fn, guide, optimizer, Trace_ELBO(num_particles=num_particles))

        svi_result = svi.run(
            rng_fit,
            svi_steps,
            jnp.array(train_arrays[feature_name]),
            jnp.array(y_train),
            hidden_dim=hidden_dim,
            prior_scale=prior_scale,
            num_classes=num_classes,
        )
        params = svi_result.params
        losses = np.asarray(svi_result.losses)

        print(f"Final ELBO loss: {losses[-1]:.4f}")
        if len(losses) > 10:
            print(f"Initial ELBO loss: {losses[0]:.4f}")

        variational_samples = guide.sample_posterior(
            rng_post,
            params,
            sample_shape=(posterior_samples,),
        )

        fitted[feature_name] = {
            "guide": guide,
            "params": params,
            "losses": losses,
            "posterior_samples": variational_samples,
        }

    return fitted

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
        probs_samples=results.get("probs_samples"),
        y_true=results.get("y_true"),
        confusion_matrix=results["confusion_matrix"],
        ypred=results["y_pred"],
        roc_fpr=results["roc_fpr"],
        roc_tpr=results["roc_tpr"],
        roc_thresholds=results["roc_thresholds"],
        prior = PRIOR_SCALE
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

    id_train, id_val, id_test = build_split_indices(features)
    print("Train rows:", len(id_train), "Val rows:", len(id_val), "Test rows:", len(id_test))
    y = np.asarray(features["SHD"])

    train_arrays: dict[str, np.ndarray] = {}
    test_arrays: dict[str, np.ndarray] = {}
    y_train = None
    y_test = None

    for feature_name, feature_columns in FEATURE_SETS.items():
        X_full = concat_feature_set(features, feature_columns)
        X_train, y_train_curr, X_test, y_test_curr = prepare_train_test_arrays(
            X_full=X_full,
            y=y,
            id_train=id_train,
            id_test=id_test,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        train_arrays[feature_name] = X_train
        test_arrays[feature_name] = X_test
        if y_train is None:
            y_train = y_train_curr
            y_test = y_test_curr

    assert y_train is not None and y_test is not None
    print("y_test shape:", y_test.shape)

    fitted_families: dict[str, dict[str, dict[str, Any]]] = {}
    for family_name, model_fn in MODEL_SPECS.items():
        fitted_families[family_name] = run_svi_family(
            model_fn=model_fn,
            train_arrays=train_arrays,
            y_train=y_train,
            hidden_dim=args.hidden_dim,
            prior_scale=args.prior_scale,
            num_classes=NUM_CLASSES,
            svi_steps=args.svi_steps,
            learning_rate=args.learning_rate,
            num_particles=args.num_particles,
            posterior_samples=args.posterior_predictive_samples,
            base_seed=args.seed + int(family_name) * 100,
        )

    for family_name, family_fits in fitted_families.items():
        for feature_name, fit in family_fits.items():
            predictive = Predictive(
                MODEL_SPECS[family_name],
                posterior_samples=fit["posterior_samples"],
                return_sites=["obs", "logits"],
            )
            pred_rng = jax.random.PRNGKey(args.seed + int(family_name) * 1000 + len(feature_name))
            pred_samples = predictive(
                pred_rng,
                jnp.array(test_arrays[feature_name]),
                None,
                hidden_dim=args.hidden_dim,
                prior_scale=args.prior_scale,
                num_classes=NUM_CLASSES,
            )
            results = evaluate_posterior_predictions(
                pred_samples=pred_samples,
                y_true=y_test,
                site="logits",
                threshold=0.5,
                plot=args.plot,
                n_posterior_samples=args.posterior_predictive_samples,
                plot_sample_rocs=min(500, args.posterior_predictive_samples),
                random_state=args.seed,
            )
            output_path = output_dir / RESULT_FILENAMES[(family_name, feature_name)]
            save_results(results, output_path)
            print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
