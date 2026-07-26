"""
Hidden-unit encoding analysis for the 21-place x 24-surface RNN task.

The script uses two files produced by the training run:

    model.pt
    evaluation_hidden_states.npz

It does not retrain or modify the RNN. The checkpoint is used only to validate
metadata such as hidden size and place/surface dimensions. The saved held-out
hidden states are used for all analyses.

For every hidden unit, four encoding models are compared:

    place-only:       h ~ place
    surface-only:     h ~ surface
    additive:         h ~ place + surface
    conjunctive:      h ~ place + surface + place:surface

The conjunctive model is implemented as a saturated mean for each observed
place-surface class. The additive model is a weighted categorical regression.

Scientific safeguards:

* Cross-validation is split by whole trials, never by timesteps.
* Every trial is held out exactly once using K-fold trial-level CV.
* Class support is determined from the training fold only.
* Additive and conjunctive models are scored on exactly the same test rows.
* Interaction evidence is tested from paired per-trial test errors.
* Benjamini-Hochberg FDR correction is applied across all hidden units.
* Unsupported and low-support conjunction classes are not treated as evidence.

Main outputs:

    unit_encoding_results.csv
    analysis_summary.json
    unit_rate_maps.npz
    model_r2_comparison.png
    interaction_delta_distribution.png
    classification_summary.png
    example_unit_maps.png
    conjunction_support.png

Example usage:

    python analyze_hidden_units.py \
        --npz evaluation_hidden_states.npz \
        --model model.pt \
        --out-dir hidden_unit_analysis

The default classification is intentionally conservative:

    unresponsive  : conjunctive held-out R^2 < 0.05
    additive      : responsive, but no reliable interaction improvement
    conjunctive   : responsive, delta R^2 >= 0.02, positive trial-level
                    improvement, and FDR q < 0.05

All thresholds are configurable from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def recursive_find_int(obj: Any, candidate_keys: Iterable[str]) -> int | None:
    """Find the first integer-valued metadata field in a nested checkpoint."""
    wanted = {str(key).lower() for key in candidate_keys}

    def visit(value: Any) -> int | None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in wanted:
                    if isinstance(child, (int, np.integer)):
                        return int(child)
                    if torch.is_tensor(child) and child.numel() == 1:
                        return int(child.item())
            for child in value.values():
                # Do not recurse into tensors/state arrays.
                if isinstance(child, (dict, list, tuple)):
                    found = visit(child)
                    if found is not None:
                        return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (dict, list, tuple)):
                    found = visit(child)
                    if found is not None:
                        return found
        return None

    return visit(obj)


def load_checkpoint(path: Path) -> Any:
    """Load a trusted local PyTorch checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_output_dim_from_checkpoint(checkpoint: Any, hidden_dim: int) -> int | None:
    """Infer readout output width from a nested state_dict when possible."""
    candidate_tensors: list[torch.Tensor] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif torch.is_tensor(value) and value.ndim == 2:
            if int(value.shape[1]) == hidden_dim:
                candidate_tensors.append(value)

    visit(checkpoint)
    if not candidate_tensors:
        return None

    # The readout is normally the widest matrix receiving hidden_dim inputs.
    return max(int(tensor.shape[0]) for tensor in candidate_tensors)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    if not np.any(finite):
        return q_values

    finite_indices = np.flatnonzero(finite)
    p = p_values[finite]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q_values[finite_indices] = restored
    return q_values


def two_sided_normal_p(z: np.ndarray) -> np.ndarray:
    """Two-sided standard-normal p-values without requiring SciPy."""
    z = np.asarray(z, dtype=np.float64)
    # numpy does not expose erfc on every installation, so vectorize math.erfc.
    return np.array(
        [math.erfc(abs(float(value)) / math.sqrt(2.0)) for value in z],
        dtype=np.float64,
    )


def one_sided_normal_p(z: np.ndarray) -> np.ndarray:
    """Upper-tail standard-normal p-values without requiring SciPy."""
    z = np.asarray(z, dtype=np.float64)
    return np.array(
        [0.5 * math.erfc(float(value) / math.sqrt(2.0)) for value in z],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Data and metadata loading
# ---------------------------------------------------------------------------


def load_inputs(
    npz_path: Path,
    model_path: Path,
    place_bins_override: int | None,
    surface_bins_override: int | None,
):
    if not npz_path.exists():
        raise FileNotFoundError(f"Hidden-state file not found: {npz_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    data = np.load(npz_path, allow_pickle=False)
    required = {"hidden_states", "valid_mask", "target_class"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(
            f"{npz_path} is missing required arrays: {sorted(missing)}"
        )

    hidden = np.asarray(data["hidden_states"])
    valid = np.asarray(data["valid_mask"]).astype(bool)
    target = np.asarray(data["target_class"])

    if hidden.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [trials, time, units], got {hidden.shape}"
        )
    if valid.shape != hidden.shape[:2]:
        raise ValueError(
            f"valid_mask shape {valid.shape} does not match {hidden.shape[:2]}"
        )
    if target.shape != hidden.shape[:2]:
        raise ValueError(
            f"target_class shape {target.shape} does not match {hidden.shape[:2]}"
        )
    if not np.issubdtype(target.dtype, np.integer):
        if np.all(np.isfinite(target)) and np.all(target == np.round(target)):
            target = target.astype(np.int64)
        else:
            raise TypeError("target_class must contain integer class indices")
    else:
        target = target.astype(np.int64, copy=False)

    # Check finiteness in trial chunks to avoid creating a second full
    # [valid_timesteps, hidden_units] copy in memory.
    for start in range(0, hidden.shape[0], 64):
        stop = min(start + 64, hidden.shape[0])
        chunk_valid = valid[start:stop]
        if np.any(chunk_valid):
            chunk_values = hidden[start:stop][chunk_valid]
            if not np.all(np.isfinite(chunk_values)):
                raise ValueError(
                    "Non-finite hidden activations were found at valid timesteps"
                )

    checkpoint = load_checkpoint(model_path)
    n_trials, n_time, n_units = hidden.shape

    checkpoint_hidden = recursive_find_int(
        checkpoint,
        ["hidden_dim", "hidden_size", "n_hidden", "num_hidden"],
    )
    if checkpoint_hidden is not None and checkpoint_hidden != n_units:
        raise ValueError(
            f"Checkpoint hidden size ({checkpoint_hidden}) does not match "
            f"hidden_states ({n_units}). The files may come from different runs."
        )

    place_bins = place_bins_override or recursive_find_int(
        checkpoint,
        ["place_bins", "n_place_bins", "num_place_bins", "place_dim"],
    )
    surface_bins = surface_bins_override or recursive_find_int(
        checkpoint,
        [
            "surface_bins",
            "gaze_bins",
            "n_surface_bins",
            "num_surface_bins",
            "n_gaze_bins",
        ],
    )

    output_dim = infer_output_dim_from_checkpoint(checkpoint, n_units)

    # Known dimensions of the current experiment are a safe fallback only when
    # the checkpoint readout confirms 504 classes or valid targets fit 504.
    valid_targets = target[valid]
    max_target_plus_one = int(valid_targets.max()) + 1

    if place_bins is None and surface_bins is None:
        inferred_total = output_dim or max_target_plus_one
        if inferred_total == 504:
            place_bins, surface_bins = 21, 24
        else:
            raise ValueError(
                "Could not infer place/surface dimensions from model.pt. "
                "Pass --place-bins and --surface-bins explicitly."
            )
    elif place_bins is None:
        inferred_total = output_dim or max_target_plus_one
        if inferred_total % int(surface_bins) != 0:
            raise ValueError("Output dimension is not divisible by surface_bins")
        place_bins = inferred_total // int(surface_bins)
    elif surface_bins is None:
        inferred_total = output_dim or max_target_plus_one
        if inferred_total % int(place_bins) != 0:
            raise ValueError("Output dimension is not divisible by place_bins")
        surface_bins = inferred_total // int(place_bins)

    place_bins = int(place_bins)
    surface_bins = int(surface_bins)
    n_classes = place_bins * surface_bins

    if valid_targets.min() < 0 or valid_targets.max() >= n_classes:
        raise ValueError(
            f"Valid target range is {valid_targets.min()}..{valid_targets.max()}, "
            f"but place_bins*surface_bins = {n_classes}."
        )
    if output_dim is not None and output_dim != n_classes:
        raise ValueError(
            f"Checkpoint readout width ({output_dim}) does not match "
            f"place_bins*surface_bins ({n_classes})."
        )

    saved_prediction_metrics = None
    if "predicted_class" in data.files:
        predicted = np.asarray(data["predicted_class"])
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted_class shape {predicted.shape} does not match target_class {target.shape}"
            )
        predicted_valid = predicted[valid].astype(np.int64, copy=False)
        true_valid = target[valid]
        exact = predicted_valid == true_valid
        true_place = true_valid // surface_bins
        true_surface = true_valid % surface_bins
        predicted_place = predicted_valid // surface_bins
        predicted_surface = predicted_valid % surface_bins
        class_support = np.bincount(true_valid, minlength=n_classes)
        class_correct = np.bincount(true_valid[exact], minlength=n_classes)
        class_accuracy = np.full(n_classes, np.nan, dtype=np.float64)
        supported_classes = class_support > 0
        class_accuracy[supported_classes] = (
            class_correct[supported_classes] / class_support[supported_classes]
        )
        saved_prediction_metrics = {
            "top1_accuracy": float(exact.mean()),
            "place_accuracy": float((predicted_place == true_place).mean()),
            "surface_accuracy": float((predicted_surface == true_surface).mean()),
            "supported_classes": int(supported_classes.sum()),
            "macro_class_accuracy": float(np.nanmean(class_accuracy)),
        }

    metadata = {
        "n_trials": n_trials,
        "n_time": n_time,
        "n_units": n_units,
        "place_bins": place_bins,
        "surface_bins": surface_bins,
        "n_classes": n_classes,
        "valid_timesteps": int(valid.sum()),
        "checkpoint_hidden_dim": checkpoint_hidden,
        "checkpoint_output_dim": output_dim,
        "saved_prediction_metrics": saved_prediction_metrics,
    }

    return hidden, valid, target, metadata


# ---------------------------------------------------------------------------
# Fold-level sufficient statistics and predictions
# ---------------------------------------------------------------------------


def aggregate_joint_statistics(
    hidden: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    trial_indices: np.ndarray,
    n_classes: int,
    chunk_trials: int,
):
    """Return per-class counts and activation sums without flattening all data."""
    n_units = hidden.shape[2]
    counts = np.zeros(n_classes, dtype=np.int64)
    sums = np.zeros((n_classes, n_units), dtype=np.float64)

    for start in range(0, len(trial_indices), chunk_trials):
        batch_trials = trial_indices[start:start + chunk_trials]
        batch_valid = valid[batch_trials]
        if not np.any(batch_valid):
            continue

        labels = target[batch_trials][batch_valid]
        values = hidden[batch_trials][batch_valid].astype(np.float64, copy=False)

        counts += np.bincount(labels, minlength=n_classes)
        np.add.at(sums, labels, values)

    return counts, sums


def build_additive_design_for_classes(
    class_ids: np.ndarray,
    place_bins: int,
    surface_bins: int,
) -> np.ndarray:
    """Reference-coded design: intercept + place[1:] + surface[1:]."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    place = class_ids // surface_bins
    surface = class_ids % surface_bins

    n_rows = len(class_ids)
    n_columns = 1 + (place_bins - 1) + (surface_bins - 1)
    design = np.zeros((n_rows, n_columns), dtype=np.float64)
    design[:, 0] = 1.0

    non_reference_place = place > 0
    place_columns = place[non_reference_place]
    design[
        np.flatnonzero(non_reference_place),
        place_columns,
    ] = 1.0

    surface_offset = 1 + (place_bins - 1)
    non_reference_surface = surface > 0
    surface_columns = surface_offset + surface[non_reference_surface] - 1
    design[
        np.flatnonzero(non_reference_surface),
        surface_columns,
    ] = 1.0

    return design


def fit_weighted_additive_model(
    class_counts: np.ndarray,
    class_sums: np.ndarray,
    supported: np.ndarray,
    place_bins: int,
    surface_bins: int,
    ridge_alpha: float,
) -> np.ndarray:
    """Fit weighted place+surface regression using class sufficient statistics."""
    class_ids = np.flatnonzero(supported)
    counts = class_counts[class_ids].astype(np.float64)
    means = class_sums[class_ids] / counts[:, None]

    design = build_additive_design_for_classes(
        class_ids,
        place_bins,
        surface_bins,
    )

    sqrt_weights = np.sqrt(counts)[:, None]
    x_weighted = design * sqrt_weights
    y_weighted = means * sqrt_weights

    gram = x_weighted.T @ x_weighted
    rhs = x_weighted.T @ y_weighted

    if ridge_alpha > 0:
        penalty = np.eye(gram.shape[0], dtype=np.float64)
        penalty[0, 0] = 0.0  # do not penalize the intercept
        gram = gram + ridge_alpha * penalty

    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs

    all_classes = np.arange(place_bins * surface_bins, dtype=np.int64)
    all_design = build_additive_design_for_classes(
        all_classes,
        place_bins,
        surface_bins,
    )
    return all_design @ coefficients


def fit_fold_predictions(
    class_counts: np.ndarray,
    class_sums: np.ndarray,
    place_bins: int,
    surface_bins: int,
    min_train_class_support: int,
    ridge_alpha: float,
):
    n_classes, n_units = class_sums.shape
    supported = class_counts >= min_train_class_support
    if not np.any(supported):
        raise RuntimeError(
            "No conjunction classes meet the training support threshold. "
            "Lower --min-train-class-support."
        )

    supported_ids = np.flatnonzero(supported)
    supported_counts = class_counts[supported_ids].astype(np.float64)
    supported_sums = class_sums[supported_ids]

    global_sum = supported_sums.sum(axis=0)
    global_count = supported_counts.sum()
    global_mean = global_sum / global_count

    additive_prediction = fit_weighted_additive_model(
        class_counts,
        class_sums,
        supported,
        place_bins,
        surface_bins,
        ridge_alpha,
    )

    # The saturated interaction model is the train mean for each supported
    # conjunction class. Unsupported classes are never evaluated, but are filled
    # with additive predictions as a safe fallback.
    full_prediction = additive_prediction.copy()
    full_prediction[supported_ids] = (
        supported_sums / supported_counts[:, None]
    )

    place_of_supported = supported_ids // surface_bins
    surface_of_supported = supported_ids % surface_bins

    place_counts = np.zeros(place_bins, dtype=np.float64)
    place_sums = np.zeros((place_bins, n_units), dtype=np.float64)
    np.add.at(place_counts, place_of_supported, supported_counts)
    np.add.at(place_sums, place_of_supported, supported_sums)

    surface_counts = np.zeros(surface_bins, dtype=np.float64)
    surface_sums = np.zeros((surface_bins, n_units), dtype=np.float64)
    np.add.at(surface_counts, surface_of_supported, supported_counts)
    np.add.at(surface_sums, surface_of_supported, supported_sums)

    place_means = np.repeat(global_mean[None, :], place_bins, axis=0)
    place_present = place_counts > 0
    place_means[place_present] = (
        place_sums[place_present] / place_counts[place_present, None]
    )

    surface_means = np.repeat(global_mean[None, :], surface_bins, axis=0)
    surface_present = surface_counts > 0
    surface_means[surface_present] = (
        surface_sums[surface_present] / surface_counts[surface_present, None]
    )

    all_classes = np.arange(n_classes, dtype=np.int64)
    place_prediction = place_means[all_classes // surface_bins]
    surface_prediction = surface_means[all_classes % surface_bins]
    baseline_prediction = np.repeat(global_mean[None, :], n_classes, axis=0)

    return {
        "supported": supported,
        "baseline": baseline_prediction,
        "place": place_prediction,
        "surface": surface_prediction,
        "additive": additive_prediction,
        "full": full_prediction,
    }


# ---------------------------------------------------------------------------
# Cross-validated scoring
# ---------------------------------------------------------------------------


def trial_folds(n_trials: int, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    if n_folds > n_trials:
        raise ValueError("--folds cannot exceed the number of trials")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_trials)
    return [np.asarray(fold, dtype=np.int64) for fold in np.array_split(shuffled, n_folds)]


def cross_validated_models(
    hidden: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    place_bins: int,
    surface_bins: int,
    n_folds: int,
    min_train_class_support: int,
    ridge_alpha: float,
    chunk_trials: int,
    seed: int,
):
    n_trials, _, n_units = hidden.shape
    n_classes = place_bins * surface_bins
    folds = trial_folds(n_trials, n_folds, seed)

    model_names = ["baseline", "place", "surface", "additive", "full"]
    sse = {
        name: np.zeros(n_units, dtype=np.float64)
        for name in model_names
    }

    y_sum = np.zeros(n_units, dtype=np.float64)
    y_square_sum = np.zeros(n_units, dtype=np.float64)
    evaluated_samples = 0
    possible_test_samples = 0

    # Positive value means the full interaction model has lower test MSE.
    trial_mse_improvement = np.full((n_trials, n_units), np.nan, dtype=np.float64)
    trial_test_sample_count = np.zeros(n_trials, dtype=np.int64)

    all_trials = np.arange(n_trials, dtype=np.int64)
    fold_reports = []

    for fold_index, test_trials in enumerate(folds, start=1):
        is_test_trial = np.zeros(n_trials, dtype=bool)
        is_test_trial[test_trials] = True
        train_trials = all_trials[~is_test_trial]

        class_counts, class_sums = aggregate_joint_statistics(
            hidden,
            valid,
            target,
            train_trials,
            n_classes,
            chunk_trials,
        )

        predictions = fit_fold_predictions(
            class_counts,
            class_sums,
            place_bins,
            surface_bins,
            min_train_class_support,
            ridge_alpha,
        )
        supported = predictions["supported"]

        fold_possible = 0
        fold_evaluated = 0

        for trial in test_trials:
            labels_all = target[trial, valid[trial]]
            values_all = hidden[trial, valid[trial]].astype(np.float64, copy=False)
            fold_possible += len(labels_all)

            keep = supported[labels_all]
            labels = labels_all[keep]
            values = values_all[keep]
            if len(labels) == 0:
                continue

            fold_evaluated += len(labels)
            trial_test_sample_count[trial] = len(labels)

            y_sum += values.sum(axis=0)
            y_square_sum += np.square(values).sum(axis=0)
            evaluated_samples += len(labels)

            trial_squared_errors = {}
            for name in model_names:
                predicted = predictions[name][labels]
                squared_error = np.square(values - predicted)
                sse[name] += squared_error.sum(axis=0)
                trial_squared_errors[name] = squared_error

            trial_mse_improvement[trial] = (
                trial_squared_errors["additive"].mean(axis=0)
                - trial_squared_errors["full"].mean(axis=0)
            )

        possible_test_samples += fold_possible
        fold_reports.append(
            {
                "fold": fold_index,
                "train_trials": int(len(train_trials)),
                "test_trials": int(len(test_trials)),
                "supported_train_classes": int(supported.sum()),
                "possible_test_timesteps": int(fold_possible),
                "evaluated_test_timesteps": int(fold_evaluated),
                "coverage": float(fold_evaluated / fold_possible) if fold_possible else float("nan"),
            }
        )

        print(
            f"Fold {fold_index}/{n_folds}: "
            f"supported classes={supported.sum()}, "
            f"test coverage={fold_evaluated}/{fold_possible} "
            f"({100.0 * fold_evaluated / fold_possible:.2f}%)"
        )

    if evaluated_samples == 0:
        raise RuntimeError("No held-out samples were eligible for evaluation")

    ss_total = y_square_sum - np.square(y_sum) / evaluated_samples
    ss_total = np.where(ss_total > 1e-12, ss_total, np.nan)

    r2 = {
        name: 1.0 - sse[name] / ss_total
        for name in model_names
    }

    return {
        "r2": r2,
        "sse": sse,
        "ss_total": ss_total,
        "trial_mse_improvement": trial_mse_improvement,
        "trial_test_sample_count": trial_test_sample_count,
        "evaluated_samples": int(evaluated_samples),
        "possible_test_samples": int(possible_test_samples),
        "coverage": float(evaluated_samples / possible_test_samples),
        "fold_reports": fold_reports,
    }


# ---------------------------------------------------------------------------
# Trial-clustered interaction test
# ---------------------------------------------------------------------------


def interaction_trial_test(trial_mse_improvement: np.ndarray):
    """
    Test whether the full model lowers held-out MSE across independent trials.

    The test statistic uses trials, not timesteps, as independent observations.
    With thousands of trials, the normal approximation to the paired mean is
    accurate and avoids the invalid label-shuffling null used by many naive
    implementations.
    """
    finite_rows = np.all(np.isfinite(trial_mse_improvement), axis=1)
    differences = trial_mse_improvement[finite_rows]
    n_trials = differences.shape[0]
    if n_trials < 2:
        raise RuntimeError("Too few evaluable trials for interaction testing")

    mean_improvement = differences.mean(axis=0)
    sd = differences.std(axis=0, ddof=1)
    se = sd / math.sqrt(n_trials)

    z = np.zeros_like(mean_improvement)
    nonzero = se > 0
    z[nonzero] = mean_improvement[nonzero] / se[nonzero]
    z[~nonzero & (mean_improvement > 0)] = np.inf
    z[~nonzero & (mean_improvement < 0)] = -np.inf

    # One-sided alternative: full-model held-out MSE is lower than additive MSE.
    p_values = one_sided_normal_p(z)
    q_values = benjamini_hochberg(p_values)

    ci_low = mean_improvement - 1.959963984540054 * se
    ci_high = mean_improvement + 1.959963984540054 * se

    return {
        "n_evaluable_trials": int(n_trials),
        "mean_mse_improvement": mean_improvement,
        "sd_trial_improvement": sd,
        "se_trial_improvement": se,
        "z_value": z,
        "p_value": p_values,
        "q_value": q_values,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


# ---------------------------------------------------------------------------
# Rate maps
# ---------------------------------------------------------------------------


def compute_rate_maps(
    hidden: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    n_classes: int,
    place_bins: int,
    surface_bins: int,
    chunk_trials: int,
):
    all_trials = np.arange(hidden.shape[0], dtype=np.int64)
    counts, sums = aggregate_joint_statistics(
        hidden,
        valid,
        target,
        all_trials,
        n_classes,
        chunk_trials,
    )

    means = np.full_like(sums, np.nan, dtype=np.float64)
    supported = counts > 0
    means[supported] = sums[supported] / counts[supported, None]

    # [class, unit] -> [unit, place, surface]
    maps = means.reshape(place_bins, surface_bins, hidden.shape[2]).transpose(2, 0, 1)
    support_map = counts.reshape(place_bins, surface_bins)
    return maps, support_map


# ---------------------------------------------------------------------------
# Classification and output
# ---------------------------------------------------------------------------


def classify_units(
    cv_results,
    interaction_test,
    min_full_r2: float,
    min_delta_r2: float,
    fdr_alpha: float,
):
    r2 = cv_results["r2"]
    delta_r2 = r2["full"] - r2["additive"]
    q_values = interaction_test["q_value"]
    mean_improvement = interaction_test["mean_mse_improvement"]

    responsive = np.isfinite(r2["full"]) & (r2["full"] >= min_full_r2)
    conjunctive = (
        responsive
        & np.isfinite(delta_r2)
        & (delta_r2 >= min_delta_r2)
        & (mean_improvement > 0)
        & np.isfinite(q_values)
        & (q_values < fdr_alpha)
    )

    labels = np.full(len(delta_r2), "unresponsive", dtype=object)
    labels[responsive] = "additive"
    labels[conjunctive] = "conjunctive"

    dominant_component = np.full(len(delta_r2), "none", dtype=object)
    linear_units = responsive & ~conjunctive
    place_better = r2["place"] > r2["surface"]
    surface_better = r2["surface"] > r2["place"]
    dominant_component[linear_units & place_better] = "place"
    dominant_component[linear_units & surface_better] = "surface"
    dominant_component[
        linear_units & ~(place_better | surface_better)
    ] = "tie"
    dominant_component[conjunctive] = "interaction"

    return labels, dominant_component, delta_r2


def write_unit_csv(
    path: Path,
    cv_results,
    interaction_test,
    labels: np.ndarray,
    dominant_component: np.ndarray,
    delta_r2: np.ndarray,
):
    r2 = cv_results["r2"]
    fieldnames = [
        "unit_id",
        "r2_baseline",
        "r2_place",
        "r2_surface",
        "r2_additive",
        "r2_conjunctive",
        "delta_r2_conjunctive_minus_additive",
        "mean_trial_mse_improvement",
        "trial_improvement_ci95_low",
        "trial_improvement_ci95_high",
        "interaction_z",
        "interaction_p",
        "interaction_q_bh",
        "label",
        "dominant_linear_component",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for unit in range(len(labels)):
            writer.writerow(
                {
                    "unit_id": unit,
                    "r2_baseline": r2["baseline"][unit],
                    "r2_place": r2["place"][unit],
                    "r2_surface": r2["surface"][unit],
                    "r2_additive": r2["additive"][unit],
                    "r2_conjunctive": r2["full"][unit],
                    "delta_r2_conjunctive_minus_additive": delta_r2[unit],
                    "mean_trial_mse_improvement": interaction_test["mean_mse_improvement"][unit],
                    "trial_improvement_ci95_low": interaction_test["ci95_low"][unit],
                    "trial_improvement_ci95_high": interaction_test["ci95_high"][unit],
                    "interaction_z": interaction_test["z_value"][unit],
                    "interaction_p": interaction_test["p_value"][unit],
                    "interaction_q_bh": interaction_test["q_value"][unit],
                    "label": labels[unit],
                    "dominant_linear_component": dominant_component[unit],
                }
            )


def make_plots(
    out_dir: Path,
    cv_results,
    interaction_test,
    labels: np.ndarray,
    delta_r2: np.ndarray,
    rate_maps: np.ndarray,
    support_map: np.ndarray,
):
    r2 = cv_results["r2"]

    # 1. Additive versus conjunctive held-out R2.
    fig, ax = plt.subplots(figsize=(7, 7))
    for label in ["unresponsive", "additive", "conjunctive"]:
        mask = labels == label
        ax.scatter(
            r2["additive"][mask],
            r2["full"][mask],
            s=24,
            alpha=0.75,
            label=f"{label} (n={int(mask.sum())})",
        )
    finite_values = np.concatenate(
        [r2["additive"][np.isfinite(r2["additive"])], r2["full"][np.isfinite(r2["full"])]]
    )
    lower = min(-0.05, float(np.percentile(finite_values, 1)))
    upper = max(0.1, float(np.percentile(finite_values, 99)))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Held-out R²: additive place + surface")
    ax.set_ylabel("Held-out R²: full place × surface")
    ax.set_title("Hidden-unit encoding model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_r2_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Interaction delta distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_r2[np.isfinite(delta_r2)], bins=50)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("ΔR² = conjunctive − additive")
    ax.set_ylabel("Number of hidden units")
    ax.set_title("Held-out interaction improvement")
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_delta_distribution.png", dpi=200)
    plt.close(fig)

    # 3. Classification summary.
    order = ["unresponsive", "additive", "conjunctive"]
    counts = [int(np.sum(labels == label)) for label in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(order, counts)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Number of hidden units")
    ax.set_title("Hidden-unit classification")
    fig.tight_layout()
    fig.savefig(out_dir / "classification_summary.png", dpi=200)
    plt.close(fig)

    # 4. Conjunction support map.
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(np.log1p(support_map), aspect="auto")
    ax.set_xlabel("Surface bin")
    ax.set_ylabel("Place bin")
    ax.set_title("Evaluation support: log(1 + count)")
    ax.set_xticks(range(support_map.shape[1]))
    ax.set_xticklabels(range(1, support_map.shape[1] + 1), fontsize=7)
    ax.set_yticks(range(support_map.shape[0]))
    ax.set_yticklabels(range(1, support_map.shape[0] + 1), fontsize=8)
    fig.colorbar(image, ax=ax, label="log(1 + support)")
    fig.tight_layout()
    fig.savefig(out_dir / "conjunction_support.png", dpi=200)
    plt.close(fig)

    # 5. Example unit maps: strongest three conjunctive and strongest three additive.
    conjunctive_units = np.flatnonzero(labels == "conjunctive")
    additive_units = np.flatnonzero(labels == "additive")

    top_conjunctive = conjunctive_units[
        np.argsort(-delta_r2[conjunctive_units])[:3]
    ] if len(conjunctive_units) else np.array([], dtype=int)

    top_additive = additive_units[
        np.argsort(-r2["additive"][additive_units])[:3]
    ] if len(additive_units) else np.array([], dtype=int)

    examples = [
        *(('conjunctive', int(unit)) for unit in top_conjunctive),
        *(('additive', int(unit)) for unit in top_additive),
    ]

    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
        if len(examples) == 1:
            axes = [axes]
        for ax, (label, unit) in zip(axes, examples):
            image = ax.imshow(rate_maps[unit], aspect="auto")
            ax.set_xlabel("Surface bin")
            ax.set_ylabel("Place bin")
            ax.set_title(
                f"{label} unit {unit}\n"
                f"R² full={r2['full'][unit]:.3f}, ΔR²={delta_r2[unit]:.3f}"
            )
            fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "example_unit_maps.png", dpi=200)
        plt.close(fig)

    # 6. Trial-level test effect versus q-value.
    q = interaction_test["q_value"]
    valid_q = np.isfinite(q) & (q > 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        interaction_test["mean_mse_improvement"][valid_q],
        -np.log10(q[valid_q]),
        s=24,
        alpha=0.75,
    )
    ax.set_xlabel("Mean held-out trial MSE improvement")
    ax.set_ylabel("−log10(BH q-value)")
    ax.set_title("Trial-clustered evidence for interaction")
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_evidence.png", dpi=200)
    plt.close(fig)


def build_summary(
    args,
    metadata,
    cv_results,
    interaction_test,
    labels,
    delta_r2,
    support_map,
):
    label_counts = {
        label: int(np.sum(labels == label))
        for label in ["unresponsive", "additive", "conjunctive"]
    }

    return {
        "input_npz": str(Path(args.npz).resolve()),
        "input_model": str(Path(args.model).resolve()),
        "metadata": metadata,
        "method": {
            "models": {
                "place": "hidden activity ~ place category",
                "surface": "hidden activity ~ surface category",
                "additive": "hidden activity ~ place + surface",
                "conjunctive": "hidden activity ~ place + surface + place:surface",
            },
            "cross_validation": f"{args.folds}-fold, split by whole trials",
            "support_rule": (
                "A test conjunction class is evaluated only when its training-fold "
                f"support is at least {args.min_train_class_support}."
            ),
            "interaction_test": (
                "One-sided paired test of per-trial held-out MSE improvement, "
                "using trials as independent units and a large-sample normal approximation."
            ),
            "multiple_comparisons": "Benjamini-Hochberg FDR across all hidden units",
        },
        "thresholds": {
            "minimum_conjunctive_r2_for_responsive": args.min_full_r2,
            "minimum_delta_r2_for_conjunctive": args.min_delta_r2,
            "fdr_alpha": args.fdr_alpha,
            "ridge_alpha_additive": args.ridge_alpha,
        },
        "held_out_evaluation": {
            "evaluated_timesteps": cv_results["evaluated_samples"],
            "possible_timesteps": cv_results["possible_test_samples"],
            "coverage": cv_results["coverage"],
            "evaluable_trials_for_interaction_test": interaction_test["n_evaluable_trials"],
            "folds": cv_results["fold_reports"],
        },
        "supported_classes_in_complete_evaluation": int(np.sum(support_map > 0)),
        "classification_counts": label_counts,
        "median_scores": {
            "r2_place": float(np.nanmedian(cv_results["r2"]["place"])),
            "r2_surface": float(np.nanmedian(cv_results["r2"]["surface"])),
            "r2_additive": float(np.nanmedian(cv_results["r2"]["additive"])),
            "r2_conjunctive": float(np.nanmedian(cv_results["r2"]["full"])),
            "delta_r2": float(np.nanmedian(delta_r2)),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trial-level additive versus conjunctive hidden-unit analysis"
    )
    parser.add_argument("--npz", required=True, help="Path to evaluation_hidden_states.npz")
    parser.add_argument("--model", required=True, help="Path to model.pt")
    parser.add_argument("--out-dir", default="hidden_unit_analysis", help="Output directory")
    parser.add_argument("--place-bins", type=int, default=None, help="Override checkpoint metadata")
    parser.add_argument("--surface-bins", type=int, default=None, help="Override checkpoint metadata")
    parser.add_argument("--folds", type=int, default=5, help="Trial-level CV folds")
    parser.add_argument(
        "--min-train-class-support",
        type=int,
        default=20,
        help="Minimum training samples required for a conjunction class",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="Small ridge penalty for additive categorical regression",
    )
    parser.add_argument(
        "--min-full-r2",
        type=float,
        default=0.05,
        help="Minimum held-out conjunctive-model R2 for responsiveness",
    )
    parser.add_argument(
        "--min-delta-r2",
        type=float,
        default=0.02,
        help="Minimum held-out R2 gain required for conjunctive classification",
    )
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--chunk-trials",
        type=int,
        default=64,
        help="Number of trials processed at once during aggregation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden, valid, target, metadata = load_inputs(
        npz_path,
        model_path,
        args.place_bins,
        args.surface_bins,
    )

    print("Loaded evaluation data")
    print(f"  trials: {metadata['n_trials']}")
    print(f"  valid timesteps: {metadata['valid_timesteps']}")
    print(f"  hidden units: {metadata['n_units']}")
    print(
        f"  classes: {metadata['place_bins']} place x "
        f"{metadata['surface_bins']} surface = {metadata['n_classes']}"
    )

    cv_results = cross_validated_models(
        hidden,
        valid,
        target,
        metadata["place_bins"],
        metadata["surface_bins"],
        args.folds,
        args.min_train_class_support,
        args.ridge_alpha,
        args.chunk_trials,
        args.seed,
    )

    interaction_test = interaction_trial_test(
        cv_results["trial_mse_improvement"]
    )

    labels, dominant_component, delta_r2 = classify_units(
        cv_results,
        interaction_test,
        args.min_full_r2,
        args.min_delta_r2,
        args.fdr_alpha,
    )

    rate_maps, support_map = compute_rate_maps(
        hidden,
        valid,
        target,
        metadata["n_classes"],
        metadata["place_bins"],
        metadata["surface_bins"],
        args.chunk_trials,
    )

    write_unit_csv(
        out_dir / "unit_encoding_results.csv",
        cv_results,
        interaction_test,
        labels,
        dominant_component,
        delta_r2,
    )

    np.savez_compressed(
        out_dir / "unit_rate_maps.npz",
        rate_maps=rate_maps.astype(np.float32),
        support=support_map.astype(np.int64),
        labels=labels.astype(str),
        r2_place=cv_results["r2"]["place"].astype(np.float32),
        r2_surface=cv_results["r2"]["surface"].astype(np.float32),
        r2_additive=cv_results["r2"]["additive"].astype(np.float32),
        r2_conjunctive=cv_results["r2"]["full"].astype(np.float32),
        delta_r2=delta_r2.astype(np.float32),
        interaction_p=interaction_test["p_value"].astype(np.float64),
        interaction_q=interaction_test["q_value"].astype(np.float64),
    )

    make_plots(
        out_dir,
        cv_results,
        interaction_test,
        labels,
        delta_r2,
        rate_maps,
        support_map,
    )

    summary = build_summary(
        args,
        metadata,
        cv_results,
        interaction_test,
        labels,
        delta_r2,
        support_map,
    )
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    counts = summary["classification_counts"]
    print("\nAnalysis complete")
    print(f"  unresponsive: {counts['unresponsive']}")
    print(f"  additive:     {counts['additive']}")
    print(f"  conjunctive:  {counts['conjunctive']}")
    print(f"  held-out coverage: {100.0 * cv_results['coverage']:.2f}%")
    print(f"  outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()