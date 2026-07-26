import os
import csv
import json
import numpy as np

import RNNTrialStructures

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT_DIR = "probfix_results_conjunction_rnn_h256_u3000_eval4096_noise005"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_STEPS, MAX_STEPS = 80, 120
BATCH_SIZE = 64
TRAIN_UPDATES = 3000
EVAL_TRIALS = 4096

BIN_SIZE = 2.0
WALL_BIN_SIZE = 5.0

INPUT_DIM = 36
HIDDEN_DIM = 256
PLACE_BINS = 21
SURFACE_BINS = 24
OUTPUT_DIM = PLACE_BINS * SURFACE_BINS

LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
LOG_EVERY = 50
SEED = 1234
EVAL_NOISE_STD = 0.05


class ConjunctionRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            INPUT_DIM,
            HIDDEN_DIM,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x, return_hidden=False):
        hidden, _ = self.rnn(x)
        logits = self.fc(hidden)
        return (logits, hidden) if return_hidden else logits


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_generator():
    trialstruct = RNNTrialStructures.get_navigation_trialstruct(
        MIN_STEPS,
        MAX_STEPS,
        ["distance", "movement", "texture"],
        ["conjunction"],
    )
    return RNNTrialStructures.get_batch_generator(
        trialstruct,
        BATCH_SIZE,
        binsize=BIN_SIZE,
        binsize_wall=WALL_BIN_SIZE,
    )


def make_batch(generator, device):
    x_np, y_np, w_np = generator()

    x_np = np.asarray(x_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32)
    w_np = np.asarray(w_np, dtype=np.float32)

    if x_np.shape[0] != INPUT_DIM:
        raise RuntimeError(f"Expected {INPUT_DIM} inputs, got {x_np.shape}")
    if y_np.shape[0] != OUTPUT_DIM:
        raise RuntimeError(f"Expected {OUTPUT_DIM} classes, got {y_np.shape}")

    x = torch.from_numpy(x_np).permute(2, 1, 0).contiguous().to(device)
    y = torch.from_numpy(y_np).permute(2, 1, 0).contiguous().to(device)
    w = torch.from_numpy(w_np).permute(2, 1, 0).contiguous().to(device)

    target = y.argmax(dim=-1)
    valid = w[:, :, 0] > 0.5

    return x, y, target, valid


def check_targets(y, valid):
    values = y[valid].detach().cpu().numpy()

    if not np.all(np.isclose(values, 0.2) | np.isclose(values, 0.8)):
        raise RuntimeError("Targets contain values other than 0.2 and 0.8")

    if not np.all(np.isclose(values, 0.8).sum(axis=1) == 1):
        raise RuntimeError("Every valid timestep must contain exactly one 0.8")

    classes = values.argmax(axis=1)
    places = classes // SURFACE_BINS
    surfaces = classes % SURFACE_BINS

    if len(np.unique(places)) <= 5 and places.max() <= 4:
        raise RuntimeError(
            "Only five place indices were generated. Fix the Julia assign_bin "
            "place-index bug before training."
        )

    print(
        f"Target check passed: {len(np.unique(places))}/{PLACE_BINS} place bins, "
        f"{len(np.unique(surfaces))}/{SURFACE_BINS} surface bins"
    )


def metrics_from_logits(logits, target, valid):
    logits = logits[valid]
    target = target[valid]
    prediction = logits.argmax(dim=-1)

    top5 = logits.topk(5, dim=-1).indices
    true_place = target // SURFACE_BINS
    true_surface = target % SURFACE_BINS
    pred_place = prediction // SURFACE_BINS
    pred_surface = prediction % SURFACE_BINS

    return {
        "top1": (prediction == target).float().mean().item(),
        "top5": (top5 == target.unsqueeze(-1)).any(dim=-1).float().mean().item(),
        "place": (pred_place == true_place).float().mean().item(),
        "surface": (pred_surface == true_surface).float().mean().item(),
    }


def train(model, generator, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []

    for update in range(1, TRAIN_UPDATES + 1):
        model.train()
        x, _, target, valid = make_batch(generator, device)

        logits = model(x)
        loss = F.cross_entropy(logits[valid], target[valid])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if update == 1 or update % LOG_EVERY == 0:
            m = metrics_from_logits(logits, target, valid)
            history.append([update, loss.item(), m["top1"], m["top5"]])
            print(
                f"Update {update:4d}/{TRAIN_UPDATES} | "
                f"loss {loss.item():.4f} | "
                f"top1 {m['top1'] * 100:.2f}% | "
                f"top5 {m['top5'] * 100:.2f}%"
            )

    return np.asarray(history)


@torch.no_grad()
def evaluate(model, generator, device):
    model.eval()

    true_all, pred_all, top5_all = [], [], []
    hidden_all, input_all, valid_all = [], [], []
    target_time_all, pred_time_all = [], []

    loss_sum = 0.0
    valid_count = 0

    for _ in range(EVAL_TRIALS // BATCH_SIZE):
        x, _, target, valid = make_batch(generator, device)

        x_noisy = x.clone()
        noise = EVAL_NOISE_STD * torch.randn_like(x[:, :, 4:])
        x_noisy[:, :, 4:] = torch.where(
            valid.unsqueeze(-1),
            torch.clamp(x[:, :, 4:] + noise, 0.0, 1.0),
            x[:, :, 4:],
        )

        logits, hidden = model(x_noisy, return_hidden=True)

        valid_logits = logits[valid]
        valid_targets = target[valid]
        predictions = valid_logits.argmax(dim=-1)

        loss_sum += F.cross_entropy(
            valid_logits,
            valid_targets,
            reduction="sum",
        ).item()
        valid_count += valid_targets.numel()

        true_all.append(valid_targets.cpu().numpy())
        pred_all.append(predictions.cpu().numpy())
        top5_all.append(
            (valid_logits.topk(5, dim=-1).indices == valid_targets.unsqueeze(-1))
            .any(dim=-1)
            .cpu()
            .numpy()
        )

        hidden_all.append(hidden.cpu().numpy())
        input_all.append(x_noisy.cpu().numpy())
        valid_all.append(valid.cpu().numpy())
        target_time_all.append(target.cpu().numpy())
        pred_time_all.append(logits.argmax(dim=-1).cpu().numpy())

    true = np.concatenate(true_all)
    pred = np.concatenate(pred_all)
    top5 = np.concatenate(top5_all)
    correct = true == pred

    true_place = true // SURFACE_BINS
    true_surface = true % SURFACE_BINS
    pred_place = pred // SURFACE_BINS
    pred_surface = pred % SURFACE_BINS

    support = np.bincount(true, minlength=OUTPUT_DIM)
    correct_per_class = np.bincount(true[correct], minlength=OUTPUT_DIM)

    class_accuracy = np.full(OUTPUT_DIM, np.nan)
    class_accuracy[support > 0] = (
        correct_per_class[support > 0] / support[support > 0]
    )

    place_support = np.bincount(true_place, minlength=PLACE_BINS)
    place_correct = np.bincount(true_place[correct], minlength=PLACE_BINS)

    floor_accuracy = np.full(PLACE_BINS, np.nan)
    floor_accuracy[place_support > 0] = (
        place_correct[place_support > 0] / place_support[place_support > 0]
    )

    metrics = {
        "evaluation_trials": EVAL_TRIALS,
        "valid_timesteps": int(valid_count),
        "cross_entropy_loss": loss_sum / valid_count,
        "top1_accuracy": float(correct.mean()),
        "error_rate": float(1.0 - correct.mean()),
        "top5_accuracy": float(top5.mean()),
        "place_accuracy": float((pred_place == true_place).mean()),
        "surface_accuracy": float((pred_surface == true_surface).mean()),
        "supported_classes": int((support > 0).sum()),
        "total_classes": OUTPUT_DIM,
        "macro_class_accuracy": float(np.nanmean(class_accuracy)),
        "evaluation_noise_std": EVAL_NOISE_STD,
        "noised_input_channels": "distance_and_texture",
    }

    hidden_data = {
        "hidden_states": np.concatenate(hidden_all),
        "inputs": np.concatenate(input_all),
        "valid_mask": np.concatenate(valid_all),
        "target_class": np.concatenate(target_time_all),
        "predicted_class": np.concatenate(pred_time_all),
    }

    return metrics, support, correct_per_class, class_accuracy, floor_accuracy, hidden_data


def floor_grid(values):
    pillars = {(1, 1), (3, 1), (1, 3), (3, 3)}
    cells = [
        (x, y)
        for x in range(5)
        for y in range(5)
        if (x, y) not in pillars
    ]

    grid = np.full((5, 5), np.nan)
    for place, (x, y) in enumerate(cells):
        grid[y, x] = values[place]

    return grid, pillars


def plot_floor_accuracy(values):
    grid, pillars = floor_grid(values)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        vmin=0,
        vmax=1,
        cmap=cmap,
    )

    for x, y in pillars:
        ax.add_patch(
            plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black")
        )

    for y in range(5):
        for x in range(5):
            if np.isfinite(grid[y, x]):
                ax.text(
                    x,
                    y,
                    f"{grid[y, x] * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel("Floor x-bin")
    ax.set_ylabel("Floor y-bin")
    ax.set_title("Exact conjunction accuracy by floor bin")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "floor_accuracy.png"), dpi=200)
    plt.close(fig)

def plot_heatmap(values, title, filename, label, accuracy=False, log_support=False):
    matrix = values.reshape(PLACE_BINS, SURFACE_BINS)
    cmap = plt.cm.viridis.copy()

    fig, ax = plt.subplots(figsize=(13, 8))

    if accuracy:
        # Accuracy heatmap: NaN values shown in light gray
        cmap.set_bad("lightgray")
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap=cmap,
        )

    elif log_support:
        # Support heatmap:
        # 0 support -> white
        # 1 and above -> logarithmic color scale
        masked_matrix = np.ma.masked_equal(matrix, 0)
        cmap.set_bad("white")

        positive_values = matrix[matrix > 0]
        image = ax.imshow(
            masked_matrix,
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=1, vmax=positive_values.max()),
        )

    else:
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
        )

    ax.set_xlabel("Surface bin")
    ax.set_ylabel("Place bin")
    ax.set_title(title)
    ax.set_xticks(range(SURFACE_BINS))
    ax.set_xticklabels(range(1, SURFACE_BINS + 1), fontsize=7)
    ax.set_yticks(range(PLACE_BINS))
    ax.set_yticklabels(range(1, PLACE_BINS + 1), fontsize=8)

    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close(fig)

def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history[:, 0], history[:, 1])
    plt.xlabel("Training update")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history[:, 0], history[:, 2], label="Top-1")
    plt.plot(history[:, 0], history[:, 3], label="Top-5")
    plt.xlabel("Training update")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_accuracy.png"), dpi=200)
    plt.close()


def save_class_csv(support, correct, accuracy):
    path = os.path.join(OUT_DIR, "conjunction_class_accuracy.csv")

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_id",
                "place_bin",
                "surface_bin",
                "support",
                "correct",
                "accuracy",
                "error_rate",
            ]
        )

        for class_id in range(OUTPUT_DIM):
            value = accuracy[class_id]
            writer.writerow(
                [
                    class_id,
                    class_id // SURFACE_BINS + 1,
                    class_id % SURFACE_BINS + 1,
                    int(support[class_id]),
                    int(correct[class_id]),
                    "" if not np.isfinite(value) else float(value),
                    "" if not np.isfinite(value) else float(1.0 - value),
                ]
            )


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("Device:", device)

    generator = make_generator()

    x, y, target, valid = make_batch(generator, device)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Valid mask shape:", tuple(valid.shape))
    check_targets(y, valid)

    model = ConjunctionRNN().to(device)
    history = train(model, generator, device)

    (
        metrics,
        support,
        correct,
        class_accuracy,
        floor_accuracy,
        hidden_data,
    ) = evaluate(model, generator, device)

    plot_history(history)
    plot_floor_accuracy(floor_accuracy)

    plot_heatmap(
        class_accuracy,
        "Accuracy for each conjunction class",
        "conjunction_accuracy_heatmap.png",
        "Accuracy",
        accuracy=True,
    )

    plot_heatmap(
        support.astype(float),
        "Evaluation support for each conjunction class",
        "conjunction_support_heatmap.png",
        "Valid timesteps",
        log_support=True,
    )

    save_class_csv(support, correct, class_accuracy)

    np.savetxt(
        os.path.join(OUT_DIR, "training_history.csv"),
        history,
        delimiter=",",
        header="update,loss,top1,top5",
        comments="",
    )

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)

    np.savez_compressed(
        os.path.join(OUT_DIR, "evaluation_hidden_states.npz"),
        **hidden_data,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "place_bins": PLACE_BINS,
            "surface_bins": SURFACE_BINS,
            "min_steps": MIN_STEPS,
            "max_steps": MAX_STEPS,
            "binsize": BIN_SIZE,
            "binsize_wall": WALL_BIN_SIZE,
        },
        os.path.join(OUT_DIR, "model.pt"),
    )

    print("\nFinal evaluation")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print("\nResults saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
