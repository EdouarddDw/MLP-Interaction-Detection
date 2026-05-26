import csv
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

import argparse
import sys
import datetime
import uuid
import platform
import subprocess

from multilayer_perceptron import MLP
from synth import get_data, functions
from config import EXPERIMENTS
from missing_exp import MISSING

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


# basic parameters: --------------------------
set_seed = 42
l2_const_set = 1e-4 
dropout_rate = 0.2
learning_rate_set = 0.01
n_epochs_set = 100
# --------------------------------------------


# architecture of the model
base_parameters = {
    "num_features": 10,
    "hidden_units": [64, 64],
    "use_main_effect_nets": False,
    "main_effect_net_units": [10, 10, 10],
}

"""need to add:
    - dropout logic [X]
    - make optimizer and l2 work with config.py [X]
    - add data_loader [X]
    - add snapshot logic and make it save properly in neat file for each function and experiment [X]
    - just the double loop to train for every function and every experiment [X]
"""


def evaluate(net, data_loader, criterion, device):
    net.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.detach().cpu())
    return torch.stack(losses).mean()



def train(
    net,
    data_loaders,
    criterion=None,
    nepochs=n_epochs_set,
    verbose=True,
    L2=False,
    l1_const=1e-4,
    learning_rate=learning_rate_set,
    opt_func="adam",
    snapshots=True,
    snap_every=5,
    snapshot_root="snapshots",
    function_name="unknown_function",
    experiment_name="unknown_experiment",
    experiment_settings=None,
    run_id=None,
    num_samples=None,
    hidden_units=None,
    device=device,
):
    if criterion is None:
        criterion = nn.MSELoss(reduction="mean")

    if L2:
        l2_const = l2_const_set
    else:
        l2_const = 0.0

    opt_func = opt_func.lower()
    if opt_func == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=l2_const)
    elif opt_func == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_const)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_func}")

    net.to(device)
    best_loss = float("inf")
    best_state_dict = None
    best_epoch = None
    best_snapshot = None

    if run_id:
        snapshot_dir = Path(snapshot_root) / function_name / experiment_name / run_id
    else:
        snapshot_dir = Path(snapshot_root) / function_name / experiment_name
    if snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # try to collect git SHA for provenance
        git_sha = None
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_sha = None

        # write run_info.json with provenance for this run
        run_info = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "function": function_name,
            "experiment": experiment_name,
            "num_samples": num_samples,
            "hidden_units": hidden_units,
            "experiment_settings": experiment_settings or {},
            "seed": set_seed,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_sha": git_sha,
            "cmdline": sys.argv,
        }
        try:
            with (snapshot_dir / "run_info.json").open("w") as f:
                json.dump(run_info, f, sort_keys=True, indent=2)
        except Exception:
            # non-fatal; continue without failing training if metadata cannot be written
            pass

    metrics_file = snapshot_dir / "metrics.csv"
    experiment_settings = experiment_settings or {}
    experiment_settings_str = json.dumps(experiment_settings, sort_keys=True)

    if snapshots and not metrics_file.exists():
        with metrics_file.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["epoch", "train_loss", "val_loss", "experiment_settings"],
            )
            writer.writeheader()

    for epoch in range(nepochs):
        net.train()
        running_loss = 0.0
        run_count = 0

        for inputs, labels in data_loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            reg_loss = 0.0
            for name, param in net.named_parameters():
                if "interaction_mlp" in name and "weight" in name:
                    reg_loss = reg_loss + torch.sum(torch.abs(param))

            total_loss = loss + reg_loss * l1_const
            total_loss.backward()
            optimizer.step()

            running_loss += loss.item()
            run_count += 1

        key = "val" if "val" in data_loaders else "train"
        val_loss = evaluate(net, data_loaders[key], criterion, device)

        if snapshots and ((epoch + 1) <= 20 or (epoch + 1) % snap_every == 0):
            snapshot_file = snapshot_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "function": function_name,
                    "experiment": experiment_name,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": running_loss / max(run_count, 1),
                    "val_loss": val_loss.item(),
                },
                snapshot_file,
            )

        if snapshots:
            with metrics_file.open("a", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["epoch", "train_loss", "val_loss", "experiment_settings"],
                )
                writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "train_loss": running_loss / max(run_count, 1),
                        "val_loss": val_loss.item(),
                        "experiment_settings": experiment_settings_str,
                    }
                )

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_epoch = epoch + 1
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            best_snapshot = {
                "epoch": best_epoch,
                "function": function_name,
                "experiment": experiment_name,
                "model_state_dict": best_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": running_loss / max(run_count, 1),
                "val_loss": val_loss.item(),
            }

        if verbose and epoch % 2 == 0:
            print(
                "[epoch %d, total %d] train loss: %.4f, val loss: %.4f"
                % (epoch + 1, nepochs, running_loss / run_count, val_loss.item())
            )

    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    if snapshots and best_snapshot is not None:
        best_snapshot_file = snapshot_dir / f"best_epoch_{best_epoch:04d}.pt"
        torch.save(best_snapshot, best_snapshot_file)

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"

    test_loss = evaluate(net, data_loaders[key], criterion, device).item()

    if verbose:
        print("Finished Training. Test loss:", test_loss)

    return net, test_loss



def make_data_loaders(function, num_samples, noise, seed, batch_size=64, val_ratio=0.2):
    X, y, gt = get_data(function, num_samples=num_samples, noise=noise, seed=seed)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    n_total = len(dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }

    return data_loaders, gt



def run_experiments():
    torch.manual_seed(set_seed)

def _parse_hidden_units(hidden_units_str):
    """
    Parse hidden_units input accepting either "[64,64]" or "64,64" and
    return a list of positive integers. Raise ValueError on invalid input.
    """
    if hidden_units_str is None:
        return None
    s = hidden_units_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if s == "":
        raise ValueError("empty hidden_units string")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("no hidden unit values found")
    units = []
    for p in parts:
        try:
            v = int(p)
        except Exception:
            raise ValueError(f"invalid integer in hidden_units: '{p}'")
        if v <= 0:
            raise ValueError("hidden unit sizes must be positive integers")
        units.append(v)
    return units


def run_experiments(num_samples=30000, hidden_units=None, run_id=None,
                    weight_decay_only=False, l2_const=None):
    torch.manual_seed(set_seed)

    if l2_const is not None:
        import train as _train_module
        _train_module.l2_const_set = l2_const

    # create a run id (timestamp + short uuid) if not provided
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]

    # If caller didn't provide hidden_units, use base defaults
    if hidden_units is None:
        hidden_units = base_parameters.get("hidden_units", [64, 64])

    experiments_to_run = EXPERIMENTS
    if weight_decay_only:
        experiments_to_run = [e for e in EXPERIMENTS if e.get("weight_decay", False)]

    sep = "═" * 56
    n_funcs = len(functions)
    n_total_exp = len(EXPERIMENTS)
    n_run_exp = len(experiments_to_run)
    exp_str = (
        f"{n_run_exp} / {n_total_exp}  (weight-decay-only filter active)"
        if weight_decay_only
        else str(n_run_exp)
    )
    l2_str = (
        f"{l2_const}  (overridden from default {l2_const_set:.2e})"
        if l2_const is not None
        else f"default ({l2_const_set:.2e})"
    )
    print(sep)
    print("Training run")
    print(f"Functions:      {n_funcs}")
    print(f"Experiments:    {exp_str}")
    print(f"L2 constant:    {l2_str}")
    print(f"Hidden units:   {hidden_units}")
    print(f"Snapshots root: snapshots/")
    print(f"Run ID:         {run_id}")
    print(sep)

    for f in functions:
        for e in experiments_to_run:
            function_name = f.__name__
            experiment_name = e.get("name", "unnamed_experiment")
            print(f"Running function={function_name} with experiment={experiment_name}")

            data_loaders, gt = make_data_loaders(
                function=f,
                num_samples=num_samples,
                noise=e["noise"],
                seed=set_seed,
                batch_size=64,
                val_ratio=0.2,
            )

            model_parameters = {
                **base_parameters,
                "hidden_units": hidden_units,
                "dropout": e.get("dropout", dropout_rate),
            }

            net = MLP(**model_parameters).to(device)

            train(
                net=net,
                data_loaders=data_loaders,
                verbose=True,
                L2=e.get("weight_decay", False),
                opt_func=e.get("optimizer", "adam"),
                snapshots=True,
                snap_every=1,
                function_name=function_name,
                experiment_name=experiment_name,
                run_id=run_id,
                num_samples=num_samples,
                hidden_units=hidden_units,
                experiment_settings={
                    **base_parameters,
                    **e,
                },
                learning_rate=e.get("learning_rate", learning_rate_set),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30000,
        help="Number of samples to generate for training and evaluation (default: 30000)",
    )
    parser.add_argument(
        "--hidden_units",
        type=str,
        default=None,
        help="Hidden layer sizes as comma-separated ints or bracketed list, e.g. '64,64' or '[64,64]'",
    )

    parser.add_argument(
        "--weight-decay-only",
        action="store_true",
        help="Only run experiments where weight_decay=True. Skips all non-weight-decay experiments.",
    )
    parser.add_argument(
        "--l2-const",
        type=float,
        default=None,
        help="Override the L2 regularization constant (default: uses l2_const_set=1e-4 from source). "
             "Recommended values: 1e-4 (weak), 1e-3 (moderate), 1e-2 (strong).",
    )

    args = parser.parse_args()

    try:
        parsed_hidden = _parse_hidden_units(args.hidden_units)
    except ValueError as exc:
        print(f"Error parsing --hidden_units: {exc}", file=sys.stderr)
        sys.exit(1)

    run_experiments(
        num_samples=args.num_samples,
        hidden_units=parsed_hidden,
        weight_decay_only=args.weight_decay_only,
        l2_const=args.l2_const,
    )
