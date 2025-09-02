import os
import argparse
import torch
import optuna
import numpy as np

from T_main import parse_args as parse_main_args, update_config_from_args
from T_dataset import ProteinDataset, custom_collate_fn
from torch.utils.data import DataLoader

# import training + validate from your training module
from T_all_train import train, validate, ProteinLoss

# import model constructors used in T_main
from T_Gnn_Block import ProteinEGNN

import Traj_preprocess as tj  # adjust if module name differs

# from T_IPA_Block import ProteinIPA  # uncomment if testing IPA or combined


def build_data_and_model(config):
    # run your existing preprocessing (T_main does this). If T_main uses tj.traj_preprocess, call it.
    # Here assume T_main.traj_preprocess is available as tj.traj_preprocess

    chain_feats = tj.traj_preprocess(config, config.top_Name, config.p_Name)

    full_dataset = ProteinDataset(chain_feats)

    # split small for speed; you can keep original split logic
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # build model according to config.name_model (here we demonstrate EGNN)
    input_dim = full_dataset.atom_feat.shape[-1]
    edge_dim = full_dataset.edge_attr.shape[-1]
    valid_atom = int(full_dataset.atom_mask.sum(dim=1)[0])

    if config.name_model == "egnn":
        model = ProteinEGNN(
            node_dim=input_dim, edge_dim=edge_dim, valid_atom=valid_atom, config=config
        )
    else:
        raise NotImplementedError(
            "Only eg nn branch implemented in this quick tuner. Extend as needed."
        )

    device = config.device
    model = model.to(device)
    return train_loader, val_loader, model


def objective(trial, base_args):
    # build config
    from Config import Config

    config = Config()

    # Create a Namespace compatible with T_main.parse_args expected fields.
    # Do NOT call parse_main_args() to avoid re-parsing sys.argv (which contains optuna args).
    main_defaults = dict(
        pdb_name=getattr(base_args, "pdb_name", "2ala"),
        top_name=getattr(base_args, "top_name", "2ala"),
        traj_name=getattr(base_args, "traj_name", "prod1"),
        batch_size=getattr(base_args, "batch_size", config.batch_size),
        stage1_epochs=getattr(base_args, "stage1_epochs", config.stage1_epochs),
        stage2_epochs=getattr(base_args, "stage2_epochs", config.stage2_epochs),
        lr=getattr(base_args, "lr", config.lr),
        model=getattr(base_args, "model", "egnn"),
        dropout=getattr(base_args, "dropout", config.dropout),
        ipa_heads=getattr(base_args, "ipa_heads", 4),
        scalar_key_dim=getattr(base_args, "scalar_key_dim", 4),
        point_key_dim=getattr(base_args, "point_key_dim", 16),
        dim=getattr(base_args, "dim", config.dim),
        edge_dim=getattr(base_args, "edge_dim", config.edge_dim),
        depth=getattr(base_args, "depth", config.depth),
        pool=getattr(base_args, "pool", config.m_pool_method),
        d_model=getattr(base_args, "d_model", config.d_model),
        d_state=getattr(base_args, "d_state", config.d_state),
        d_conv=getattr(base_args, "d_conv", config.d_conv),
        n_layers=getattr(base_args, "n_layers", config.n_layers),
        use_cache=getattr(base_args, "use_cache", False),
        ver=getattr(base_args, "ver", config.ver),
        seed=getattr(base_args, "seed", 42),
    )
    main_args = argparse.Namespace(**main_defaults)
    # update config using synthesized main_args
    config = update_config_from_args(config, main_args)
    # For tuning we usually want short runs: ensure epochs == stage1_epochs
    config.stage1_epochs = int(
        getattr(base_args, "stage1_epochs", config.stage1_epochs)
    )
    config.stage2_epochs = int(
        getattr(base_args, "stage2_epochs", getattr(config, "stage2_epochs", 0))
    )
    config.epochs = config.stage1_epochs

    # search space
    config.lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    config.batch_size = int(trial.suggest_categorical("batch_size", [1, 2, 4, 8]))
    config.dropout = float(trial.suggest_uniform("dropout", 0.1, 0.6))
    config.dim = int(trial.suggest_categorical("dim", [64, 128, 256]))
    config.depth = int(trial.suggest_int("depth", 2, 6))
    config.weight_decay_main = float(
        trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    )

    # build data + model
    train_loader, val_loader, model = build_data_and_model(config)

    # set deterministic seed per trial for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # call training (short runs)
    model = train(model, train_loader, val_loader, config)

    # after train returns, evaluate validation loss
    criterion = ProteinLoss()
    val_loss, val_coords, val_recon = validate(
        model, val_loader, criterion, config.epochs - 1, config
    )

    # free cuda memory between trials
    torch.cuda.empty_cache()

    trial.report(val_loss, step=0)
    # optionally prune
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="egnn_tuning")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument(
        "--stage1_epochs", type=int, default=8, help="short epochs per trial"
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=0,
        help="stage2 epochs (optional, default 0)",
    )
    # base args forwarded to update_config_from_args; you can add more as needed
    parser.add_argument("--model", type=str, default="egnn")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--pdb_name", type=str, default="2ala")
    parser.add_argument("--top_name", type=str, default="2ala")
    parser.add_argument("--traj_name", type=str, default="prod1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    def wrapped_obj(trial):
        return objective(trial, args)

    study.optimize(wrapped_obj, n_trials=args.n_trials, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial
    print(trial.params)
    print("Best value:", trial.value)


if __name__ == "__main__":
    main()
