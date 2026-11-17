"""Train N ensemble members with distinct, recorded seeds.

    python -m src.uncertainty.train_ensemble --config CONFIG --num_models 5 \
        --base_seed 1000 [--run_prefix ensemble_lstm]

Member i is trained with seed base_seed + i into
<logging.save_dir>/<run_prefix>/model_<i>/ via src.training.train.
"""

import argparse
import os

import yaml

from src.training.train import resolve_config, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--run_prefix", type=str, default=None)
    parser.add_argument("--member", type=int, default=None,
                        help="Train only this member index (for SLURM arrays).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    prefix = args.run_prefix or config["logging"]["run_name"]

    members = [args.member] if args.member is not None else range(args.num_models)
    for i in members:
        run_name = os.path.join(prefix, f"model_{i}")
        cfg = resolve_config(config, seed=args.base_seed + i, run_name=run_name)
        print(f"Training ensemble member {i} (seed {args.base_seed + i}) -> {run_name}")
        train(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
