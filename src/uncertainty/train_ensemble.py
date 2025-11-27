import argparse
import shutil
import subprocess
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_models", type=int, default=5)
    args = parser.parse_args()

    save_root = "experiments/results/uncertainty/ensemble_lstm"
    os.makedirs(save_root, exist_ok=True)

    print(f"\n✅ Training {args.num_models} independent LSTM models via subprocess\n")

    for i in range(args.num_models):
        print(f"\n🚀 Training ensemble member {i+1}/{args.num_models}")

        model_dir = os.path.join(save_root, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)

        # Copy config so each run is fully reproducible
        local_config = os.path.join(model_dir, "config.yaml")
        shutil.copy(args.config, local_config)

        cmd = [
            "python3",
            "-m",
            "src.training.train_baseline",
            "--config",
            local_config,
        ]

        subprocess.run(cmd, check=True)

        print(f"✅ Model {i+1} saved to {model_dir}")

    print("\n✅ All ensemble members trained successfully.\n")


if __name__ == "__main__":
    main()
