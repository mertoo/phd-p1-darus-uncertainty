"""Legacy entry point, kept so existing commands keep working.

Delegates to src.training.train, which writes checkpoints to
<logging.save_dir>/<logging.run_name>/ (the old version wrote best_model.pt
next to the config file, so configs sharing a directory overwrote each other)
and requires an explicit seed (training.seed or --seed).
"""

from src.training.train import main

if __name__ == "__main__":
    main()
