"""Provenance checks applied to every checkpoint that enters an evaluation.

Each member is a dict with keys:
    dir, sha256, seed, config (dict), data_state (dict or None for legacy),
    state_shapes ({param name: shape tuple})
"""

import hashlib

from src.data_loading.darus_dataset import OUTPUT_COLUMNS

# data_state fields that must agree across all members of one evaluation
DATA_KEYS = ("base_path", "history", "horizon", "stride", "features", "targets",
             "split_files", "split_hash", "x_scaler", "y_scaler")


class ProvenanceError(RuntimeError):
    pass


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def member_record(run_dir, ckpt_path, ckpt, config):
    return {
        "dir": run_dir,
        "sha256": file_sha256(ckpt_path),
        "seed": ckpt.get("seed"),
        "config": config,
        "data_state": ckpt.get("data_state"),
        "state_shapes": {k: tuple(v.shape) for k, v in ckpt["model_state"].items()},
        "best_epoch": ckpt.get("best_epoch"),
        "git": ckpt.get("git"),
    }


def validate_members(members, expected_n=None, method="ensemble"):
    """
    Raise ProvenanceError unless every member is compatible with the first
    and the member set is what was intended. Returns a summary dict.
    """
    errors = []
    n = len(members)
    if expected_n is not None and n != expected_n:
        errors.append(f"expected {expected_n} members, found {n}")
    if method == "ensemble" and n < 2:
        errors.append("an ensemble needs at least 2 members")

    legacy = [m["data_state"] is None for m in members]
    if any(legacy) and not all(legacy):
        # cannot compare data states meaningfully; stop here
        raise ProvenanceError("mix of legacy (no provenance) and provenance-bearing checkpoints")

    shas = [m["sha256"] for m in members]
    if len(set(shas)) != n:
        errors.append("duplicate checkpoint files (identical sha256) among members")
    dirs = [m["dir"] for m in members]
    if len(set(dirs)) != n:
        errors.append("duplicate member directories")

    ref = members[0]
    for m in members:
        tag = m["dir"]
        if m["config"].get("model") != ref["config"].get("model"):
            errors.append(f"{tag}: model config differs from {ref['dir']}")
        if m["state_shapes"] != ref["state_shapes"]:
            errors.append(f"{tag}: parameter names/shapes differ from {ref['dir']}")
        ds = m["data_state"]
        if ds is None:
            continue
        if list(ds.get("targets", [])) != list(OUTPUT_COLUMNS):
            errors.append(f"{tag}: target order {ds.get('targets')} != {OUTPUT_COLUMNS}")
        for k in DATA_KEYS:
            if ds.get(k) != ref["data_state"].get(k):
                errors.append(f"{tag}: data_state['{k}'] differs from {ref['dir']}")

    if not all(legacy):
        seeds = [m["seed"] for m in members]
        if any(s is None for s in seeds) and method != "point":
            errors.append("member without a recorded seed")
        elif len(set(seeds)) != n:
            errors.append(f"duplicate seeds among members: {seeds}")

    if errors:
        raise ProvenanceError("; ".join(errors))
    return {
        "n_members": n,
        "legacy": all(legacy),
        "seeds": [m["seed"] for m in members],
        "sha256": shas,
        "split_hash": None if all(legacy) else ref["data_state"]["split_hash"],
    }
