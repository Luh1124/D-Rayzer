"""Merge multiple YAML config files (later files override earlier, deep merge for dicts)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, MutableMapping, Sequence, Tuple, Union

import yaml

# Ordered stack for ``training.train`` / Gradio when no ``--config`` is passed.
DEFAULT_ERAYZER_CONFIG_REL_PATHS: Tuple[str, ...] = (
    "config/model/default.yaml",
    "config/data/default.yaml",
    "config/optimizer/adamw.yaml",
    "config/train/default.yaml",
)


def default_erayzer_config_paths(repo_root: str) -> List[str]:
    """Absolute paths to the default layered config (model → data → optimizer → train)."""
    root = os.path.abspath(repo_root)
    return [os.path.join(root, rel) for rel in DEFAULT_ERAYZER_CONFIG_REL_PATHS]


def deep_merge(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``. Lists and scalars are replaced."""
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if isinstance(data, dict) else {}


def load_merged_dict(paths: Sequence[str]) -> dict:
    """Load YAML paths in order; each file deep-merges on top of the previous."""
    if not paths:
        return {}
    merged: dict = {}
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"config file not found: {p}")
        merged = deep_merge(merged, load_yaml_dict(p))
    return merged


def load_merged_edict(paths: Sequence[str]):
    from easydict import EasyDict as edict

    return edict(load_merged_dict(paths))


def normalize_config_paths(config_arg: Union[None, str, Sequence[str]]) -> List[str]:
    """
    Accept a single path string or a non-empty sequence of paths (e.g. from argparse append).
    """
    if config_arg is None:
        return []
    if isinstance(config_arg, str):
        return [config_arg]
    return [p for p in config_arg if p]
