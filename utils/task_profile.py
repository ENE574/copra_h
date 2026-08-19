"""Merge task-specific mutation / physics profiles into model+train configs."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

import yaml
from easydict import EasyDict


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        elif k not in out:
            out[k] = copy.deepcopy(v)
    return out


def load_task_profile(profile_name: str, profiles_dir: Optional[str] = None) -> Tuple[EasyDict, str]:
    if profiles_dir is None:
        profiles_dir = os.path.join(os.path.dirname(__file__), "..", "config", "task_profiles")
    path = os.path.join(profiles_dir, f"{profile_name}.yml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Task profile not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return EasyDict(raw), profile_name


def apply_task_profile(
    model_args: EasyDict,
    profile_name: Optional[str] = None,
    profiles_dir: Optional[str] = None,
) -> EasyDict:
    """Merge ``task_profile`` from model_args or explicit profile_name into model + train sections."""
    name = profile_name or getattr(model_args, "task_profile", None)
    if not name:
        return model_args
    profile, _ = load_task_profile(name, profiles_dir=profiles_dir)
    merged = copy.deepcopy(dict(model_args))
    if "model" in profile:
        merged["model"] = _deep_merge(merged.get("model", {}), dict(profile.model))
    if "train" in profile:
        merged["train"] = _deep_merge(merged.get("train", {}), dict(profile.train))
    if "entity_pair" in profile:
        ep = dict(profile.entity_pair)
        merged["entity_pair"] = ep
        merged["model"] = _deep_merge(merged.get("model", {}), ep)
    return EasyDict(merged)
