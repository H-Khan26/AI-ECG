import os
import re
import inspect
import numpy as np
import pytest
import torch
import importlib

# Global seeds for reproducibility
@pytest.fixture(autouse=True, scope="session")
def _set_seeds():
    np.random.seed(42)
    torch.manual_seed(42)

# Synthetic ECG fixtures (8 leads)
@pytest.fixture
def synthetic_ecg_8xL():
    leads, length = 8, 2000
    t = np.linspace(0, 2*np.pi, length, endpoint=False)
    base = np.sin(5*t)[None, :]                         # (1, L)
    noise = 0.01 * np.random.randn(leads, length)       # (8, L)
    arr = (base + noise).astype(np.float32)
    return torch.from_numpy(arr)                        # (8, L) float32

@pytest.fixture
def tiny_batch():
    B, L = 12, 256
    x = torch.randn(B, 8, L, dtype=torch.float32)
    y = torch.randint(0, 2, (B,), dtype=torch.long)
    return x, y

# Helpers to adapt to module naming
def _maybe_import(modname):
    try:
        return importlib.import_module(modname)
    except Exception:
        return None

@pytest.fixture(scope="session")
def io_mod():
    return _maybe_import("ecg_io")

@pytest.fixture(scope="session")
def aug_mod():
    return _maybe_import("ecg_augmentations")

@pytest.fixture(scope="session")
def features_mod():
    return _maybe_import("ecg_features")

@pytest.fixture(scope="session")
def models_mod():
    return _maybe_import("models")

def _candidate_model_classes(models_mod):
    if models_mod is None:
        return []
    out = []
    for name, obj in vars(models_mod).items():
        if inspect.isclass(obj) and hasattr(obj, "__mro__"):
            try:
                import torch.nn as nn
                if issubclass(obj, nn.Module):
                    out.append((name, obj))
            except Exception:
                pass
    return out

def _can_construct_with_args(cls, **kwargs):
    try:
        sig = inspect.signature(cls)
        params = set(sig.parameters.keys())
        # Accept common arg name variants for leads and classes.
        lead_keys = {"in_leads", "in_channels", "num_leads", "channels"}
        class_keys = {"num_classes", "out_dim", "classes"}
        kw = {}
        # prefer an 'in_leads' like key
        for k in lead_keys:
            if k in params:
                kw[k] = kwargs.get("in_leads", 8)
                break
        for k in class_keys:
            if k in params:
                kw[k] = kwargs.get("num_classes", 2)
                break
        # Allow init with no args
        cls(**kw)
        return True
    except Exception:
        return False

@pytest.fixture(scope="session")
def discovered_model_cls(models_mod):
    for name, cls in _candidate_model_classes(models_mod):
        if _can_construct_with_args(cls, in_leads=8, num_classes=2):
            return cls
    return None

# Canonical 8-lead order
@pytest.fixture(scope="session")
def canonical_lead_order():
    # Common reduced set: limb (I, II) + precordials V1-V6
    return ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
