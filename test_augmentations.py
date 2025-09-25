import torch
import pytest

def _find_aug(aug_mod, *names):
    if aug_mod is None:
        return None
    for n in names:
        if hasattr(aug_mod, n):
            return getattr(aug_mod, n)
    return None

def test_noise_preserves_shape_dtype_and_finiteness(aug_mod, synthetic_ecg_8xL):
    add_noise = _find_aug(aug_mod, "add_noise", "gaussian_noise", "jitter")
    if add_noise is None:
        pytest.skip("No noise augmentation found (add_noise/gaussian_noise/jitter).")
    y = add_noise(synthetic_ecg_8xL, sigma=0.02, seed=123)
    y = torch.as_tensor(y)
    assert y.shape == synthetic_ecg_8xL.shape
    assert y.dtype == synthetic_ecg_8xL.dtype
    assert torch.isfinite(y).all()

def test_time_shift_is_lead_synchronous_when_requested(aug_mod, synthetic_ecg_8xL):
    time_shift = _find_aug(aug_mod, "time_shift", "shift_time")
    if time_shift is None:
        pytest.skip("No time-shift augmentation found (time_shift/shift_time).")
    y_sync = time_shift(synthetic_ecg_8xL, max_shift=10, seed=0, synchronous=True)
    # If synchronous=True, all leads should be shifted equally -> exact equality across leads.
    assert torch.allclose(y_sync[0], y_sync[1]), "Synchronous shift should be identical across leads"

def test_augmentations_are_deterministic_under_seed(aug_mod, synthetic_ecg_8xL):
    time_shift = _find_aug(aug_mod, "time_shift", "shift_time")
    add_noise = _find_aug(aug_mod, "add_noise", "gaussian_noise", "jitter")
    if time_shift is None or add_noise is None:
        pytest.skip("Missing augmentations for determinism check.")
    a1 = time_shift(synthetic_ecg_8xL, max_shift=7, seed=42, synchronous=False)
    a2 = time_shift(synthetic_ecg_8xL, max_shift=7, seed=42, synchronous=False)
    n1 = add_noise(synthetic_ecg_8xL, sigma=0.01, seed=99)
    n2 = add_noise(synthetic_ecg_8xL, sigma=0.01, seed=99)
    assert torch.equal(torch.as_tensor(a1), torch.as_tensor(a2))
    assert torch.equal(torch.as_tensor(n1), torch.as_tensor(n2))
