import io as pyio
import torch
import pytest

def _pick_reader(io_mod):
    if io_mod is None:
        return None
    # using function names
    for fname in ["read_xml", "load_xml_ecg", "parse_xml", "read_ecg_xml"]:
        if hasattr(io_mod, fname):
            return getattr(io_mod, fname)
    return None

@pytest.mark.parametrize("length_min", [200])  # ensure we read non-trivial sequences
def test_xml_reader_returns_8xL(io_mod, canonical_lead_order, length_min):
    reader = _pick_reader(io_mod)
    if reader is None:
        pytest.skip("No XML reader found in ecg_io.*")
    # expect to accept a file path or file-like; skip if path missing.
    sample_path = "tests/data/sample_8lead.xml"
    try:
        arr_meta = reader(sample_path)
    except FileNotFoundError:
        pytest.skip("tests/data/sample_8lead.xml not present; add a tiny de-identified example.")
    except TypeError:
        # Some readers return (array, meta), others just array
        pass

    if isinstance(arr_meta, tuple):
        arr, meta = arr_meta
    else:
        arr, meta = arr_meta, {}

    x = torch.as_tensor(arr)
    assert x.ndim == 2 and x.shape[0] == 8, "Reader must return (8, L) tensor/ndarray"
    assert x.shape[1] >= length_min, "ECG length too short for a valid sample"
    assert x.dtype in (torch.float32, torch.float64)
    assert torch.isfinite(x).all()

    lead_order = meta.get("lead_order")
    if lead_order is not None:
        assert list(lead_order) == canonical_lead_order, "Lead order mismatch with canonical contract"

def _pick_normalizer(io_mod):
    if io_mod is None:
        return None
    for fname in ["normalize", "zscore_per_lead", "standardize_per_lead"]:
        if hasattr(io_mod, fname):
            return getattr(io_mod, fname)
    return None

def test_normalization_contract(io_mod, synthetic_ecg_8xL):
    norm = _pick_normalizer(io_mod)
    if norm is None:
        pytest.skip("No normalization function found (normalize/zscore_per_lead/standardize_per_lead).")
    x = synthetic_ecg_8xL.clone()
    y = norm(x)
    y = torch.as_tensor(y)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.isfinite(y).all()
    # If z-scored per lead, mean≈0, std≈1 (tolerances allow noise)
    means = y.mean(dim=1)
    stds = y.std(dim=1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=0.15)
    assert torch.all(stds > 0.5) and torch.all(stds < 1.5)
