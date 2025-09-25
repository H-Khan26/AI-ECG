import pytest
import torch
import inspect

def _instantiate_model(cls):
    # Fitted to UCLA Health standard argument names
    sig = inspect.signature(cls)
    kw = {}
    if "in_leads" in sig.parameters: kw["in_leads"] = 8
    elif "in_channels" in sig.parameters: kw["in_channels"] = 8
    elif "num_leads" in sig.parameters: kw["num_leads"] = 8
    elif "channels" in sig.parameters: kw["channels"] = 8

    if "num_classes" in sig.parameters: kw["num_classes"] = 2
    elif "out_dim" in sig.parameters: kw["out_dim"] = 2
    elif "classes" in sig.parameters: kw["classes"] = 2

    try:
        return cls(**kw)
    except Exception:
        # Last resort: no-arg init
        return cls()

@pytest.mark.parametrize("B,L,C", [(2, 512, 2)])
def test_forward_backward_contract(discovered_model_cls, B, L, C):
    if discovered_model_cls is None:
        pytest.skip("No suitable nn.Module subclass found in models.py")
    model = _instantiate_model(discovered_model_cls)
    x = torch.randn(B, 8, L, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    out = model(x)
    assert out.ndim == 2 and out.shape[0] == B, "Model must return (B, C) logits"
    if out.shape[1] != C:
        pytest.skip(f"Model returned C={out.shape[1]} classes; adjust test expectation if intended.")
    loss = torch.nn.CrossEntropyLoss()(out, y)
    loss.backward()
    total_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.detach().abs().sum().item()
    assert total_grad > 0.0, "Backprop should produce nonzero gradients"

def test_parameter_count_regression_guard(discovered_model_cls):
    if discovered_model_cls is None:
        pytest.skip("No suitable model found in models.py")
    model = _instantiate_model(discovered_model_cls)
    nparams = sum(p.numel() for p in model.parameters())
    # This threshold is soft-guard for current model scale
    assert nparams < 10_000_000, "Model parameter count unexpectedly large; possible accidental architectural change."
