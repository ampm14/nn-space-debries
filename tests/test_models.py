import torch
from src.models.pinn import PINN

def test_model_output_shape():
    model = PINN([2, 16, 16, 1])
    xt = torch.rand(10, 2)
    out = model(xt)
    assert out.shape == (10, 1), f"Expected (10,1), got {out.shape}"
