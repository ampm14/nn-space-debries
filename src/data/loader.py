import torch

def generate_synthetic_data(num_points=1000):
    """
    Generate synthetic temperature data as (x, t, T) tuples.
    Useful for testing data pipelines.
    """
    x = torch.rand(num_points, 1)
    t = torch.rand(num_points, 1)
    T = torch.sin(torch.pi * x) * torch.exp(-t)

    return x, t, T
