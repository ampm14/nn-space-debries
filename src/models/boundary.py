def apply_dirichlet_boundary(x, t, T_value):
    """
    Returns input coordinates and target temperatures for Dirichlet BCs.

    Args:
        x (torch.Tensor): x-coordinates of the boundary (e.g., 0 or 1)
        t (torch.Tensor): corresponding times
        T_value (float or torch.Tensor): target temperature at the boundary

    Returns:
        xt: torch.Tensor of input coordinates
        T: target temperature values
    """
    xt = torch.cat([x, t], dim=1)
    T = T_value if isinstance(T_value, torch.Tensor) else torch.tensor([[T_value]])
    return xt, T.expand_as(x)
