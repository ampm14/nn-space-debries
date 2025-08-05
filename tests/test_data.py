from src.data.loader import generate_synthetic_data

def test_data_shapes():
    x, t, T = generate_synthetic_data(100)
    assert x.shape == t.shape == T.shape == (100, 1), "Shape mismatch in synthetic data"
