import numpy as np

def create_data(size: int = 1000, n_noise_points: int = 100, **kwargs) -> np.ndarray:
    np.random.seed(42)
    arr = np.random.uniform(
        kwargs.get("low", 240), high=kwargs.get("high", 240), size=(size,)
    )

    noise = np.random.normal(
        loc=kwargs.get("loc", 240),
        scale=kwargs.get("scale", 20),
        size=(n_noise_points,),
    )
    noise_point_indices = np.random.choice(
        range(size), replace=False, size=(n_noise_points,)
    )

    arr[noise_point_indices] = noise
    return arr