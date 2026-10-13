import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_with_noise(n_noise=60, n_correlated=10, seed=0):
    """Breast cancer data plus pure noise columns and noisy copies of real features.

    Real columns keep their names; the added ones start with 'noise_' or 'copy_', so a method's
    choices can be checked against the truth.
    """
    rng = np.random.default_rng(seed)
    bunch = load_breast_cancer(as_frame=True)
    x, y = bunch.data.copy(), bunch.target
    extra = {}
    for i in range(n_noise):
        extra[f"noise_{i:02d}"] = rng.normal(0, 1, len(x))
    picks = rng.choice(x.columns, n_correlated, replace=False)
    for col in picks:
        scale = x[col].std()
        extra[f"copy_{col.replace(' ', '_')}"] = x[col] + rng.normal(0, 0.3 * scale, len(x))
    x = pd.concat([x, pd.DataFrame(extra, index=x.index)], axis=1)
    return x, y


def is_real(name):
    return not (name.startswith("noise_") or name.startswith("copy_"))
