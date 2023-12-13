import numpy as np
from itertools import permutations
from src.data_processing import HyperspectralImage


def compare_abundances(hsi: HyperspectralImage, predicted_abundances):
    true_abundances = hsi.S_gt.reshape(hsi.n_sources, hsi.n_col, hsi.n_row)
    predicted_abundances = predicted_abundances.reshape(hsi.n_sources, hsi.n_col, hsi.n_row)

    min_mse = np.inf
    permuts = list(permutations(np.arange(0, hsi.n_sources, 1)))
    for perm in permuts:
        mse = ((predicted_abundances[perm, :, :] - true_abundances) ** 2).mean()
        if mse < min_mse:
            min_mse = mse
    return min_mse


def compare_endmembers(hsi: HyperspectralImage, predicted_endmembers):
    min_mse = np.inf
    permuts = list(permutations(np.arange(0, hsi.n_sources, 1)))
    for perm in permuts:
        mse = ((predicted_endmembers[:, perm] - hsi.A_gt) ** 2).mean()
        if mse < min_mse:
            min_mse = mse
    return min_mse
