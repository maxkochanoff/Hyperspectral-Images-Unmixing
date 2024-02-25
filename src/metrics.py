import numpy as np
from itertools import permutations
from src.data_processing import HyperspectralImage
from sklearn.metrics.pairwise import cosine_similarity


def compare_abundances(hsi: HyperspectralImage, predicted_abundances, metric="mse"):
    true_abundances = hsi.S_gt.reshape(hsi.n_sources, hsi.n_col, hsi.n_row)
    predicted_abundances = predicted_abundances.reshape(hsi.n_sources, hsi.n_col, hsi.n_row)

    if metric == "mse":
        res = np.inf
        permuts = list(permutations(np.arange(0, hsi.n_sources, 1)))
        for perm in permuts:
            mse = ((predicted_abundances[perm, :, :] - true_abundances) ** 2).mean()
            res = min(mse, res)

    elif metric == "cos_sim":
        res = 0
        permuts = list(permutations(np.arange(0, hsi.n_sources, 1)))
        for perm in permuts:
            cos_sim = cosine_similarity(predicted_abundances[perm, :, :].reshape(1, -1),
                                        true_abundances.reshape(1, -1)).item()
            res = max(cos_sim, res)

    else:
        res = None

    return res


def compare_endmembers(hsi: HyperspectralImage, predicted_endmembers, metric="mse"):
    for i in range(predicted_endmembers.shape[1]):
        M = predicted_endmembers[:, i].max().item()
        predicted_endmembers[:, i] = predicted_endmembers[:, i] / M

    if metric == "mse":
        res = np.inf
        permuts = list(permutations(np.arange(0, hsi.n_sources, 1)))
        for perm in permuts:
            mse = ((predicted_endmembers[:, perm] - hsi.A_gt) ** 2).mean()
            res = min(mse, res)

    elif metric == "cos_sim":
        res = 0
        permuts = list(permutations(np.arange(0, hsi.n_sources, 1)))
        for perm in permuts:
            cos_sim = cosine_similarity(predicted_endmembers[:, perm].reshape(1, -1),
                                        hsi.A_gt.reshape(1, -1)).item()
            res = max(cos_sim, res)

    else:
        res = None

    return res
