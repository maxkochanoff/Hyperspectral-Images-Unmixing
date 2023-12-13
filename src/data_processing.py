import scipy.io as sci
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from tqdm.notebook import tqdm
from src.conf import PATCH_SIZE, PATCH_NUMBER, NUMBER_IMAGES, NOISE_STD


class HyperspectralImage:
    def __init__(self, image_path: str, ground_truth_path: str, n_col: int, n_row: int, n_bands: int, n_sources: int):
        self.n_col = n_col
        self.n_row = n_row
        self.n_bands = n_bands
        self.n_sources = n_sources

        hsi = sci.loadmat(image_path)
        self.X = hsi["X"] if "X" in hsi else hsi["V"]
        self.X = self.X.reshape(-1, self.n_col, self.n_row)
        # self.X = self.X.T.reshape(self.n_col, self.n_row, -1)
        self.X = self.X.astype(float)
        self.X /= self.X.max()

        gt = sci.loadmat(ground_truth_path)
        self.S_gt = gt['A']
        self.A_gt = gt['M']


class OriginalImageDataset(torch.utils.data.Dataset):
    def __init__(self, hsi, with_patches=True):
        self.hsi = hsi
        self.with_patches = with_patches
        if with_patches:
            patches = extract_patches_2d(np.moveaxis(self.hsi.X, 0, -1),
                                         (PATCH_SIZE, PATCH_SIZE), max_patches=PATCH_NUMBER)
            self.X = np.moveaxis(patches, -1, 1)
        else:
            self.X = self.hsi.X

    def __len__(self):
        if self.with_patches:
            return len(self.X)
        else:
            return 1

    def __getitem__(self, idx):
        if self.with_patches:
            return {'X': torch.Tensor(self.X[idx])}
        else:
            return {'X': torch.Tensor(self.X)}


def _f(x, x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1 + 1e-7)
    b = y1 - a * x1
    return a * x + b


def _perturbation(A, c_var=0.5):
    A_pert = np.zeros(A.shape)
    ksi = np.random.uniform(1 - 0.5 * c_var, 1 + 0.5 * c_var, 3)
    L = A.shape[0]
    U = np.random.normal(0, NOISE_STD)
    L_break = int(L / 2) + int(L * U / 3)

    # до L_break домножаем на линию по точкам (0, кси_1), (L_break, кси_2)
    # далее после L_break домножаем  на линию по точкам (L_break, кси_2), (L, кси_3)

    x1, y1 = 0, ksi[0]
    x2, y2 = L_break, ksi[1]
    for l in range(L):
        A_pert[l] = A[l] * _f(l, x1, y1, x2, y2)
        if l == L_break:
            x1, y1 = L_break, ksi[1]
            x2, y2 = L - 1, ksi[2]

    return A_pert


def _spectral_variability_model(k, X_pert, A_pert, S_pert):
    sv_S_k = S_pert[:, k]
    sv_A = _perturbation(A_pert)
    N = np.random.normal(0, NOISE_STD, X_pert.shape[0])  # vector of L-size

    sv_X_k = sv_A @ sv_S_k + N

    return sv_X_k, sv_A, sv_S_k


class PerturbedSimpleData(torch.utils.data.Dataset):
    def __init__(self, hsi: HyperspectralImage, endmembers_1_estim, abundances_1_estim):
        self.hsi = hsi
        X = self.hsi.X
        self.X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        self.A = endmembers_1_estim
        self.S = abundances_1_estim.reshape(abundances_1_estim.shape[0],
                                            abundances_1_estim.shape[1] * abundances_1_estim.shape[2])
        self.N_train = NUMBER_IMAGES

    def __len__(self):
        return self.N_train

    def __getitem__(self, idx):
        A_pert = _perturbation(self.A)

        N = np.random.normal(0, NOISE_STD, self.X.shape)
        S_pert = self.S
        X_pert = A_pert @ S_pert + N

        X_reshaped = X_pert.reshape((self.hsi.n_bands, self.hsi.n_row, self.hsi.n_col))
        S_reshaped = S_pert.reshape((self.hsi.n_sources, self.hsi.n_row, self.hsi.n_col))

        return {
            'X': torch.Tensor(X_reshaped),
            'A': torch.Tensor(A_pert),
            'S': torch.Tensor(S_reshaped)
        }


class PerturbedSVData(torch.utils.data.Dataset):
    def __init__(self, hsi: HyperspectralImage, endmembers_1_estim, abundances_1_estim):
        self.hsi = hsi
        self.X = hsi.X
        self.A = endmembers_1_estim
        self.S = abundances_1_estim
        self.N_train = NUMBER_IMAGES

    def __len__(self):
        return self.N_train

    def __getitem__(self, idx):
        sv_X_reshaped, sv_A_mean, sv_S_reshaped = _generate_sv_data(self.hsi, self.X, self.A, self.S, need_A_mean=True)

        return {
            'X': torch.Tensor(sv_X_reshaped),
            'A': torch.Tensor(sv_A_mean),
            'S': torch.Tensor(sv_S_reshaped)
        }


def _generate_sv_data(hsi: HyperspectralImage, X, endmembers_1_estim, abundances_1_estim, need_A_mean=False):

    n_row, n_col = X.shape[1:]

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    S = abundances_1_estim.reshape(abundances_1_estim.shape[0], abundances_1_estim.shape[1] * abundances_1_estim.shape[2])

    A_pert = _perturbation(endmembers_1_estim)

    N = np.random.normal(0, NOISE_STD, X.shape)
    S_pert = S
    X_pert = A_pert @ S_pert + N

    sv_X, sv_A_array, sv_S = np.zeros(X_pert.shape), [], np.zeros(S_pert.shape)

    for k in tqdm(range(X.shape[1])):
        sv_X_k, sv_A, sv_S_k = _spectral_variability_model(k, X_pert, A_pert, S_pert)
        sv_X[:, k] = sv_X_k
        sv_A_array.append(sv_A)
        sv_S[:, k] = sv_S_k  # на самом деле вообще не меняется

    sv_A_array = np.array(sv_A_array)

    sv_X_reshaped = sv_X.reshape((hsi.n_bands, n_row, n_col))
    sv_S_reshaped = sv_S.reshape((hsi.n_sources, n_row, n_col))

    if need_A_mean:
        sv_A_array_reshaped = np.mean(sv_A_array, axis=0)
    else:
        sv_A_array_reshaped = sv_A_array.reshape((hsi.n_row, hsi.n_col, hsi.n_bands, hsi.n_sources))

    return sv_X_reshaped, sv_A_array_reshaped, sv_S_reshaped


def _create_patches(hsi: HyperspectralImage, sv_X, sv_A_array, sv_S, patch_size, max_patches, is_pixel_wise=True):
    n, m = sv_X.shape[1:]
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    coordinate_matrix = np.stack((x, y), axis=-1)

    patches_mask = extract_patches_2d(coordinate_matrix, patch_size, max_patches=max_patches)

    sv_X_patches = []
    sv_A_patches = []
    sv_S_patches = []

    for patch_mask in patches_mask:
        x1, y1 = patch_mask[0][0][:]
        x2, y2 = patch_mask[patch_size[0] - 1][patch_size[1] - 1][:]

        sv_X_patches.append(sv_X[:, x1:x2 + 1, y1:y2 + 1])
        sv_S_patches.append(sv_S[:, x1:x2 + 1, y1:y2 + 1])
        if is_pixel_wise:
            sv_A_patches.append(sv_A_array[x1:x2 + 1, y1:y2 + 1, :, :].reshape((patch_size[0] * patch_size[1],
                                                                                hsi.n_bands, hsi.n_sources)).mean(axis=0))

    return np.array(sv_X_patches), np.array(sv_A_patches), np.array(sv_S_patches)


def generate_honest_sv_patches(hsi: HyperspectralImage, endmembers_1_estim, abundances_1_estim):
    """
    Example usage:
    generate_sv_patches(N_train=10, max_patches=100, X=some_X_data, A=some_A_data, S=some_S_data)
    """
    for i in tqdm(range(NUMBER_IMAGES)):
        sv_X, sv_A_array, sv_S = _generate_sv_data(hsi, hsi.X, endmembers_1_estim, abundances_1_estim)
        X_patches, A_patches, S_patches = _create_patches(hsi, sv_X, sv_A_array, sv_S,
                                                          patch_size=(PATCH_SIZE, PATCH_SIZE),
                                                          max_patches=PATCH_NUMBER)
        # SAVE OBTAINED PATCHES TO FILE
        # Save 100+100+100 patches for each image
        # Later, when accessing, you can go through random indices
        for j, (pX, pA, pS) in tqdm(enumerate(zip(X_patches, A_patches, S_patches))):
            name_file_X = f'temp_data/honest_sv_patches/im_{i * PATCH_NUMBER + j}_X.npy'
            name_file_A = f'temp_data/honest_sv_patches/im_{i * PATCH_NUMBER + j}_A.npy'
            name_file_S = f'temp_data/honest_sv_patches/im_{i * PATCH_NUMBER + j}_S.npy'

            # Save data using NumPy's np.save
            np.save(name_file_X, pX)
            np.save(name_file_A, pA)
            np.save(name_file_S, pS)


def generate_simple_patches(hsi: HyperspectralImage, A, S):
    X = hsi.X.reshape(hsi.X.shape[0], hsi.X.shape[1] * hsi.X.shape[2])
    S = S.reshape(S.shape[0], S.shape[1] * S.shape[2])
    for i in tqdm(range(NUMBER_IMAGES)):

        A_pert = _perturbation(A)

        N = np.random.normal(0, NOISE_STD, X.shape)
        S_pert = S
        X_pert = A_pert @ S_pert + N

        X_pert = X_pert.reshape(X_pert.shape[0], hsi.X.shape[1] , hsi.X.shape[2])
        S_pert = S_pert.reshape(S_pert.shape[0], hsi.X.shape[1], hsi.X.shape[2])

        X_patches, _, S_patches = _create_patches(hsi, X_pert, A_pert, S_pert,
                                                  patch_size=(PATCH_SIZE, PATCH_SIZE), max_patches=PATCH_NUMBER,
                                                  is_pixel_wise=False)
        # SAVE OBTAINED PATCHES TO FILE
        # Save 100+100+100 patches for each image
        # Later, when accessing, you can go through random indices
        for j, (pX, pS) in tqdm(enumerate(zip(X_patches, S_patches))):
            name_file_X = f'temp_data/simple_patches/im_{i * PATCH_NUMBER + j}_X.npy'
            name_file_A = f'temp_data/simple_patches/im_{i * PATCH_NUMBER + j}_A.npy'
            name_file_S = f'temp_data/simple_patches/im_{i * PATCH_NUMBER + j}_S.npy'

            # Save data using NumPy's np.save
            np.save(name_file_X, pX)
            np.save(name_file_A, A_pert)
            np.save(name_file_S, pS)


class PerturbedSimpleDataPatches(torch.utils.data.Dataset):
    def __init__(self):
        self.N_train = NUMBER_IMAGES
        self.patch_size = (PATCH_SIZE, PATCH_SIZE)
        self.max_patches = PATCH_NUMBER

    def __len__(self):
        return self.N_train * self.max_patches

    def __getitem__(self, idx):
        name_file_X = f'temp_data/simple_patches/im_{idx}_X.npy'
        name_file_A = f'temp_data/simple_patches/im_{idx}_A.npy'
        name_file_S = f'temp_data/simple_patches/im_{idx}_S.npy'

        sv_X_patch = np.load(name_file_X)
        sv_A_patch = np.load(name_file_A)
        sv_S_patch = np.load(name_file_S)

        return {
            'X': torch.Tensor(sv_X_patch),
            'A': torch.Tensor(sv_A_patch),
            'S': torch.Tensor(sv_S_patch)
        }


class PerturbedSVDataPatches(torch.utils.data.Dataset):
    def __init__(self):
        self.N_train = NUMBER_IMAGES
        self.patch_size = (PATCH_SIZE, PATCH_SIZE)
        self.max_patches = PATCH_NUMBER

    def __len__(self):
        return self.N_train * self.max_patches

    def __getitem__(self, idx):
        name_file_X = f'temp_data/honest_sv_patches/im_{idx}_X.npy'
        name_file_A = f'temp_data/honest_sv_patches/im_{idx}_A.npy'
        name_file_S = f'temp_data/honest_sv_patches/im_{idx}_S.npy'

        sv_X_patch = np.load(name_file_X)
        sv_A_patch = np.load(name_file_A)
        sv_S_patch = np.load(name_file_S)

        return {
            'X': torch.Tensor(sv_X_patch),
            'A': torch.Tensor(sv_A_patch),
            'S': torch.Tensor(sv_S_patch)
        }


def _extract_random_patch(X, A, S):
    n, m = X.shape[1:]

    x1 = np.random.randint(0, n - PATCH_SIZE + 1)
    y1 = np.random.randint(0, m - PATCH_SIZE + 1)
    x2 = x1 + PATCH_SIZE
    y2 = y1 + PATCH_SIZE

    return X[:, x1:x2, y1:y2], A, S[:, x1:x2, y1:y2]


class PerturbedSVDataRandomPatches(torch.utils.data.Dataset):
    def __init__(self, hsi: HyperspectralImage, endmembers_1_estim, abundances_1_estim):
        self.hsi = hsi
        self.X = self.hsi.X  # .reshape(X.shape[0], X.shape[1] * X.shape[2])
        self.A = endmembers_1_estim
        self.S = abundances_1_estim  # .reshape(S.shape[0], S.shape[1] * S.shape[2])
        self.N_train = NUMBER_IMAGES
        self.max_patches = PATCH_NUMBER

    def __len__(self):
        return self.N_train * self.max_patches

    def __getitem__(self, idx):
        X_patch, A, S_patch = _extract_random_patch(self.X, self.A, self.S)
        sv_X_reshaped, sv_A_mean, sv_S_reshaped = _generate_sv_data(self.hsi, X_patch, A, S_patch, need_A_mean=True)

        return {
            'X': torch.Tensor(sv_X_reshaped),
            'A': torch.Tensor(sv_A_mean),
            'S': torch.Tensor(sv_S_reshaped)
        }
