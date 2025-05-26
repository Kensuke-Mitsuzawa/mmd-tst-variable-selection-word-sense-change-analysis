import typing as ty
from pathlib import Path
import random
import dataclasses

import numpy as np
import torch

import logging


from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator, MmdValues

from .gpu_manager import get_device

module_logger = logging.getLogger()

"""A script to comput MMD distances over all possible pair-wise pairs"""





def calibrate_gaussian_kernel(seq_files_source_npy: ty.List[Path],
                              n_calibration_per_epoch: int = 100,
                              random_seed: int = 42) -> QuadraticKernelGaussianKernel:
    """Loading word embedding files, selecting subset (calibration data), and calculating the length scale"""

    def initialise_length_scale(calibration_emb: torch.Tensor,
                                kernel_length_scale_median_option: str = "dimensionwise"):

        if 'single' in kernel_length_scale_median_option:
            _is_dimension_wise = False
        elif 'dimensionwise' in kernel_length_scale_median_option:
            _is_dimension_wise = True
        else:
            raise ValueError(f"Invalid median options: {kernel_length_scale_median_option}")
        # end if

        kernel_func_obj = QuadraticKernelGaussianKernel(
            is_dimension_median_heuristic=_is_dimension_wise,
            ard_weights=torch.ones(calibration_emb.shape[1])
        )
        device_obj = get_device()
        calibration_emb = calibration_emb.to(device_obj)
        kernel_func_obj.to(device_obj)

        with torch.no_grad():
            module_logger.debug("Computing length scale using the calibration set...")
            if _is_dimension_wise:
                # TODO: there is the safe guard avoiding L2(x, x).
                tensor_length_scale = kernel_func_obj._get_median_dim(
                    x=calibration_emb,
                    y=calibration_emb,
                    is_safe_guard_same_xy=False)
            else:
                tensor_length_scale = kernel_func_obj._get_median_single(
                    x=calibration_emb,
                    y=calibration_emb)
            # end if
            module_logger.debug("Done computing the length scale...")    
            assert tensor_length_scale is not None
        # end with
        tensor_length_scale.to(device_obj)
        kernel_func_obj.bandwidth = torch.nn.Parameter(tensor_length_scale)

        kernel_func_obj.to(torch.device('cpu'))  # setting back to CPU device to avoid troubles.

        return kernel_func_obj

    random_gen = random.Random(random_seed)

    # constructing a set of numpy array
    stack_calibration_array: ty.List[np.ndarray] = []
    for _path_npy in seq_files_source_npy:
        np_array: np.ndarray = np.load(_path_npy)
        # I need to reshape it into (sample-size, feature-size).
        np_array = np_array.reshape(np_array.shape[0], -1)
        _sample_id_calibtation = random_gen.sample(range(np_array.shape[0]), k=n_calibration_per_epoch)

        stack_calibration_array.append(np_array[_sample_id_calibtation, :])
    # end for
    _calibration_embedding = torch.from_numpy(np.array(stack_calibration_array))  # (n-time-epoch, n-calibration-per-epoch, dimension)
    # Melt the array
    calibration_embedding = _calibration_embedding.reshape(-1, 100)

    kernel_obj = initialise_length_scale(calibration_embedding)

    return kernel_obj


def get_mmd_estimator(kernel_obj: QuadraticKernelGaussianKernel) -> QuadraticMmdEstimator:
    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel_obj)

    return mmd_estimator


def compute_mmd_distance(mmd_estimator: QuadraticMmdEstimator, file_numpy_file_x: Path, file_numpy_file_y: Path) -> MmdValues:
    device_obj = get_device()
    mmd_estimator = mmd_estimator.to(device_obj)

    # loading embedding files
    array_x = np.load(file_numpy_file_x)
    array_y = np.load(file_numpy_file_y)

    tensor_x = torch.from_numpy(array_x)
    tensor_y = torch.from_numpy(array_y)

    tensor_x= tensor_x.to(device_obj)
    tensor_y = tensor_y.to(device_obj)

    with torch.no_grad():
        _mmd_distance_container = mmd_estimator.forward(tensor_x, tensor_y)
    # end with

    # setting back to CPU object
    res_dict_obj = dataclasses.asdict(_mmd_distance_container)
    for _k, _v in res_dict_obj.items():
        if isinstance(_v, torch.Tensor):
            res_dict_obj[_k] = _v.to(torch.device('cpu'))
    # end for
    obj = MmdValues(**res_dict_obj)
    
    return obj