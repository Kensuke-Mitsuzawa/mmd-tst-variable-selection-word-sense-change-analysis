import typing as ty
import torch
import GPUtil


def get_less_busy_cuda_device() -> int:
    gpu_device_info = GPUtil.getGPUs()
    seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
    gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
    return gpu_id_less_busy[0]


def get_device(is_prefer_gpu: bool = True, gpu_device_id_preference: ty.Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        if is_prefer_gpu:
            if gpu_device_id_preference is None:
                _device_id_less_busy = get_less_busy_cuda_device()
                return torch.device(f'cuda:{_device_id_less_busy}')
            else:
                return torch.device(f'cuda:{gpu_device_id_preference}')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')
