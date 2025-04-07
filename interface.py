import logzero
import typing as ty
from pathlib import Path
from dataclasses import dataclass
import toml
import tqdm
import shutil
import itertools
import subprocess
import sys
import os

import torch
import numpy as np

from mmd_tst_variable_detector.datasets.base import BaseDataset
from mmd_tst_variable_detector import (
    # interface
    Interface,
    InterfaceConfigArgs,
    ResourceConfigArgs,
    ApproachConfigArgs,
    DataSetConfigArgs,
    DetectorAlgorithmConfigArgs,
    CvSelectionConfigArgs,
    AlgorithmOneConfigArgs,
    BasicVariableSelectionResult,
    DistributedConfigArgs,
    RegularizationSearchParameters
)

logger = logzero.logger



# ------------------------------------------------------------------------------
# config obj

@dataclass
class BaseConfig:
    path_experiment_root: ty.Union[Path, str]
    file_name_sqlite3: str = "exp_result.sqlite3"
    name_experiment_db: str = "experiment.json"

    dir_name_data: str = "data"
    dir_models: str = "models"
    dir_logs: str = "logs"    

    def __post_init__(self):
        self.path_experiment_root = Path(self.path_experiment_root)


@dataclass
class DataSettingConfig:
    # file path to word embeddings X and Y.
    path_data_source_x: ty.Union[str, Path]
    path_data_source_y: ty.Union[str, Path]
    
    def __post_init__(self):
        self.path_data_source_x = Path(self.path_data_source_x)
        self.path_data_source_y = Path(self.path_data_source_y)


# @dataclass
# class DataGenerationConfig:
#     sample_size_train: int
#     sample_size_test: int
    
    
@dataclass
class MmdBaselineConfig:
    MAX_EPOCH: int = 9999
    
@dataclass
class CvSelectionConfig:
    MAX_EPOCH: int = 9999
    candidate_regularization_parameter: str = 'auto'
    n_regularization_parameter: int = 5

    n_subsampling: int = 10
    
    n_search_iteration: int = 10
    n_max_concurrent: int = 3
    
@dataclass
class DeviceConfig:
    train_accelerator: str = 'cpu'
    # switch to 'single' if you encounter issues.
    distributed_mode: str = 'dask'  # dask or single.
    dask_n_workers: int = 4  # the number of workers that processes hyper-parameter searching jobs.
    dask_threads_per_worker: int = 4  # the number of threads per worker.
    

@dataclass
class RootConfig:
    base: BaseConfig
    data_setting: DataSettingConfig
    data_setting_test: DataSettingConfig
    # data_generation: DataGenerationConfig
    mmd_baseline: MmdBaselineConfig
    mmd_algorithm_one: AlgorithmOneConfigArgs
    cv_selection: CvSelectionConfig
    device: DeviceConfig


import dacite


def main_single_run(path_toml_config: Path):
    """The main function to execute a pairwise comparison between two word embeddings.
    """
    assert path_toml_config.exists(), f'Not found: {path_toml_config}'
    __config_obj = toml.loads(path_toml_config.open().read())
    config_obj = dacite.from_dict(data_class=RootConfig, data=__config_obj)

    path_root_dir: Path = config_obj.base.path_experiment_root  # type: ignore
    path_root_dir.mkdir(parents=True, exist_ok=True)

    # copy the toml file to the root directory.
    shutil.copy(path_toml_config, path_root_dir / path_toml_config.name)
    
    path_dir_log = path_root_dir / config_obj.base.dir_logs
    path_dir_log.mkdir(parents=True, exist_ok=True)
    logzero.logfile(path_dir_log / 'log.txt', maxBytes=1e6, backupCount=3)
    
    path_dir_data = path_root_dir / config_obj.base.dir_name_data
    path_dir_data_train = path_dir_data / 'train'
    path_dir_data_test = path_dir_data / 'test'
    path_dir_data_train.mkdir(parents=True, exist_ok=True)
    path_dir_data_test.mkdir(parents=True, exist_ok=True)

    logger.info(f'loading word embedding file from {config_obj.data_setting.path_data_source_x}...')
    logger.info(f'loading word embedding file from {config_obj.data_setting.path_data_source_y}...')

    # Processing training data.    
    np_array_x = np.load(config_obj.data_setting.path_data_source_x)
    np_array_y = np.load(config_obj.data_setting.path_data_source_y)
    
    logger.info(f'X and Y are ready! np_array_x.shape: {np_array_x.shape}, np_array_y.shape: {np_array_y.shape}')
    assert np_array_x.shape[0] == np_array_y.shape[0], 'X and Y must have the same number of samples.'
    
    # ------------ saving the word embedding files into a py torch file. ------------
    # saving the word embedding files into a py torch file.
    # I need to reshape it into (sample-size, feature-size).
    np_array_x = np_array_x.reshape(np_array_x.shape[0], -1)
    np_array_y = np_array_y.reshape(np_array_y.shape[0], -1)
    
    path_embedding_x = path_dir_data_train / 'embedding_x.pt'
    path_embedding_y = path_dir_data_train / 'embedding_y.pt'
    
    # loading training file pair.
    torch_x = torch.from_numpy(np_array_x)
    torch_y = torch.from_numpy(np_array_y)
    torch.save({'array': torch_x}, path_embedding_x)
    torch.save({'array': torch_y}, path_embedding_y)
    
    logger.info(f'Training Pytorch files are ready at {path_dir_data_train}')
    # end processing the training data.
    
    # processing the test data.
    if config_obj.data_setting_test.path_data_source_x != "" and config_obj.data_setting_test.path_data_source_y != "":
        logger.debug('Processing Test data...')
        _test_np_array_x = np.load(config_obj.data_setting_test.path_data_source_x)
        _test_np_array_y = np.load(config_obj.data_setting_test.path_data_source_y)
        
        logger.info(f'X and Y are ready! np_array_x.shape: {_test_np_array_x.shape}, np_array_y.shape: {_test_np_array_y.shape}')
        assert _test_np_array_x.shape[0] == _test_np_array_y.shape[0], 'X and Y must have the same number of samples.'
        
        # ------------ saving the word embedding files into a py torch file. ------------
        # saving the word embedding files into a py torch file.
        # I need to reshape it into (sample-size, feature-size).
        _test_np_array_x = _test_np_array_x.reshape(_test_np_array_x.shape[0], -1)
        _test_np_array_y = _test_np_array_y.reshape(_test_np_array_y.shape[0], -1)
        
        _test_path_embedding_x = path_dir_data_test / 'embedding_x.pt'
        _test_path_embedding_y = path_dir_data_test / 'embedding_y.pt'
        
        # loading training file pair.
        _test_torch_x = torch.from_numpy(_test_np_array_x)
        _test_torch_y = torch.from_numpy(_test_np_array_y)
        torch.save({'array': _test_torch_x}, _test_path_embedding_x)
        torch.save({'array': _test_torch_y}, _test_path_embedding_y)
        
        logger.info(f'Test Pytorch files are ready at {path_dir_data_test}')
    else:
        logger.debug('No test data.')
        _test_torch_x = None
        _test_torch_y = None
    # end if


    path_work_dir = path_root_dir / 'work_dir'
    path_work_dir.mkdir(parents=True, exist_ok=True)
    
    path_detection_output = path_root_dir / 'detection_output'
    path_detection_output.mkdir(parents=True, exist_ok=True)
    
    detection_approaches = [
        ('wasserstein_independence', ''), 
        ('interpretable_mmd', 'algorithm_one')
    ]
    for __t_detection_approach in detection_approaches:
        # run the algorithm by interface.
        interface_args = InterfaceConfigArgs(
            resource_config_args=ResourceConfigArgs(
                path_work_dir=path_work_dir,
                dask_config_preprocessing=DistributedConfigArgs(),
                dask_config_detection=DistributedConfigArgs(
                    distributed_mode=config_obj.device.distributed_mode,
                    dask_n_workers=config_obj.device.dask_n_workers,
                    dask_threads_per_worker=config_obj.device.dask_threads_per_worker),
                train_accelerator=config_obj.device.train_accelerator,),
            approach_config_args=ApproachConfigArgs(
                approach_data_representation='sample_based',
                approach_variable_detector=__t_detection_approach[0],
                approach_interpretable_mmd=__t_detection_approach[1]),
            data_config_args=DataSetConfigArgs(
                data_x_train=torch_x,
                data_y_train=torch_y,
                data_x_test=_test_torch_x,
                data_y_test=_test_torch_y,
                dataset_type_backend='ram',
                dataset_type_charactersitic='static'),
            detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                mmd_algorithm_one_args=AlgorithmOneConfigArgs(
                    aggregation_kernel_length_scale='median',
                    parameter_search_parameter=RegularizationSearchParameters(
                        n_search_iteration=10,
                        max_concurrent_job=3,
                        n_regularization_parameter=6)),
                mmd_cv_selection_args=CvSelectionConfigArgs(
                    max_epoch=config_obj.cv_selection.MAX_EPOCH,
                    n_subsampling=config_obj.cv_selection.n_subsampling,
                    parameter_search_parameter=RegularizationSearchParameters(
                        n_search_iteration=config_obj.cv_selection.n_search_iteration,
                        max_concurrent_job=config_obj.cv_selection.n_max_concurrent,
                        n_regularization_parameter=config_obj.cv_selection.n_regularization_parameter
                    )
                ))
        )
    
        __interface = Interface(interface_args)
        __interface.fit()
        result_obj = __interface.get_result(output_mode='verbose')
        assert isinstance(result_obj.detection_result_sample_based, BasicVariableSelectionResult)
        
        detection_obj_json: str = result_obj.as_json()
        
        __approach_name_concat = '-'.join(__t_detection_approach)
        with open(path_detection_output / f'{__approach_name_concat}.json', 'w') as f:
            f.write(detection_obj_json)
        # end with


def execute_all_pairwise_combination(path_toml_config: Path,
                                     path_python_interpreter: ty.Optional[Path] = None,
                                     path_this_script: ty.Optional[Path] = None
                                     ):
    """Executing the variable detection for all pairwise combinations of the given word embeddings.
    
    Due to Dask processing issue, this function calls subprocess via Shell. The issue description is below.
    
    <Issue>
    Dask subprocess causes often shutdown of the Dask workers. The current my API design launces Dask workers "EVERY TIME" I call the API.
    Hence, Dask workers keep increasing. That causes bottleneck in processing speed.
    Dask workers are automatically shutdown if I call it via Shell since Python process is terminated.
    I use this mechanism to avoid the bottleneck.  
    """
    if path_python_interpreter is None:
        path_python_interpreter = Path(sys.executable)
    # end if
    if path_this_script is None:
        path_this_script = Path(__file__)
    # end if
    
    assert path_toml_config.exists(), f'Not found: {path_toml_config}'
    __config_obj = toml.loads(path_toml_config.open().read())
    config_obj = dacite.from_dict(data_class=RootConfig, data=__config_obj)

    logger.debug(f'Making the root directory: {config_obj.base.path_experiment_root}')
    path_root_dir: Path = config_obj.base.path_experiment_root  # type: ignore
    path_root_dir.mkdir(parents=True, exist_ok=True)
    
    # listing up all files in the source directory.
    config_obj_x = Path(config_obj.data_setting.path_data_source_x)
    config_obj_y = Path(config_obj.data_setting.path_data_source_y)
    assert config_obj_x.exists(), f'Not found: {config_obj_x}'
    assert config_obj_x.as_posix() == config_obj_y.as_posix(), 'X and Y must be the same.'
    seq_files_source_npy = list(config_obj_x.rglob('*npy'))
    
    file_pairs = {(x, y) for x, y in itertools.combinations(seq_files_source_npy, 2) if x != y}  # list of file path pairs.
    
    for __t_file_pair in tqdm.tqdm(file_pairs):
        logger.info(f'Comparing {__t_file_pair[0].name} and {__t_file_pair[1].name}')
        # copy the toml file and create a new toml config file for this combination.
        __new_toml_config = path_root_dir / f'config_{__t_file_pair[0].stem}_{__t_file_pair[1].stem}.toml'
        shutil.copy(path_toml_config, __new_toml_config)
        __config_obj = toml.loads(__new_toml_config.open().read())
        
        # a new root directory for this comparison.
        __path_pair_root = path_root_dir / f'{__t_file_pair[0].stem}_{__t_file_pair[1].stem}'
        __path_pair_root_alt = path_root_dir / f'{__t_file_pair[1].stem}_{__t_file_pair[0].stem}'
        __config_obj['base']['path_experiment_root'] = __path_pair_root.as_posix()
        
        # skip the same process if the directory already exists.
        if __path_pair_root.exists() or __path_pair_root_alt.exists():
            logger.info(f'Skipping the same process since the directory already exists: {__path_pair_root}')
            continue
        # end if
        
        # replacing training file source of x and y.
        __config_obj['data_setting']['path_data_source_x'] = __t_file_pair[0].as_posix()
        __config_obj['data_setting']['path_data_source_y'] = __t_file_pair[1].as_posix()
        
        # replacing test file source of x and y.
        __path_test_file_x = __t_file_pair[0].__str__().replace('train', 'test')
        __path_test_file_y = __t_file_pair[1].__str__().replace('train', 'test')
        assert Path(__path_test_file_x).exists(), f'Not found: {__path_test_file_x}'
        assert Path(__path_test_file_y).exists(), f'Not found: {__path_test_file_y}'
        __config_obj['data_setting_test']['path_data_source_x'] = __path_test_file_x
        __config_obj['data_setting_test']['path_data_source_y'] = __path_test_file_y
        
        # saving and updating the current toml config file.
        with open(__new_toml_config, "w") as toml_file:
            toml.dump(__config_obj, toml_file)
        # end with
        
        logger.info(f'The new config file is ready at {__new_toml_config}. The work directory is {__path_pair_root}')
        
        # Define the shell command.
        _shell_commands = [path_python_interpreter.as_posix(), 
                           path_this_script.as_posix(), 
                           '--mode', 
                           'single_run', 
                           '--path_config', 
                           __new_toml_config.as_posix()]
        
        # executing the main function.
        logger.info(f'Executing -> {_shell_commands}')
        try:
            result = subprocess.run(_shell_commands, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess returned an error: {e}")
            raise Exception(f"Subprocess returned an error: {e}")
        # end try
        
        if result.returncode != 0:
            logger.error(result.stderr)
            raise Exception(f"Subprocess returned an error: {result.stderr}")
        # end if
        logger.info(result.stdout)
        # main_single_run(__new_toml_config)
    # end for


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    opt = ArgumentParser()
    opt.add_argument('--mode', type=str, required=True, choices=['single_run', 'all'])
    opt.add_argument('--path_config', 
                     type=str, 
                     required=True, 
                     help='Path to the config file. TOML format. \
                         When mode is "single_run", you put the a toml file ready-to-use. \
                             When the mode is "all", you put the toml file that you"ve described the base path.')
    __args = opt.parse_args()

    logger.info("---- Begin of the script ----")
    if __args.mode == 'single_run':
        main_single_run(Path(__args.path_config))
    elif __args.mode == 'all':
        execute_all_pairwise_combination(Path(__args.path_config))
    else:
        raise ValueError(f'Unknown mode: {__args.mode}')
    # end if
    logger.info("---- End of the script ----")