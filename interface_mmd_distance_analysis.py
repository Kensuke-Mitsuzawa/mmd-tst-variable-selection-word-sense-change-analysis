import logzero
import typing as ty
from pathlib import Path
from dataclasses import asdict, dataclass
import toml
import tqdm
import itertools
import re
import json
import pandas as pd
import seaborn as sns
import tempfile

import matplotlib.pyplot as plt

# import torch
import numpy as np

# from mmd_tst_variable_detector.datasets.base import BaseDataset
# from mmd_tst_variable_detector import (
#     # interface
#     AlgorithmOneConfigArgs,
# )

from word_sense_change_analysis.preprocessing import PreprocessingOutputConfig
from word_sense_change_analysis import mmd_distance_analysis
from word_sense_change_analysis import visualisations, visualisation_header


logger = logzero.logger

"""An interface script for """

# ------------------------------------------------------------------------------


def visualisation_main(path_experiment_root: Path,
                       config_obj: ty.Dict, 
                       file_name_result_json: str = "mmd_distance_result.json",
                       sub_dir_name: str = "mmd_distances"):
    def visualise_mmd_distance(seq_stack_detection_results: ty.List[ty.Dict],
                               target_field: str, 
                                path_plot_save: Path,
                                year_start: int,
                                year_end: int,
                                year_range: int,
                                sep_skip_year: ty.List):
        assert target_field in ('mmd', 'ratio')

        year_convertor = visualisations.TimeEpochConfiguration(
            year_start=year_start,
            year_end=year_end,
            year_range=year_range,
            sep_skip_year=sep_skip_year
        )

        seq_obj_matrix = []
        _obj: ty.Dict
        for _obj in seq_stack_detection_results:
            assert "epoch_no_x" in _obj
            assert "epoch_no_y" in _obj
            assert "mmd_distance" in _obj

            _obj_updated = {
                'x': year_convertor.get_year_label(int(_obj['epoch_no_x'])),
                'y': year_convertor.get_year_label(int(_obj['epoch_no_y'])),
            }
            
            if target_field == "mmd":
                _obj_updated.update({'mmd': _obj['mmd_distance']})
            elif target_field == "ratio":
                _obj_updated.update({'ratio': _obj['ratio']})
            else:
                raise ValueError()
            # end if

            seq_obj_matrix.append(_obj_updated)
        # end for

        df_vals = pd.DataFrame(seq_obj_matrix)
        
        f, ax = plt.subplots(figsize=(9, 6))
        df_pivot = df_vals.pivot(index='x', columns='y', values=target_field)

        sns.heatmap(df_pivot, annot=False, linewidths=.5, ax=ax, annot_kws={"fontsize": 12})

        # ax.set_title(f'Variable-count heatmap')
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        logger.debug(f'Saved the heatmap to {path_plot_save}')
        f.savefig(path_plot_save.as_posix(), bbox_inches='tight')
    # end def

    # getting configuration objects
    _config_analysis = config_obj['Analysis']

    # -------------------------------------------------
    # Analysis config
    path_figure_dir = Path(_config_analysis['path_analysis_output'])
    path_figure_dir.mkdir(parents=True, exist_ok=True)

    time_label_start: int = _config_analysis['TimeEpochLabelStart']
    time_label_end: int = _config_analysis['TimeEpochLabelEnd']
    time_label_range: int = _config_analysis['TimeEpochLabelRange']

    if 'sep_skip_year' in _config_analysis:
        sep_skip_year : ty.List[int] = _config_analysis["sep_skip_year"]
    else:
        sep_skip_year = []
    # end if
    # skip_epoch_index: ty.List[int] = _config_analysis['skip_epoch_index']
    # -------------------------------------------------

    path_mmd_distance_result_json = Path(path_experiment_root) / sub_dir_name / file_name_result_json
    assert path_mmd_distance_result_json.exists()
    seq_dict_mmd_distance = json.loads(path_mmd_distance_result_json.open('r').read())

    # sorting the epoch keys. The younger number should be always at x.
    _obj: ty.Dict
    keys_numbers_processed = []
    for _obj in seq_dict_mmd_distance:
        _pair_key = (_obj['epoch_no_x'], _obj['epoch_no_y'])
        if _pair_key in keys_numbers_processed:
            logger.debug(f'The pair already exists. I skip it.')
        # end if
        if _obj['epoch_no_x'] < _obj['epoch_no_y']:
            # revese the number of (X, Y)
            __epoch_no_x_new = _obj['epoch_no_y']
            __epoch_no_y_new = _obj['epoch_no_x']
            _obj['epoch_no_x'] = __epoch_no_x_new
            _obj['epoch_no_y'] = __epoch_no_y_new
        # end if
        keys_numbers_processed.append(_pair_key)
    # end for


    path_figure_save = Path(path_experiment_root) / sub_dir_name / "mmd_heatmap.png"
    visualise_mmd_distance(
        seq_stack_detection_results=seq_dict_mmd_distance,
        target_field='mmd',
        path_plot_save=path_figure_save,
        year_start=time_label_start,
        year_end=time_label_end,
        year_range=time_label_range,
        sep_skip_year=sep_skip_year)
    
    path_figure_save = Path(path_experiment_root) / sub_dir_name / "ratio.png"
    visualise_mmd_distance(
        seq_stack_detection_results=seq_dict_mmd_distance,
        target_field='ratio',
        path_plot_save=path_figure_save,
        year_start=time_label_start,
        year_end=time_label_end,
        year_range=time_label_range,
        sep_skip_year=sep_skip_year)

    # # coha dataset temp codes
    # # filtering out the file index 0
    # seq_dict_mmd_distance_filtered = [_d for _d in seq_dict_mmd_distance if _d['epoch_no_x'] != 1 and _d['epoch_no_y'] != 1]
    # path_figure_save = Path(path_experiment_root) / sub_dir_name / "mmd_heatmap_from_1830.png"
    # visualise_mmd_distance(
    #     seq_stack_detection_results=seq_dict_mmd_distance_filtered,
    #     target_field='mmd',
    #     path_plot_save=path_figure_save,
    #     year_start=1830,
    #     year_end=time_label_end,
    #     year_range=time_label_range,
    #     sep_skip_year=sep_skip_year)
    
    # path_figure_save = Path(path_experiment_root) / sub_dir_name / "ratio_from_1830.png"
    # visualise_mmd_distance(
    #     seq_stack_detection_results=seq_dict_mmd_distance_filtered,
    #     target_field='ratio',
    #     path_plot_save=path_figure_save,
    #     year_start=1830,
    #     year_end=time_label_end,
    #     year_range=time_label_range,
    #     sep_skip_year=sep_skip_year)


def execute_all_pairwise_combination(path_experiment_root: Path,
                                     preprocessing_config: PreprocessingOutputConfig,
                                     sub_dir_name: str = "mmd_distances",
                                     file_name_result_json: str = "mmd_distance_result.json",
                                     n_calibration_data: int = 400
                                     ):
    """Computing MMD distances for all possible combinations.    
    """
    def _merge_train_and_test_npy(seq_train_npy: ty.List[Path], seq_test_npy: ty.List[Path]) -> ty.List[Path]:
        """merging train and test files, saving into a temp file and returning the list of path"""
        dict_f_name2path_train = {_f_path.name: _f_path for _f_path in seq_train_npy}
        dict_f_name2path_test = {_f_path.name: _f_path for _f_path in seq_test_npy}

        path_work_dir = Path(tempfile.mkdtemp())
        path_work_dir.mkdir(parents=True, exist_ok=True)

        stack_path = []
        for _f_name, _path in dict_f_name2path_train.items():
            _array_data_train: np.ndarray = np.load(_path)
            _f_path_test = dict_f_name2path_test[_f_name]
            _array_data_test = np.load(_f_path_test)
            _array_merged = np.concatenate([_array_data_train, _array_data_test])

            _path_array = path_work_dir / _f_name
            np.save(_path_array, _array_merged)
        
            stack_path.append(_path_array)
        # end for
        return stack_path


    logger.debug(f'Making the root directory: {path_experiment_root}')
    path_root_dir: Path = Path(path_experiment_root)
    path_root_dir.mkdir(parents=True, exist_ok=True)

    path_dir_mmd_distances = path_root_dir / sub_dir_name
    path_dir_mmd_distances.mkdir(parents=True, exist_ok=True)
    
    # listing up all files in the source directory.
    path_dir_training = Path(preprocessing_config.path_resource_output) / preprocessing_config.dir_name_train
    seq_files_source_npy_train = list(path_dir_training.rglob('*npy'))
    assert len(seq_files_source_npy_train) > 0, f'No npy files found in {path_dir_training}'

    path_dir_test = Path(preprocessing_config.path_resource_output) / preprocessing_config.dir_name_test
    seq_files_source_npy_test = list(path_dir_test.rglob('*npy'))
    assert len(seq_files_source_npy_test) > 0, f'No npy files found in {path_dir_training}'
    
    # mixing up train and test.
    seq_files_source_npy_merged = list(sorted(_merge_train_and_test_npy(seq_files_source_npy_train, seq_files_source_npy_test)))
    
    # calculating the length scale of gaussian kernel.
    kernel_obj_global = mmd_distance_analysis.calibrate_gaussian_kernel(seq_files_source_npy_merged, n_calibration_per_epoch=n_calibration_data)  # global: common kernel over all time-epochs.
    mmd_estimator_global = mmd_distance_analysis.get_mmd_estimator(kernel_obj_global)
    
    file_pairs = {(x, y) for x, y in itertools.combinations(seq_files_source_npy_merged, 2) if x != y}  # list of file path pairs.
    file_pairs = list(sorted(file_pairs, key=lambda x: (x[0].stem, x[1].stem)))

    _t_file_pair: ty.Tuple[Path, Path]
    
    stack_distance_obj = []
    for _t_file_pair in tqdm.tqdm(file_pairs):
        _mmd_distance = mmd_distance_analysis.compute_mmd_distance(mmd_estimator_global, _t_file_pair[0], _t_file_pair[1])
        _file_name_x = _t_file_pair[0].name
        _file_name_y = _t_file_pair[1].name

        _epoch_no_x = int(re.match(r"embedding_time_([0-9]+)\.npy", _file_name_x).group(1))
        _epoch_no_y = int(re.match(r"embedding_time_([0-9]+)\.npy", _file_name_y).group(1))
        
        _distance_obj = dict(
            file_name_x=_file_name_x,
            file_name_y=_file_name_y,
            epoch_no_x=_epoch_no_x,
            epoch_no_y=_epoch_no_y,
            mmd_distance=_mmd_distance.mmd.item(),
            variance=_mmd_distance.variance.item(),
            ratio=_mmd_distance.variance.item()
        )
        stack_distance_obj.append(_distance_obj)
    # end for

    path_mmd_distances = path_dir_mmd_distances / file_name_result_json
    with path_mmd_distances.open("w") as f:
        f.write(json.dumps(stack_distance_obj, ensure_ascii=False, indent=4))
    # end with

    logger.info(f"MMD distances are saved at {path_mmd_distances}")

    # deleting the tmp files.
    for _f_path in seq_files_source_npy_merged:
        _f_path.unlink()
    # end for

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    import dacite

    opt = ArgumentParser()
    opt.add_argument('-m', '--mode', choices=['compute', 'vis'])
    opt.add_argument('-c', '--path_config', 
                     type=str, 
                     required=True, 
                     help='Path to the config file. TOML format. \
                         When mode is "single_run", you put the a toml file ready-to-use. \
                             When the mode is "all", you put the toml file that you"ve described the base path.')
    opt.add_argument('--sub_dir_name', required=False, default="mmd_distances")
    __args = opt.parse_args()


    # load the config file.
    path_config = Path(__args.path_config)
    assert path_config.exists(), f'Not found: {path_config}'

    __config_obj = toml.loads(path_config.open().read())


    logger.info("---- Begin of the script ----")
    assert "PreprocessingOutputConfig" in __config_obj, f"PreprocessingOutputConfig is not found in {path_config}"
    __preprocessing_config = dacite.from_dict(data_class=PreprocessingOutputConfig, data=__config_obj['PreprocessingOutputConfig'])
    
    # extracting the output directory
    assert "ExecutionConfig" in __config_obj
    __ExecutionConfig_obj = __config_obj["ExecutionConfig"]
    assert "base" in __ExecutionConfig_obj
    __ExecutionConfig_base = __ExecutionConfig_obj["base"]
    __path_experiment_root = Path(__ExecutionConfig_base["path_experiment_root"])

    __sub_dir_name = __args.sub_dir_name

    if __args.mode == 'compute':
        execute_all_pairwise_combination(
            __path_experiment_root,
            __preprocessing_config,
            sub_dir_name=__sub_dir_name)
    elif __args.mode == "vis":
        visualisation_main(
            path_experiment_root=__path_experiment_root,
            config_obj=__config_obj,
            sub_dir_name=__sub_dir_name)
    else:
        raise ValueError(f"No mode named {__args.mode}")
    # end if

    logger.info("---- End of the script ----")