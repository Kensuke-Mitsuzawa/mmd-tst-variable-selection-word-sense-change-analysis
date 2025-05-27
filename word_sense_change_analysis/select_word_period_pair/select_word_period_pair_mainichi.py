import toml
import json
import typing as ty
from pathlib import Path

import numpy as np
import torch


"""An independent script to extract top-10 words associated with the selected variables.
This script is for CCOHA corpus word embedding vectors.
"""

# ----------------------------------------------------------------------------------------
# UPDATE THIS SECTION
# loading a pair of time epoch: use coha dataset
path_config = Path("/home/kmitsuzawa/codes/mmd-tst-variable-selection-word-sense-change-analysis/configs/mainichi_corpus.toml")
assert path_config.exists()

seq_t_pair_analysis_target_year = [
    (2008, 2009),
    (2008, 2010),
    (2009, 2010),    
    (2010, 2011),    
    (2011, 2012),    
]  # use the index number, not the year label.

n_top_ranking = 30
# ----------------------------------------------------------------------------------------

# %%
# loading the variable selection result
with path_config.open('r') as f:
    config_obj = toml.load(f)

# %%
path_embedding_dir = Path(config_obj["PreprocessingOutputConfig"]["path_resource_output"])
assert path_embedding_dir.exists()
path_embedding_dir_full = path_embedding_dir / 'full'
assert path_embedding_dir_full.exists()

# %%
path_result_dir = Path(config_obj['ExecutionConfig']['base']['path_experiment_root'])
assert path_result_dir.exists()

# %%

def convert_into_index_mainichi(year_label: int) -> int:
    _year = ((year_label) - 2003) / 1
    return int(_year) + 1
# end def


def main(t_pair_analysis_target_year: ty.Tuple[int, int], n_top: int = 10):

    t_pair_analysis_target = (convert_into_index_mainichi(t_pair_analysis_target_year[0]), convert_into_index_mainichi(t_pair_analysis_target_year[1]))

    print(t_pair_analysis_target)

    detection_file_name = f"embedding_time_{t_pair_analysis_target[0]}_embedding_time_{t_pair_analysis_target[1]}"
    path_detection_result_file = path_result_dir / detection_file_name

    print(path_detection_result_file)
    if path_detection_result_file.exists() is False:
        detection_file_name = f"embedding_time_{t_pair_analysis_target[1]}_embedding_time_{t_pair_analysis_target[0]}"
        path_detection_result_file = path_result_dir / detection_file_name
        assert path_detection_result_file.exists(), f"No file {path_detection_result_file}"
    # end if


    path_detection_result = path_detection_result_file / "detection_output" / "interpretable_mmd-algorithm_one.json"
    assert path_detection_result.exists()

    print(path_detection_result)
    with path_detection_result.open('r') as f:
        detection_result_obj = json.loads(f.read())
    # end with


    # paring the json file
    assert "detection_result" in detection_result_obj

    seq_selected_variables = detection_result_obj['detection_result']['variables']

    print(f'N(variable) = {len(seq_selected_variables)}')

    # %%
    # loading the variable selection result
    embedding_file_x = path_embedding_dir_full / f'embedding_time_{t_pair_analysis_target[0]}.npy'
    embedding_file_y = path_embedding_dir_full / f'embedding_time_{t_pair_analysis_target[1]}.npy'

    path_d_dictionary = path_embedding_dir_full / 'full_updated_token_entry.json'

    array_x = np.load(embedding_file_x)
    array_y = np.load(embedding_file_y)

    d_dictionary = {int(_k): _v for _k, _v in json.load(path_d_dictionary.open()).items()}

    # %%
    # select variables and compute distances on the vectors
    array_x_selected = array_x[:, seq_selected_variables]
    array_y_selected = array_y[:, seq_selected_variables]

    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import cosine

    assert len(array_x_selected) == len(array_y_selected)

    def func_get_ranking_wasserstein(array_x_selected, array_y_selected):
        seq_stack_score: ty.List[ty.Tuple[str, float]] = []
        for _ind, _t_sample_xy in enumerate(zip(array_x_selected, array_y_selected)):
            _sample_x, _sample_y = _t_sample_xy
            _d_value = wasserstein_distance(_sample_x, _sample_y)
            _token = d_dictionary[_ind]
            seq_stack_score.append( (_token, _d_value) )
        # end for
        return sorted(seq_stack_score, key=lambda t: t[1], reverse=True)


    def func_get_ranking_cosine(array_x_selected, array_y_selected):
        seq_stack_score: ty.List[ty.Tuple[str, float]] = []
        for _ind, _t_sample_xy in enumerate(zip(array_x_selected, array_y_selected)):
            _sample_x, _sample_y = _t_sample_xy
            _d_value = cosine(_sample_x, _sample_y)
            _token = d_dictionary[_ind]
            seq_stack_score.append( (_token, _d_value) )
        # end for
        return sorted(seq_stack_score, key=lambda t: t[1], reverse=True)
    # end def

    ranking_wass = func_get_ranking_wasserstein(array_x_selected, array_y_selected)
    ranking_cosine = func_get_ranking_cosine(array_x_selected, array_y_selected)


    return dict(
        wassterstein=ranking_wass[:n_top],
        cosine=ranking_cosine[:n_top]
    )
# end def

stack_obj = []
for _t_pair in seq_t_pair_analysis_target_year:
    _d_ranking = main(_t_pair, n_top_ranking)

    _t_pair_analysis_target = (convert_into_index_mainichi(_t_pair[0]), convert_into_index_mainichi(_t_pair[1]))

    _d_record = dict(
        year_label_x=_t_pair[0],
        year_label_y=_t_pair[1],
        index_x=_t_pair_analysis_target[0],
        index_y=_t_pair_analysis_target[1],
        ranking=_d_ranking
    )
    stack_obj.append(_d_record)
# end for


path_dir_result = path_result_dir / "word_ranking_pairwise_period"
path_dir_result.mkdir(parents=True, exist_ok=True)

path_output_json = path_dir_result / f"result_{n_top_ranking}.json"

with path_output_json.open("w") as f:
    f.write(json.dumps(stack_obj, ensure_ascii=False, indent=4))
# end with

print(path_output_json)