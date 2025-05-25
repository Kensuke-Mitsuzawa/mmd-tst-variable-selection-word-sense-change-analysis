"""Making visualisations of multiple detection result"""

import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
from pathlib import Path
import toml
import json
import typing as ty
import tqdm
import copy
import pickle
from scipy.spatial.distance import cosine
import itertools

import seaborn as sns

import dataclasses

import pandas as pd

import logzero
logger = logzero.logger


try:
    import openpyxl
except ImportError as e:
    logger.error(f"openpyxl not found. Please install it. {e}")
    raise e
# end try

try:
    import japanize_matplotlib
except ImportError as e:
    logger.error(f"japanize-matplotlib not found. Please install it. {e}")
    raise e
# end try

try:
    import tslearn
    from tslearn.metrics import dtw
except ImportError as e:
    logger.error(f"tslearn not found. Please install it. {e}")
    raise e
# end try

# -------------------------------------------------
# matplotlib configs
import matplotlib.pyplot as plt
from word_sense_change_analysis import visualisation_header


# TimeEpochLabelStart = 2003
# TimeEpochLabelEnd = 2020

class TimeEpochConfiguration(object):
    def __init__(self,
                 year_start: int,
                 year_end: int,
                 year_range: int) -> None:
        self.year_start: int = year_start
        self.year_end: int = year_end
        self.year_range: int = year_range
    
    def get_year_label(self, time_epoch_index: int) -> int:
        """Getting the year label."""
        return self.year_start + (time_epoch_index - 1) * self.year_range
    

# -------------------------------------------------


@dataclasses.dataclass
class EmbeddingArraySet:
    embedding_array_x: np.ndarray
    embedding_array_y: np.ndarray
    dict_vocab_entry: ty.Dict[int, str]

    def __post_init__(self):
        assert self.embedding_array_x.shape[0] == self.embedding_array_y.shape[0], f"Shape mismatch: {self.embedding_array_x.shape[0]} != {self.embedding_array_y.shape[0]}"
        assert len(self.dict_vocab_entry) == self.embedding_array_x.shape[0], f"Shape mismatch: {len(self.dict_vocab_entry)} != {self.embedding_array_x.shape[1]}"


@dataclasses.dataclass
class OnePairDetectionResult:
    pair_key: ty.Set[int]
    
    epoch_no_x: int
    epoch_no_y: int
    
    detection_approach: str
    
    weights: np.ndarray
    variables: ty.List
    p_value: float
    
    path_result_dir: Path

    embeddings_train: EmbeddingArraySet
    embeddings_test: ty.Optional[EmbeddingArraySet]
    embedding_full: ty.Optional[EmbeddingArraySet] = None
    
    def __str__(self) -> str:
        return 'Pair of epochs: {}'.format(self.pair_key)

    def __repr__(self) -> str:
        return self.__str__()


def __extract_detection_one_pair(path_result_dir: Path,
                                 dict_epoch2embedding_train: ty.Dict[int, np.ndarray],
                                 dict_epoch2embedding_test: ty.Dict[int, np.ndarray],
                                 dict_vocab_entry_train: ty.Dict[int, str],
                                 dict_vocab_entry_test: ty.Dict[int, str],
                                 detection_approach: str = 'interpretable_mmd-algorithm_one.json',
                                 dir_prefix_name: str = 'embedding_time_',
                                 is_use_full_vocabulary: bool = False,
                                 dict_epoch2embedding_full: ty.Optional[ty.Dict[int, np.ndarray]] = None,
                                 dict_vocab_entry_full: ty.Optional[ty.Dict[int, str]] = None,
                                 skip_epoch_index: ty.Optional[ty.List[int]] = None
                                 ) -> ty.Optional[OnePairDetectionResult]:
    """Loading various data source, packing all into an object."""
    name_directory = path_result_dir.name
    assert name_directory.startswith(dir_prefix_name), f"Directory name must start with {dir_prefix_name}: {name_directory}"
    name_directory_without_prefix = name_directory.replace(dir_prefix_name, '')
    epoch_no_x, epoch_no_y = name_directory_without_prefix.split('_')

    if skip_epoch_index is not None:
        if int(epoch_no_x) in skip_epoch_index or int(epoch_no_y) in skip_epoch_index:
            logger.info(f"Skipping the epoch pair: {epoch_no_x} {epoch_no_y}")
            return None
        # end if
    # end if
    
    path_detection_file_json = path_result_dir / 'detection_output' / detection_approach
    
    if not path_detection_file_json.exists():
        return None
    # end if
    
    with path_detection_file_json.open() as f:
        detection_obj = json.loads(f.read())
    # end with
    
    assert 'detection_result' in detection_obj, f"Detection result not found in {path_detection_file_json}"
    assert 'weights' in detection_obj['detection_result'], f"Detection result not found in {path_detection_file_json}"
    assert 'variables' in detection_obj['detection_result'], f"Detection result not found in {path_detection_file_json}"
    assert 'p_value' in detection_obj['detection_result'], f"Detection result not found in {path_detection_file_json}"
    
    array_weights = np.array(detection_obj['detection_result']['weights'])
    variables = detection_obj['detection_result']['variables']
    p_vals = detection_obj['detection_result']['p_value']

    int_epoch_no_x = int(epoch_no_x)
    int_epoch_no_y = int(epoch_no_y)    

    emb_set_train = EmbeddingArraySet(
        embedding_array_x=dict_epoch2embedding_train[int_epoch_no_x], 
        embedding_array_y=dict_epoch2embedding_train[int_epoch_no_y],
        dict_vocab_entry=dict_vocab_entry_train)
    emb_set_test = EmbeddingArraySet(
        embedding_array_x=dict_epoch2embedding_test[int_epoch_no_x], 
        embedding_array_y=dict_epoch2embedding_test[int_epoch_no_y],
        dict_vocab_entry=dict_vocab_entry_test)    

    if is_use_full_vocabulary:
        assert dict_epoch2embedding_full is not None, f"Full vocabulary not found."
        assert dict_vocab_entry_full is not None, f"Full vocabulary not found."
        emb_set_full = EmbeddingArraySet(
            embedding_array_x=dict_epoch2embedding_full[int_epoch_no_x],
            embedding_array_y=dict_epoch2embedding_full[int_epoch_no_y],
            dict_vocab_entry=dict_vocab_entry_full)
    else:
        emb_set_full = None
    # end if

    
    one_pair_detection_result = OnePairDetectionResult(
        pair_key=({int_epoch_no_x, int_epoch_no_y}),
        epoch_no_x=int_epoch_no_x,
        epoch_no_y=int_epoch_no_y,
        detection_approach=detection_approach,
        weights=array_weights,
        variables=variables,
        p_value=p_vals,
        path_result_dir=path_result_dir,
        embeddings_train=emb_set_train,
        embeddings_test=emb_set_test,
        embedding_full=emb_set_full)
    # 
    
    return one_pair_detection_result
    
    
def __visualise_pval_heatmap(seq_stack_detection_results: ty.List[OnePairDetectionResult], 
                             path_figure_dir: Path,
                             year_start: int,
                             year_end: int,
                             year_range: int,
                             is_render_binary_ho_rejection: bool = False):
    year_convertor = TimeEpochConfiguration(
        year_start=year_start,
        year_end=year_end,
        year_range=year_range
    )
    
    seq_obj_p_value_matrix = [
        {
            'x': year_convertor.get_year_label(_obj.epoch_no_x),
            'y': year_convertor.get_year_label(_obj.epoch_no_y),
            'p_value': _obj.p_value
        } 
        for _obj in seq_stack_detection_results if _obj is not None]
    if is_render_binary_ho_rejection:
        # updating the p-values into binary flag.
        for _obj in seq_obj_p_value_matrix:
            if _obj['p_value'] < 0.05:
                _obj['p_value'] = 0.0
            else:
                _obj['p_value'] = 1.0
            # end if
        # end for
    # end if
    
    df_p_vals = pd.DataFrame(seq_obj_p_value_matrix)
    
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_p_vals.pivot(index='x', columns='y', values='p_value'), fmt=".4f", linewidths=.5, ax=ax)
    if is_render_binary_ho_rejection:
        _path_save_p_val_heatmap = path_figure_dir / 'heatmap_p_values_h0_rejection.png'
        ax.set_title(f'P-value heatmap with binary flag (rejection at 0.05)')
    else:
        _path_save_p_val_heatmap = path_figure_dir / 'heatmap_p_values.png'
        ax.set_title(f'P-value heatmap')
    # end if
    
    ax.set_xlabel('')
    ax.set_ylabel('')

    logger.debug(f'Saved the heatmap to {_path_save_p_val_heatmap}')
    f.savefig(_path_save_p_val_heatmap.as_posix(), bbox_inches='tight')
    
    
def __visualise_variable_count(seq_stack_detection_results: ty.List[OnePairDetectionResult], 
                               path_figure_dir: Path,
                               year_start: int,
                               year_end: int,
                               year_range: int,
                               is_render_binary_ho_rejection: bool = False):
    year_convertor = TimeEpochConfiguration(
        year_start=year_start,
        year_end=year_end,
        year_range=year_range
    )

    seq_obj_p_value_matrix = [
        {
            'x': year_convertor.get_year_label(_obj.epoch_no_x),
            'y': year_convertor.get_year_label(_obj.epoch_no_y),
            'n_variables': len(_obj.variables),
            'p_value': _obj.p_value
        } 
        for _obj in seq_stack_detection_results if _obj is not None]
    if is_render_binary_ho_rejection:
        # updating the count by the p-values
        for _obj in seq_obj_p_value_matrix:
            if _obj['p_value'] < 0.05:
                pass
            else:
                _obj['n_variables'] = 0.0
            # end if
        # end for
    # end if    

    df_p_vals = pd.DataFrame(seq_obj_p_value_matrix)
    
    f, ax = plt.subplots(figsize=(9, 6))
    df_pivot = df_p_vals.pivot(index='x', columns='y', values='n_variables')

    sns.heatmap(df_pivot, annot=True, linewidths=.5, ax=ax, annot_kws={"fontsize": 12})

    if is_render_binary_ho_rejection:
        _path_save = path_figure_dir / 'heatmap_variable_count_p_val_filtering.png'
        ax.set_title(f'Variable-count heatmap (Zero count if p-value < 0.05)')
    else:
        _path_save = path_figure_dir / 'heatmap_variable_count.png'
        ax.set_title(f'Variable-count heatmap')
    # end if
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    logger.debug(f'Saved the heatmap to {_path_save}')
    f.savefig(_path_save.as_posix(), bbox_inches='tight')


def __visualise_whole_pair_weights(seq_stack_detection_results: ty.List[OnePairDetectionResult], 
                                   path_figure_dir: Path,):
    """I want to visualise the whole pair of the weights."""
    pass


def __collect_json_files():
    """I want to collect all json files of the detection results.
    The detection result directories contain a lot of aux. files. Analysts (my collaborators) do not need such aux. files. 
    
    I need just a few fields,
    - epoch_no_x
    - epoch_no_y
    - weights array
    - variables
    - p-value
    """
    pass


def __collect_common_and_uncommon_variables(seq_stack_detection_results: ty.List[OnePairDetectionResult], 
                                            path_output_dir: Path):
    """Detection of common and uncommon variables over the whole time epochs."""
    seq_variables_set = [_obj.variables for _obj in seq_stack_detection_results if _obj is not None]
    common_variables = set.intersection(*[set(_l) for _l in seq_variables_set])
    logger.info(f'Detected common variables: {common_variables}')
    
    # I want to do (Set-Variable-Epoch).disjoin(common-variable)
    dict_epoch2disjoint_variables = {}
    for _detection_obj in seq_stack_detection_results:
        if _detection_obj is None:
            continue
        if _detection_obj.variables is None:
            continue
        # end if
        
        _variables_dis_joint = set(_detection_obj.variables) - common_variables
        _key_name = f'{_detection_obj.epoch_no_x}-{_detection_obj.epoch_no_y}'
        dict_epoch2disjoint_variables[_key_name] = list(_variables_dis_joint)
    # end for    

    import collections
    # count frequency of variable id
    dict_variable_id2count = collections.Counter()
    for seq_vars in seq_variables_set:
        for __var_id, __ in collections.Counter(seq_vars).items():
            dict_variable_id2count[__var_id] += 1
        # end for
    # end for
    top_k = 10
    for _var_id, _freq in dict_variable_id2count.most_common(top_k):
        logger.info(f'Variable-id={_var_id}. Frequency={_freq}. Percentage={_freq/len(seq_variables_set)}')
    # end for


class _VocabularyScoreTimeEpoch(ty.NamedTuple):
    time_epoch_x: int
    time_epoch_y: int
    score: float
    p_value: float
    
    def is_time_included(self, time_epoch: int) -> bool:
        return time_epoch in {self.time_epoch_x, self.time_epoch_y}
    # end def
# end class


class TimeEpochSpecificVocabulary(ty.NamedTuple):
    time_epoch: int
    vocabularies: ty.List[ty.Tuple[str, float]]
# end class


class ComputedVocabularyScore(ty.NamedTuple):
    map_token2score: ty.Dict[str, float]
    map_token2scores_epochs: ty.Dict[str, ty.List[_VocabularyScoreTimeEpoch]]
    time_epoch2tokens_specific: ty.List[TimeEpochSpecificVocabulary]



def compute_vocabulary_score(seq_stack_detection_results: ty.List[OnePairDetectionResult],
                             is_use_full_vocabulary: bool = False) -> ComputedVocabularyScore:
    """This function computes 3 scores.
    1. score for each vocabulary.
    2. score for each vocabulary over the time epochs.
    3. score for each vocabulary for each time epoch. I want to list up vocabularies that are specific to a certain time epoch.
    
    I want to compute a score (float) per vocabulary.
    The score is mean([distance-time-i-j, ...]), where distance-time-i-j is the cosine distance between time-i and time-j, I use only selected variables for this computation.
    """
    # -------------------------------------------------
    if is_use_full_vocabulary:
        # I use the full vocabulary.
        assert seq_stack_detection_results[0].embedding_full is not None, f"Full vocabulary not found."
        vocab_join = seq_stack_detection_results[0].embedding_full.dict_vocab_entry
    else:
        # I use vocabularies, train + test
        # joining two vocab dictionary together. Assigning a new id to test-vocabs.
        assert seq_stack_detection_results[0].embeddings_test is not None, f"Test vocabularies not found."
        vocab_train = seq_stack_detection_results[0].embeddings_train.dict_vocab_entry
        vocab_test = seq_stack_detection_results[0].embeddings_test.dict_vocab_entry
        vocab_join = copy.deepcopy(vocab_train)
        for __i_vocab in vocab_test.keys():
            assert max(vocab_train.keys()) + 1 + __i_vocab not  in vocab_train, f"Vocabulary id conflict: {max(vocab_train.keys()) + 1 + __i_vocab}"
            __new_voacb_id = max(vocab_train.keys()) + 1 + __i_vocab
            vocab_join[__new_voacb_id] = vocab_test[__i_vocab]
        # end for
        assert len(vocab_train) + len(vocab_test) == len(vocab_join), f"Vocabulary join failed: {len(vocab_train)} + {len(vocab_test)} != {len(vocab_join)}"
    # end if
    # -------------------------------------------------
    # collecting time epochs
    __set_time_epochs = []
    for __obj in seq_stack_detection_results:
        __set_time_epochs += __obj.pair_key
    # end for
    seq_time_epochs = sorted(list(set(__set_time_epochs)))
    # -------------------------------------------------

    map_token2score = {}
    map_token2scores_epochs = {}
    
    # -------------------------------------------------
    # computing 1. and 2., and 3.
    logger.info(f'Computing the score for vocabularies....')

    # # TODO
    # logger.error(f'DEBUG MODE. For now, I limit the number of vocabularies to 500.')
    # vocab_join = {_k: _v for _k, _v in vocab_join.items() if _k < 500}
    # computing the score for a vocabulary
    for __vocab_id, __vocab in tqdm.tqdm(vocab_join.items()):
        stack_score_token = []  # [( (int, int), float)]

        for __obj_time in seq_stack_detection_results:
            __variables_selected = __obj_time.variables  # list of selected list.

            # -------------------------------------------------
            # concatenating the embeddings
            if is_use_full_vocabulary:
                assert __obj_time.embedding_full is not None, f"Full vocabulary not found."
                __embedding_x = __obj_time.embedding_full.embedding_array_x
                __embedding_y = __obj_time.embedding_full.embedding_array_y
            else:
                __embedding_train_x = __obj_time.embeddings_train.embedding_array_x
                __embedding_train_y = __obj_time.embeddings_train.embedding_array_y
                
                assert __obj_time.embeddings_test is not None, f"Test embeddings not found."
                __embedding_test_x = __obj_time.embeddings_test.embedding_array_x
                __embedding_test_y = __obj_time.embeddings_test.embedding_array_y
                __embedding_x = np.concatenate([__embedding_train_x, __embedding_test_x], axis=0)  # (vocab-train+vocab-test) x dim
                __embedding_y = np.concatenate([__embedding_train_y, __embedding_test_y], axis=0)  # (vocab-train+vocab-test) x dim
                assert __embedding_x.shape[0] == __embedding_y.shape[0], f"Shape mismatch: {__embedding_x.shape[0]} != {__embedding_y.shape[0]}"                
            # end if

            # -------------------------------------------------
            # computing the cosine distance
            __sub_vector_x = __embedding_x[__vocab_id, __variables_selected]
            __sub_vector_y = __embedding_y[__vocab_id, __variables_selected]
            __cosine_distance = cosine(__sub_vector_x, __sub_vector_y)

            # saving the computed distance
            if __cosine_distance == 0:
                __cosine_distance = 0.0
            # end if
            assert isinstance(__cosine_distance, float), f"Type mismatch"
            
            __t_time_epochs = tuple(__obj_time.pair_key)

            __info_tuple_score = _VocabularyScoreTimeEpoch(
                __t_time_epochs[0], 
                __t_time_epochs[1], 
                __cosine_distance, 
                __obj_time.p_value)
            stack_score_token.append( __info_tuple_score )
        # end for

        # -------------------------------------------------
        # Core of computing the score
        # getting the avg score
        __avg_score = np.mean([__t.score for __t in stack_score_token])  # float
        # logger.debug(f'Vocabulary={__vocab}. Score={__avg_score}')
        
        # saving the score
        map_token2score[__vocab] = __avg_score
        map_token2scores_epochs[__vocab] = stack_score_token
    # end for
        
    # logger.info(f'The top-10 vocabularies with the highest score.')
    # for __vocab, __score in sorted(map_token2score.items(), key=lambda x: x[1], reverse=True)[:10]:
    #     logger.info(f'Vocabulary={__vocab}. Score={__score}')
    # end for
    
    # -------------------------------------------------
    # computing 3.
        
    # a binary flag if using time epochs where H0 is accepted.
    is_include_time_epoch_accept_h0 = False
    
    __seq_tuple_vocab_score = []
    for __vocab, __seq_tuple_scores in map_token2scores_epochs.items():
        for __time_epoch in seq_time_epochs:
            # collecting tuples where the time-epoch is included.
            __seq_tuple_scores_target = [__t for __t in __seq_tuple_scores if __t.is_time_included(__time_epoch)]
            
            if is_include_time_epoch_accept_h0:
                __score_mean_vocab = np.mean([__t.score for __t in __seq_tuple_scores_target])
            else:
                __score_mean_vocab = np.mean([__t.score for __t in __seq_tuple_scores_target if __t.p_value <= 0.05])
            # end if
            __seq_tuple_vocab_score.append( (__time_epoch, __vocab, __score_mean_vocab) )
        # end for
    # end for
    
    time_epoch2tokens_specific = []
    for __time_epoch, __g_obj in itertools.groupby(sorted(__seq_tuple_vocab_score, key=lambda x: x[0]), key=lambda x: x[0]):
        __seq_scores_computed = [(__t[1], __t[2]) for __t in list(__g_obj)]
        time_epoch2tokens_specific.append(TimeEpochSpecificVocabulary(__time_epoch, __seq_scores_computed))
    # end for
    
    return ComputedVocabularyScore(
        map_token2score=map_token2score, 
        map_token2scores_epochs=map_token2scores_epochs,
        time_epoch2tokens_specific=time_epoch2tokens_specific)


def __write_out_voacb_specific_time_epochs(computed_vocabulary_score: ComputedVocabularyScore, 
                                           path_output_excel: Path,
                                           year_start: int,
                                           year_end: int,
                                           year_range: int):
    """Writing out the computed vocabulary and its score for each time epoch.
    
    The excel book has 2 sheets.
    1. vocabulary list that has high scores over the time epochs.
    2. vocabulary list per time-epoch.
    """
    # sheet-1: vocabulary-score
    df_sheet_one = pd.DataFrame(computed_vocabulary_score.map_token2score.items(), columns=['Vocabulary', 'Score'])
    df_sheet_one.sort_values(by='Score', ascending=False, inplace=True)

    # sheet-2: vocabulary per time-epoch
    stack_dataframe = {}
    for __time_epoch_container in computed_vocabulary_score.time_epoch2tokens_specific:
        __label_year = TimeEpochConfiguration(
            year_start=year_start,
            year_end=year_end,
            year_range=year_range
        ).get_year_label(__time_epoch_container.time_epoch)
        __seq_labels = [f'{__t[0]} {__t[1]}' for __t in sorted(__time_epoch_container.vocabularies, key=lambda x: x[1], reverse=True)]
        stack_dataframe[__label_year] = __seq_labels
    # end for
    df_sheet_two = pd.DataFrame(stack_dataframe)

    with pd.ExcelWriter(path_output_excel) as writer:
        df_sheet_one.to_excel(writer, sheet_name='vocabulary-score')
        df_sheet_two.to_excel(writer, sheet_name='time-epoch-vocabulary')
    # end with
    logger.info(f'The vocabulary excel book is at {path_output_excel}')

    return 


def __visualise_target_vocabulary_over_time(computed_vocabulary_score: ComputedVocabularyScore, 
                                            path_figure_dir: Path, 
                                            target_words: ty.List[str],
                                            year_start: int,
                                            year_end: int,
                                            year_range: int):
    """plotting the top-n vocabularies over the time."""
    path_figure_dir.mkdir(parents=True, exist_ok=True)

    class _ExtractedScoreTuple(ty.NamedTuple):
        year: int
        vocab: str
        score: float
    # end class

    # -------------------------------------------------
    # I need year, score, and vocab.
    seq_extracted_target_vocab = []

    for __year_extraction_container in computed_vocabulary_score.time_epoch2tokens_specific:
        for __vocab_container in __year_extraction_container.vocabularies:
            if __vocab_container[0] in target_words:
                seq_extracted_target_vocab.append(_ExtractedScoreTuple(
                    __year_extraction_container.time_epoch, 
                    __vocab_container[0], 
                    __vocab_container[1]))
            # end if
        # end for
    # end for

    # -------------------------------------------------

    year_convertor = TimeEpochConfiguration(
        year_start=year_start,
        year_end=year_end,
        year_range=year_range
    )

    for __vocab, __g_obj in itertools.groupby(sorted(seq_extracted_target_vocab, key=lambda x: x.vocab), key=lambda x: x.vocab):
        __seq_score_container = sorted(list(__g_obj), key=lambda t: t.year)
        __seq_score_container = [{'year': year_convertor.get_year_label(__t.year), 'score': __t.score} for __t in __seq_score_container]

        # plotting
        __df_plot = pd.DataFrame(__seq_score_container)
        __f, __ax = plt.subplots(figsize=(9, 6))

        sns.lineplot(x='year', y='score', data=__df_plot, ax=__ax, marker='o', color='b')
        # Reset the x-axis labels
        __unique_time_epochs = list(sorted(__df_plot['year'].unique()))
        __plot_new_labels = [str(__i) for __i in __unique_time_epochs]
        plt.xticks(__unique_time_epochs, labels=__plot_new_labels, rotation=90)

        # Renaming the x and y labels.
        __ax.set_xlabel('Year')
        __ax.set_ylabel('Score')

        # setting the plot title
        __ax.set_title(f'Vocabulary={__vocab}')

        __f.savefig((path_figure_dir / f'vocab_{__vocab}.png').as_posix(), bbox_inches='tight')
        logger.info(f'Saved the plot to {path_figure_dir / f"vocab_{__vocab}.png"}')
    # end for


def compute_dtw_score_based_on_vocabulary_time_score(
        computed_vocabulary_score: ComputedVocabularyScore,
        target_keywords: ty.List[str]) -> ty.Dict[str, ty.List[ty.Tuple[str, float]]]:
    """Given a keyword, I want to know a list of words that have similar time-evolution patterns.

    Method:
        I use Dynamic Time Wrapping (DTW).
    
    Return:
        {keyword: [(vocabulary, DTW-score)]}
    """
    class _InnerTupleVocabScore(ty.NamedTuple):
        vocab: str
        score: float
        time_epoch: int
    # end def

    # making score sequence for all vocabularies.
    # I make a structure of [(vocabulary, score, time-epoch)]
    seq_vocab_score_flatten = [
        _InnerTupleVocabScore(__t[0], __t[1], __container_time.time_epoch) 
        for __container_time in computed_vocabulary_score.time_epoch2tokens_specific 
        for __t in __container_time.vocabularies]
    # grouping by the vocabulary, {vocab: []}
    __ = itertools.groupby(sorted(seq_vocab_score_flatten, key=lambda x: x.vocab), key=lambda x: x.vocab)
    dict_vocab2score: ty.Dict[str, ty.List[_InnerTupleVocabScore]] = {__vocab: list(__g_obj) for __vocab, __g_obj in __}

    # return object. keyword -> [(vocab, DTW)]
    dict_keyword2similar_vocabs = {}

    logger.info('Computing DTW score for each keyword...')
    for __keyword in tqdm.tqdm(target_keywords):
        assert isinstance(__keyword, str), f"Type mismatch: {__keyword}"
        if __keyword not in dict_vocab2score:
            logger.error(f"Keyword not found: {__keyword}")
            continue
            # assert __keyword in dict_vocab2score, f"Keyword not found: {__keyword}"
        # end if

        # set the key to the return dict
        dict_keyword2similar_vocabs[__keyword] = []

        _seq_scores_target_keyword = [__container.score for __container in sorted(dict_vocab2score[__keyword], key=lambda x: x.time_epoch)]
        for __vocab, __seq_score in dict_vocab2score.items():
            if __vocab == __keyword:
                continue
            # end if

            _seq_score_vocab = [__container.score for __container in sorted(dict_vocab2score[__vocab], key=lambda x: x.time_epoch)]

            assert len(_seq_scores_target_keyword) > 0 and len(_seq_score_vocab) > 0, f"Empty sequence: {__keyword} {__vocab}" 
            assert len(_seq_scores_target_keyword) == len(_seq_score_vocab), f"Length mismatch: {len(_seq_scores_target_keyword)} != {len(_seq_score_vocab)}"
            # computing DTW
            __dtw_score = dtw(_seq_scores_target_keyword, _seq_score_vocab)
            
            # adding the score to the return object
            dict_keyword2similar_vocabs[__keyword].append((__vocab, __dtw_score))
        # end for
    # end for
    logger.info('Done.')

    return dict_keyword2similar_vocabs


def visualise_compute_dtw_score_based_on_vocabulary_time_score(dict_keyword2similar_vocabs: ty.Dict[str, ty.List[ty.Tuple[str, float]]],
                                                               vocab_computed_score: ComputedVocabularyScore,
                                                               path_dir_figures: Path,
                                                               year_start: int,
                                                               year_end: int,
                                                               year_range: int,
                                                               name_excel_file: str = 'dtw_scores.xlsx',                                                               
                                                               top_n: int = 50,
                                                               pos_filter_included: ty.Tuple[str, ...] = ('名詞', '動詞', '形容詞')):
    """visualiastion of the function return `compute_dtw_score_based_on_vocabulary_time_score`.

    1. making the excel book.
    2. making the visualisation of each keyword.
    """
    path_dir_figures.mkdir(parents=True, exist_ok=True)

    # making the excel book
    __path_excel_file = path_dir_figures / name_excel_file
    with pd.ExcelWriter(__path_excel_file) as writer:
        for __keyword, __seq_vocabs in dict_keyword2similar_vocabs.items():
            __df = pd.DataFrame(__seq_vocabs, columns=['Vocabulary', 'DTW-score'])
            __df.sort_values(by='DTW-score', ascending=True, inplace=True)
            __df.to_excel(writer, sheet_name=f'{__keyword}')
        # end for
    # end with


    # -------------------------------------------------
    class _InnerTupleVocabScore(ty.NamedTuple):
        vocab: str
        score: float
        time_epoch: int
    # end def

    # making score sequence for all vocabularies.
    # I make a structure of [(vocabulary, score, time-epoch)]
    seq_vocab_score_flatten = [
        _InnerTupleVocabScore(__t[0], __t[1], __container_time.time_epoch) 
        for __container_time in vocab_computed_score.time_epoch2tokens_specific 
        for __t in __container_time.vocabularies]
    # grouping by the vocabulary, {vocab: []}
    __ = itertools.groupby(sorted(seq_vocab_score_flatten, key=lambda x: x.vocab), key=lambda x: x.vocab)
    dict_vocab2score: ty.Dict[str, ty.List[_InnerTupleVocabScore]] = {__vocab: list(__g_obj) for __vocab, __g_obj in __}

    # making the visualisation
    for __keyword, __seq_vocabs in dict_keyword2similar_vocabs.items():
        __path_dir_sub = path_dir_figures / __keyword
        __path_dir_sub.mkdir(parents=True, exist_ok=True)

        # filtering by the POS
        __seq_similar_dtw_keywords = [__t for __t in __seq_vocabs if __t[0] in pos_filter_included]
        # sorting by the DTW score
        __seq_similar_dtw_keywords = sorted(__seq_vocabs, key=lambda x: x[1], reverse=False)[:top_n]
        for __dtw_rank, __t_vocab_similar in enumerate(__seq_similar_dtw_keywords):
            assert isinstance(__t_vocab_similar, tuple), f"Type mismatch: {__t_vocab_similar}"
            assert isinstance(__t_vocab_similar[0], str), f"Type mismatch: {__t_vocab_similar[0]}"
            assert isinstance(__t_vocab_similar[1], float), f"Type mismatch: {__t_vocab_similar[1]}"

            __extracted_vocab: str = __t_vocab_similar[0]
            # gettting the time sequence of the score about the keyword
            __seq_vocab_scores = dict_vocab2score[__t_vocab_similar[0]]
            __seq_vocab_scores = sorted(__seq_vocab_scores, key=lambda x: x.time_epoch)
            __seq_vocab_scores = [
                {'year': TimeEpochConfiguration(
                    year_start=year_start,
                    year_end=year_end,
                    year_range=year_range
                ).get_year_label(__t.time_epoch), 'score': __t.score} 
            for __t in __seq_vocab_scores]
            __df_plot = pd.DataFrame(__seq_vocab_scores)

            # plotting
            __f, __ax = plt.subplots(figsize=(9, 6))

            sns.lineplot(x='year', y='score', data=__df_plot, ax=__ax, marker='o', color='b')
            # Reset the x-axis labels
            __unique_time_epochs = list(sorted(__df_plot['year'].unique()))
            __plot_new_labels = [str(__i) for __i in __unique_time_epochs]
            plt.xticks(__unique_time_epochs, labels=__plot_new_labels, rotation=90)

            # Renaming the x and y labels.
            __ax.set_xlabel('Year')
            __ax.set_ylabel('Score')

            # setting the plot title
            __ax.set_title(f'Keyword={__keyword} Vocabulary={__extracted_vocab} DTW={__t_vocab_similar[1]}')

            __file_name = f'rank-{__dtw_rank}_vocab-{__extracted_vocab}.png'
            __path_file = __path_dir_sub / __file_name.replace('/', '-')

            __f.savefig(__path_file.as_posix(), bbox_inches='tight')
            logger.info(f'Saved the plot to {__path_file}')
        # end for
    # end for


def main(path_config_toml: Path, 
         is_use_full_vocabulary: bool = False,
         dir_prefix_name: str = 'embedding_time_'):
    
    with path_config_toml.open() as f:
        _config_obj = toml.loads(f.read())
    # end with

    _config_preprocessed = _config_obj['PreprocessingOutputConfig']
    _config_execution = _config_obj['ExecutionConfig']
    _config_analysis = _config_obj['Analysis']

    # -------------------------------------------------
    # Analysis config
    path_figure_dir = Path(_config_analysis['path_analysis_output'])
    path_figure_dir.mkdir(parents=True, exist_ok=True)

    target_keywords: ty.List[str] = _config_analysis['target_keywords']

    time_label_start: int = _config_analysis['TimeEpochLabelStart']
    time_label_end: int = _config_analysis['TimeEpochLabelEnd']
    time_label_range: int = _config_analysis['TimeEpochLabelRange']

    skip_epoch_index: ty.List[int] = _config_analysis['skip_epoch_index']

    # -------------------------------------------------

    assert 'base' in _config_execution, f"Config file must have 'base' key: {path_config_toml}"
    assert 'path_experiment_root' in _config_execution['base'], f"Config file must have 'path_experiment_root' key: {path_config_toml}"
    
    path_dir_exp_root = Path(_config_execution['base']['path_experiment_root'])
    assert path_dir_exp_root.exists(), f"Experiment root directory not found: {path_dir_exp_root}"

    # -------------------------------------------------
    # loading vocabulary numpy array(s).
    __path_processed_output = Path(_config_preprocessed.get('path_resource_output'))
    if is_use_full_vocabulary:
        path_obj_source_full = __path_processed_output / 'full'
        dir_path_source_numpy_full = list(path_obj_source_full.rglob('*npy'))
        assert len(dir_path_source_numpy_full) > 0, f"Directory not found: {dir_path_source_numpy_full}"
    else:
        path_obj_source_full = None
    # end if

    path_obj_source_train = __path_processed_output / 'train'
    path_obj_source_test = __path_processed_output / 'test'
    # end if
    assert path_obj_source_train is not None, f"Config file must have 'data_setting' key: {path_config_toml}"
    assert path_obj_source_test is not None, f"Config file must have 'data_setting_test' key: {path_config_toml}"

    dir_path_source_numpy_train = list(path_obj_source_train.rglob('*npy'))
    dir_path_source_numpy_test = list(path_obj_source_test.rglob('*npy'))
    assert len(dir_path_source_numpy_train) > 0, f"Directory not found: {dir_path_source_numpy_train}"
    assert len(dir_path_source_numpy_test) > 0, f"Directory not found: {dir_path_source_numpy_test}"

    # {time-epoch-id: np.ndarray}
    dict_epoch2embedding_train = {}
    dict_epoch2embedding_test = {}    

    for _path in dir_path_source_numpy_train:
        _epoch_no = int(_path.stem.split('_')[-1])
        if _epoch_no in skip_epoch_index:
            logger.debug(f"Skipping: {_epoch_no}")
            continue
        # end if

        dict_epoch2embedding_train[_epoch_no] = np.load(_path)
    # end for
    for _path in dir_path_source_numpy_test:
        _epoch_no = int(_path.stem.split('_')[-1])
        if _epoch_no in skip_epoch_index:
            logger.debug(f"Skipping: {_epoch_no}")
            continue
        # end if

        dict_epoch2embedding_test[_epoch_no] = np.load(_path)
    # end for
    logger.info(f'Loaded {len(dict_epoch2embedding_train)} train embeddings.')
    logger.info(f'Loaded {len(dict_epoch2embedding_test)} test embeddings.')

    if is_use_full_vocabulary:
        dict_epoch2embedding_full = {}
        for _path in dir_path_source_numpy_full:
            _epoch_no = int(_path.stem.split('_')[-1])
            dict_epoch2embedding_full[_epoch_no] = np.load(_path)
        # end for
        logger.info(f'Loaded {len(dict_epoch2embedding_full)} full embeddings.')
    else:
        dict_epoch2embedding_full = None
    # end if
    # -------------------------------------------------

    # loading dictionary file (int -> str).
    __path_vocab_entry_train = path_obj_source_train / 'train_updated_token_entry.json'
    assert __path_vocab_entry_train.exists(), f"Vocabulary file not found: {__path_vocab_entry_train}"
    
    __path_vocab_entry_test = path_obj_source_test / 'test_updated_token_entry.json'
    assert __path_vocab_entry_test.exists(), f"Vocabulary file not found: {__path_vocab_entry_test}"
    
    with __path_vocab_entry_train.open() as f:
        dict_vocab_entry_train = {int(_k): _v for _k, _v in json.loads(f.read()).items()}
    # end with
    
    with __path_vocab_entry_test.open() as f:
        dict_vocab_entry_test = {int(_k): _v for _k, _v in json.loads(f.read()).items()}
    # end with

    if is_use_full_vocabulary:
        assert path_obj_source_full is not None, f"Config file must have 'data_setting_full' key: {path_config_toml}"
        __path_vocab_entry_full = path_obj_source_full / 'full_updated_token_entry.json'
        assert __path_vocab_entry_full.exists(), f"Vocabulary file not found: {__path_vocab_entry_full}"
        with __path_vocab_entry_full.open() as f:
            dict_vocab_entry_full = {int(_k): _v for _k, _v in json.loads(f.read()).items()}
        # end with
    else:
        dict_vocab_entry_full = {}
    # end if
    
    assert len(set(dict_vocab_entry_train.values()).intersection(dict_vocab_entry_test.values())) == 0, f"Vocabulary conflict"
    # -------------------------------------------------
    # loading the original embedding file (single file)
    # logger.debug('Loading the original embedding file...')
    # dict_config_data_source = config_obj.get('data_source')
    # assert dict_config_data_source is not None, f"Config file must have 'data_source' key: {path_config_toml}"
    # __path_original_embedding = dict_config_data_source.get('path_array_source_original')
    # path_original_embedding = Path(__path_original_embedding)
    # embedding_vector_original = np.load(path_original_embedding)
    
    # __path_dictionary_original = dict_config_data_source.get('path_dictionary_source_original')
    # dict_embedding_vocabulary_original = pickle.load(open(__path_dictionary_original, 'rb'))
    
    # logger.debug(f'Original embedding file: {path_original_embedding}')
    # -------------------------------------------------
    # 
    # listing up directories
    list_dir_exp = [p for p in path_dir_exp_root.iterdir() if p.is_dir()]
    
    seq_stack_detection_results = []
    for path_dir_name in tqdm.tqdm(list_dir_exp):
        if not path_dir_name.name.startswith(dir_prefix_name):
            logger.error(f"Skipping: {path_dir_name}")
            continue
        # end if
        logger.debug(f"Processing: {path_dir_name}")
        # making visualisation
        __obj_extraction = __extract_detection_one_pair(path_dir_name, 
                                                        dict_epoch2embedding_train, 
                                                        dict_epoch2embedding_test,
                                                        dict_vocab_entry_train,
                                                        dict_vocab_entry_test,
                                                        is_use_full_vocabulary=is_use_full_vocabulary,
                                                        dict_epoch2embedding_full=dict_epoch2embedding_full,
                                                        dict_vocab_entry_full=dict_vocab_entry_full,
                                                        skip_epoch_index=skip_epoch_index,)
        if __obj_extraction is not None:
            seq_stack_detection_results.append(__obj_extraction)
        # end if
    # end for
    
    # removing duplicated Pair. X is priority. So, reset all values into X <- Y.
    keys_numbers_processed = []
    for _obj in seq_stack_detection_results:
        if _obj.pair_key in keys_numbers_processed:
            logger.debug(f'The pair already exists. I skip it.')
        # end if
        if _obj.epoch_no_x < _obj.epoch_no_y:
            # revese the number of (X, Y)
            __epoch_no_x_new = _obj.epoch_no_y
            __epoch_no_y_new = _obj.epoch_no_x
            _obj.epoch_no_x = __epoch_no_x_new
            _obj.epoch_no_y = __epoch_no_y_new
        # end if
        keys_numbers_processed.append(_obj.pair_key)
    # end for
    # -------------------------------------------------
    # procedure: scoring vocabularies
    
    vocab_computed_score = compute_vocabulary_score(seq_stack_detection_results, is_use_full_vocabulary=is_use_full_vocabulary)
    # writing out vocabulary scores to excel sheetbook.
    __write_out_voacb_specific_time_epochs(computed_vocabulary_score=vocab_computed_score, 
                                           path_output_excel=path_figure_dir / 'vocabulary_score.xlsx',
                                           year_start=time_label_start,
                                           year_end=time_label_end,
                                           year_range=time_label_range)

    if len(target_keywords) > 0:
        # visualisation of vobucalries over the time.
        __path_dir_time_epoch = path_figure_dir / 'vocab-time-epoch'
        __path_dir_time_epoch.mkdir(parents=True, exist_ok=True)
        __visualise_target_vocabulary_over_time(
            computed_vocabulary_score=vocab_computed_score, 
            path_figure_dir=__path_dir_time_epoch, 
            target_words=target_keywords,
            year_start=time_label_start,
            year_end=time_label_end,
            year_range=time_label_range)
        
        # computing the DTW scores.
        dict_dtw_score_keywords = compute_dtw_score_based_on_vocabulary_time_score(
            computed_vocabulary_score=vocab_computed_score, 
            target_keywords=target_keywords)
        visualise_compute_dtw_score_based_on_vocabulary_time_score(
            dict_keyword2similar_vocabs=dict_dtw_score_keywords,
            vocab_computed_score=vocab_computed_score,
            path_dir_figures=path_figure_dir / 'dtw_score',
            name_excel_file='dtw_scores.xlsx',
            year_start=time_label_start,
            year_end=time_label_end,
            year_range=time_label_range,)
    else:
        logger.info(f'Target keywords not found. Skipping the DTW score computation.')
    # end if

    # -------------------------------------------------
    # making visualisation
    
    # visualisation of heatmap of p-values.
    # making a matrix of p-values.
    __visualise_pval_heatmap(
        seq_stack_detection_results=seq_stack_detection_results,
        path_figure_dir=path_figure_dir,
        year_start=time_label_start,
        year_end=time_label_end,
        year_range=time_label_range)
    # binary flag < 0.05 or not.
    __visualise_pval_heatmap(
        seq_stack_detection_results, 
        path_figure_dir, 
        is_render_binary_ho_rejection=True,
        year_start=time_label_start,
        year_end=time_label_end,
        year_range=time_label_range)
    
    # visualisation of heatmap of variable count.
    __visualise_variable_count(
        seq_stack_detection_results=seq_stack_detection_results, 
        path_figure_dir=path_figure_dir,
        year_start=time_label_start,
        year_end=time_label_end,
        year_range=time_label_range)
    __visualise_variable_count(
        seq_stack_detection_results=seq_stack_detection_results, 
        path_figure_dir=path_figure_dir, 
        is_render_binary_ho_rejection=True,
        year_start=time_label_start,
        year_end=time_label_end,
        year_range=time_label_range)
    # -------------------------------------------------
    # non-completed codes below
    # detection of common and uncommon variables
    path_analysis_output_dir = path_figure_dir / 'variables-common-uncommon'
    __collect_common_and_uncommon_variables(seq_stack_detection_results, path_analysis_output_dir)
    # 
    # __collect_json_files()


def _test():
    # the common config toml file as the config for the interface.py
    PATH_CONFIG = '/home/kmitsuzawa/codes/mmd-tst-variable-selection-word-sense-change-analysis/configs/mainichi_corpus.toml'
    assert Path(PATH_CONFIG).exists(), f"Config file not found: {PATH_CONFIG}"
    
    main(
        path_config_toml=Path(PATH_CONFIG), 
        is_use_full_vocabulary=False)


if __name__ == '__main__':
    # the common config toml file as the config for the interface.py
    # PATH_CONFIG = '/home/kmitsuzawa/codes/mmd-tst-variable-selection-word-sense-change-analysis/configs/mainichi_corpus.toml'
    from argparse import ArgumentParser
    __opt = ArgumentParser()
    __opt.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    __opt.add_argument('-f', '--full', action='store_true', help='Use full vocabulary.')
    _parser = __opt.parse_args()
    assert Path(_parser.config).exists(), f"Config file not found: {_parser.config}"
    
    main(
        path_config_toml=Path(_parser.config), 
        is_use_full_vocabulary=True if _parser.full else False)
