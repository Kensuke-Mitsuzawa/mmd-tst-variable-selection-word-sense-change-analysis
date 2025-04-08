"""The given file (npy) is concatenated all together over multiple years.
This file is not clear for handling. Hence, I split the file into multiples."""

import typing as ty
from pathlib import Path
import numpy as np
import pickle
import random
import json

import dataclasses

import logzero
logger = logzero.logger



@dataclasses.dataclass
class PreprocessingSourceConfig:
    path_embedding_file: str
    path_token_entry_file: str

    option_use_pos: ty.Optional[str] = None

    def __post_init__(self):
        if self.option_use_pos is None:
            self.option_use_pos = ''
        # end if


@dataclasses.dataclass
class PreprocessingOutputConfig:
    path_resource_output: str

    dir_name_train: str = 'train'
    dir_name_test: str = 'test'

    n_token_train: int = 2000
    n_token_test: int = 300



def main(source_config: PreprocessingSourceConfig, output_config: PreprocessingOutputConfig):
    if output_config.n_token_train != -1:
        n_token_used = output_config.n_token_train + output_config.n_token_test
    else:
        raise Exception()
        n_token_used = -1
    # end if

    # set random seed
    random.seed(0)

    path_given_npy = Path(source_config.path_embedding_file)
    path_token_entry = Path(source_config.path_token_entry_file)

    assert path_given_npy.exists(), f"Given file not found: {path_given_npy}"
    assert path_token_entry.exists(), f"Token entry file not found: {path_token_entry}"


    path_dir_output = Path(output_config.path_resource_output)
    path_dir_output.mkdir(parents=True, exist_ok=True)

    path_dir_out_train = path_dir_output / output_config.dir_name_train
    path_dir_out_test = path_dir_output / output_config.dir_name_test
    path_dir_out_train.mkdir(parents=True, exist_ok=True)
    path_dir_out_test.mkdir(parents=True, exist_ok=True)

    
    path_dir_out_train.mkdir(parents=True, exist_ok=True)
    path_dir_out_test.mkdir(parents=True, exist_ok=True)
    
    token_entry = pickle.load(open(path_token_entry, 'rb'))  # dict file
    logger.info(f'Count token of embeddings: {len(token_entry)}')
    
    # set tokens to use.
    if source_config.option_use_pos != '':
        token_entry_updated = {}
        token_original_new_relation = {}
        new_t_id = 0
        for original_t_id, token in token_entry.items():
            if source_config.option_use_pos in token:
                token_entry_updated[new_t_id] = token
                token_original_new_relation[original_t_id] = new_t_id
                new_t_id += 1
            # end if
        # end for
        logger.info(f'Using {len(token_entry)} tokens for the comparison tasks.')
    else:
        token_entry_updated = token_entry
        token_original_new_relation = token_entry
    # end if
    
    if n_token_used != -1:
        logger.info(f'Shrinking the vocabulary size to {n_token_used} tokens.')
        
        if len(token_entry_updated) > 0:
            token_source = token_entry_updated
        else:
            token_source = token_entry
        # end if
        
        token_ids_use = sorted(random.sample(range(len(token_source)), n_token_used))
        token_entry_updated_n_limit_train = {}
        token_entry_updated_n_limit_test = {}
        token_original_new_relation_n_limit_train = {}
        token_original_new_relation_n_limit_test = {}
        
        token_ids_use_train = token_ids_use[:output_config.n_token_train]
        token_ids_use_test = token_ids_use[output_config.n_token_train:(output_config.n_token_train + output_config.n_token_test)]
        
        assert len(token_ids_use_train) > 0, f'No token for the training data: {token_ids_use_train}'
        assert len(token_ids_use_test) > 0, f'No token for the test data: {token_ids_use_test}'

        # for the training data
        new_t_id = 0
        for original_t_id, token in token_source.items():
            if original_t_id in token_ids_use_train:
                token_original_new_relation_n_limit_train[original_t_id] = new_t_id
                token_entry_updated_n_limit_train[new_t_id] = token
                new_t_id += 1
            # end if
        # end for
        
        # for the test data
        new_t_id = 0
        for original_t_id, token in token_source.items():
            if original_t_id in token_ids_use_test:
                # token_entry_updated_n_limit[new_t_id] = token
                token_original_new_relation_n_limit_test[original_t_id] = new_t_id
                token_entry_updated_n_limit_test[new_t_id] = token
                new_t_id += 1
            # end if
        # end for
        
        logger.info(f'Using {n_token_used} tokens for the comparison tasks.')
        logger.info(f'Updated dict. has {len(token_entry_updated)} entries.')
        
        token_entry_updated_train = token_entry_updated_n_limit_train
        token_entry_updated_test = token_entry_updated_n_limit_test
        token_original_new_relation_train = token_original_new_relation_n_limit_train
        token_original_new_relation_test = token_original_new_relation_n_limit_test
    else:
        # token_entry_updated = {}
        raise Exception
    # end if
    
    # loading npy file.
    whole_embeddings_over_time = np.load(path_given_npy)
    logger.debug(f'Whole embeddings over time: {whole_embeddings_over_time.shape}')
    
    # cutting word embeddings per one time epoch.
    n_total_epochs = whole_embeddings_over_time.shape[0] / len(token_entry)
    logger.debug(f'Total epochs: {n_total_epochs}')
    assert n_total_epochs.is_integer(), f'Total epochs is not integer: {n_total_epochs}'
    for i_current_time_epoch in range(0, int(n_total_epochs)):
        _name_current_time_epoch = i_current_time_epoch + 1  # used for labeling of files.
        
        _row_start = i_current_time_epoch * len(token_entry)
        _row_end = _row_start + len(token_entry)
        try:
            _embeddings_in_range = whole_embeddings_over_time[_row_start:_row_end]
        except IndexError:
            breakpoint()
            logger.error(f'IndexError: {_row_start}:{_row_end}')
            break
        
        # sampling tokens 
        if n_token_used != -1:
            try:
                _embeddings_in_range = _embeddings_in_range[token_ids_use]
            except IndexError:
                breakpoint()
        # end if
        logger.info(f'The total embedding size in range i={_name_current_time_epoch}: {_embeddings_in_range.shape}')
        
        # splitting into the training and test data
        _embeddings_in_range_train = _embeddings_in_range[:output_config.n_token_train]
        _embeddings_in_range_test = _embeddings_in_range[output_config.n_token_train:(output_config.n_token_train + output_config.n_token_test)]

        logger.info(f'Embedding Array for train: {_embeddings_in_range_train.shape}')
        logger.info(f'Embedding Array for test: {_embeddings_in_range_test.shape}')

        # saving the train
        _path_out_embedding_time_train = path_dir_out_train / f'embedding_time_{_name_current_time_epoch}.npy'
        np.save(_path_out_embedding_time_train, _embeddings_in_range_train)
        
        # saving the test
        _path_out_embedding_time_test = path_dir_out_test / f'embedding_time_{_name_current_time_epoch}.npy'
        np.save(_path_out_embedding_time_test, _embeddings_in_range_test)
        
        logger.debug(f'Saved: {_path_out_embedding_time_train} and {_path_out_embedding_time_test}')
    # end for
    
    # saving the updated token entry.
    if n_token_used != -1:
        # pickle dump, just for keeping consistency.
        path_out_token_entry_train = path_dir_out_train / 'train_updated_token_entry.pkl'
        path_out_token_entry_test = path_dir_out_test / 'test_updated_token_entry.pkl'
        pickle.dump(token_entry_updated_train, open(path_out_token_entry_train, 'wb'))
        pickle.dump(token_entry_updated_test, open(path_out_token_entry_test, 'wb'))
        logger.info(f'Saved: {path_out_token_entry_train}, {path_out_token_entry_test}')
        
        # I prefer json, to be honest.
        path_out_token_entry_train = path_dir_out_train / 'train_updated_token_entry.json'
        path_out_token_entry_test = path_dir_out_test / 'test_updated_token_entry.json'
        with path_out_token_entry_train.open('w') as f:
            f.write(json.dumps(token_entry_updated_train, ensure_ascii=False, indent=4))
        # end with
        with path_out_token_entry_test.open('w') as f:
            f.write(json.dumps(token_entry_updated_test, ensure_ascii=False, indent=4))
        # end with
        logger.info(f'Saved: {path_out_token_entry_train}, {path_out_token_entry_test}')    
    # end if

    # double-check if there are number of files as expected.
    n_files_train = len(list(path_dir_out_train.glob('embedding_time_*.npy')))
    assert n_files_train == int(n_total_epochs), f'Number of files is not as expected: {n_files_train}'
    n_files_test = len(list(path_dir_out_test.glob('embedding_time_*.npy')))
    assert n_files_test == int(n_total_epochs), f'Number of files is not as expected: {n_files_test}'

    
if __name__ == '__main__':
    import argparse
    import dacite
    import toml

    _parser = argparse.ArgumentParser(description='Preprocessing of word embeddings.')
    _parser.add_argument('-c', '--config', type=str, required=True)

    _args = _parser.parse_args()
    _path_config = Path(_args.config)
    assert _path_config.exists(), f'Config file not found: {_path_config}'
    _config = toml.load(_path_config)

    assert 'PreprocessingSourceConfig' in _config, f'Config file is not valid: {_path_config}'
    assert 'PreprocessingOutputConfig' in _config, f'Config file is not valid: {_path_config}'

    source_config = dacite.from_dict(
        data_class=PreprocessingSourceConfig,
        data=_config['PreprocessingSourceConfig'],
    )
    output_config = dacite.from_dict(
        data_class=PreprocessingOutputConfig,
        data=_config['PreprocessingOutputConfig'],
    )

    main(
        source_config=source_config,
        output_config=output_config,
    ) 