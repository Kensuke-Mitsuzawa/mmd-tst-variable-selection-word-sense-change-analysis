# README

## Procedure

1. Setting a word embedding file (numpy) and corresponding picle file of token entries. See the next section for the details.
2. Running `./word_sense_change_analysis/preprocessing.py` with a config file.
3. Running `./interface.py`
4. Running `./word_sense_change_analysis/visualisations.py`

### Visualisation

Before visualisation, do `pip install seaborn -U`. Seaborn 0.12.2 fails to plot heatmaps.

```
python ./word_sense_change_analysis/visualisations.py -c <path-to-config.toml>
```




# Required File Format

## Preprocessing

The project requires two types of files: "word entry file (pickle file)" and "word embedding file (numpy `npy`)".

The pickle file saves a dictionary object of `{entry-id (int): entry-name (str)}`. For example, `{1: "NLP}"`. The "entry-id" represents an index of a word embedding.

The word embedding file saves a array of which shape is `(entry-size * time-period-size, dimension)`. For example, suppose that `7228` word entries and `18` time-periods, and the dimension size of the word embedding is 100. Then, the array size is `(130104, 100)`.

