# Blackbird Language Matrices

## Overview 

The current reported success of machine learning architectures is based on computationally expensive algorithms and prohibitively large amounts of data that are available for only a few, non-representative languages. Moreover, despite these large training data requirements, current architectures still are not able to understand new sentences or derive the meaning of a word never seen before. 

To reach better, possibly human-like, abilities in neural networks’ abstraction and generalisation, we need to develop tasks and data that help us understand their current generalisation abilities and help us train them to more complex and compositional skills.

Inspired by computational methods on vision, we develop a new linguistic task, to learn more disentangled linguistic representations that reflect the underlying linguistic rules of grammar. The solution of the tasks requires identifying the underlying rules that generate compositional datasets (like Raven’s progressive matrices), but for language. We call them Blackbird’s Language Matrices (BLMs).

[This paper](https://arxiv.org/abs/2205.10866) describes the project, and the first BLM dataset generated within this paradigm.


## Data

### BLM-AgrF

This is a dataset for subject-verb agreement in French. It is described in an EACL 2023 paper (link will be included when available). It consists of 3 subsets:
* `type I` with minimal lexical variation
* `type II` with one word different for each sentence in a problem instance
* `type III` with each sentence in a problem instance lexically different from the others

Further details about the data in the paper.


## Code

The current code contains scripts to process the data, and baselines for testing BERT embeddings for this task:

### Data processing

* `make_train_test_data.py` produces train/test splits from the provided csv files for each type I/II/III subset. It can be called without arguments -- the data directory is the data directory included in the repository. If this is changed, it should be provided: make_train_test_data.py --data_path <data_path>.
* when running the main code, if it hasn't been done before, the system will extract sentence embeddings from the given transformer (bert or flaubert). The pre-trained models used are hardcoded (in the embeddings.py module). For BERT the model is bert-base-multilingual-cased, and for FlauBERT flaubert/flaubert_base_uncased. The necessary directories will be created under the type* subdirs in the data directory. (this takes a few hours, depending on the machine).

### Baselines

The baseline

* `FFNN` 

### Running experiments

To run experiments, call main.py. Main arguments:

* `data_dir`: the top of the directory where the data is located. Default data/BLM-AgrF/
* `transformer` : bert or flaubert. Default = bert
* `baseline_sys` : the baseline system: ffnn or cnn
* `train_perc`: how much data to use for training. Could be given as percentage or actual numbers, e.g.: "1.0 1.0 1.0" will use all the available data for each type, "2073 2073 2073" will use 2073 instances for training for each type (this is the max available for type I). The values need not be the same.
* `epochs`: the number of epochs to train. When the number of training examples is small, a higher number of epochs usually works better. When using a large number of training instances (for type II and type III subsets), one epoch will take a long time to run.
* `n_exps`: number of experiments to run

The results of the experiments (P, R, F, Acc, error counts) are exported to a csv file in the results subdir.

## Dependencies

The requirements.txt contains the list of necessary Python libraries.

## Code contributors

Vivi Nastase and Maria A. Rodriguez.
