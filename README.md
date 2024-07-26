# Blackbird Language Matrices

## Overview 

The current reported success of machine learning architectures is based on computationally expensive algorithms and prohibitively large amounts of data that are available for only a few, non-representative languages. Moreover, despite these large training data requirements, current architectures still are not able to understand new sentences or derive the meaning of a word never seen before. 

To reach better, possibly human-like, abilities in neural networks’ abstraction and generalisation, we need to develop tasks and data that help us understand their current generalisation abilities and help us train them to more complex and compositional skills.

Inspired by computational methods on vision, we develop a new linguistic task, to learn more disentangled linguistic representations that reflect the underlying linguistic rules of grammar. The solution of the tasks requires identifying the underlying rules that generate compositional datasets (like Raven’s progressive matrices), but for language. We call them Blackbird’s Language Matrices (BLMs).

### Publications

[Blackbird's language matrices (BLMs): a new benchmark to investigate disentangled generalisation in neural networks](https://arxiv.org/abs/2205.10866) describes the project, and the first BLM dataset generated within this paradigm.

[Blackbird language matrices (BLM), a new task for rule-like generalization in neural networks: Motivations and Formal Specifications](https://arxiv.org/abs/2306.11444) provides details about the motivations and formal specifications of BLM's.

[BLM-AgrF: A New French Benchmark to Investigate Generalization of Agreement in Neural Networks](https://aclanthology.org/2023.eacl-main.99/) describes the BLM_AgrF dataset -- subject-verb agreement in French.

[Grammatical information in BERT sentence embeddings as two-dimensional arrays](https://aclanthology.org/2023.repl4nlp-1.3/) describes the impact of reshaping BERT sentence embeddings to 2D arrays on detecting subject-verb agreement information.


## Data

### BLM-AgrF

This is a dataset for subject-verb agreement in French. It is described in the [arXiv paper](https://arxiv.org/abs/2205.10866), and there are more details in an EACL 2023 paper (link will be included when available). It consists of 3 subsets:
* `type I` with minimal lexical variation
* `type II` with one word different for each sentence in a problem instance
* `type III` with each sentence in a problem instance lexically different from the others

The data can be found under data/BLM-AgrF/

## Code

The current code contains scripts to process the data, baselines for testing embeddings for this task, and several VAE-based architectures to test the impact of reshaping sentence embeddings as 2D arrays. It can produce and use sentence embeddings from pretrained models of BERT, RoBERTa and Electra.

### Data processing

* `make_train_test_data.py` produces train/test splits from the provided csv files for each type I/II/III subset. It can be called without arguments -- the data directory is the data directory included in the repository. If this is changed, it should be provided: make_train_test_data.py --data_path <data_path>.
  
When running the main code (run_experiments.sh), if it hasn't been done before, the system will extract sentence embeddings from the given transformer (BERT, RoBERTa, Electra). The pre-trained models used are hardcoded (in the embeddings.py module):

* BERT => bert-base-multilingual-cased, 
* RoBERTa => xml-roberta-base,
* Electra => google/electra-base-discriminator.

The necessary directories will be created under the type* subdirs in the data directory. (this takes a few hours, depending on the machine).

### Baselines

The baseline

* `FFNN` -- a 3 layer FFNN
* `CNN` -- a 3 layer CNN

### VAE architectures

* `encoder-decoder`: encodes a sequence of sentences, decodes the answer
  * with sentence embeddings as 2D arrays, input sequence is a stack of 2D arrays (vaes/VAE.py)
  * with sentence embeddings as 1D arrays, input sequence is a stack of 1D arrays (vaes/VAE_1DxSeq.py)
  * with sentence embeddings as 1D arrays, input sequence is a 1D array (concatenated sentence representations) (vaes/VAE_1D.py)

* `two level`: a sentence level VAE encodes individual sentences, a task level uses the sentence representations from the latent of the sentence level and reconstructs the input sequence, and decodes the answer to the task

* `two level sparsified`: same as the two level system, but the encoder on the sentence level is sparsified, such that one region of the sentence contributes to only one latent unit.
   
### Running experiments

To run experiments, use run_xperiments.py. Main arguments:

* `data_dir`: the top of the directory where the data is located. Default data/BLM-AgrF/
* `transformer` : bert, roberta, electra. Default = bert
* `baseline_sys` : the baseline system: ffnn, cnn, cnn_seq
* `sys`: the VAE-based system to use
* `train_perc`: how much data to use for training. Could be given as percentage or actual numbers, e.g.: "1.0 1.0 1.0" will use all the available data for each type, "2073 2073 2073" will use 2073 instances for training for each type (this is the max available for type I). The values need not be the same.
* `epochs`: the number of epochs to train. When the number of training examples is small, a higher number of epochs usually works better. When using a large number of training instances (for type II and type III subsets), one epoch will take a long time to run.
* `lr`: the learning rate. Default = 1e-3
* `n_exps`: number of experiments to run
* ...

The script can also take a json configuration file from which it will read the provided arguments. Whatever is not specified will have the default values.

The results of the experiments (P, R, F, Acc, error counts) are exported to a csv file in the results subdir.

## Dependencies

The requirements.txt contains the list of necessary Python libraries.

## Code contributors

Vivi Nastase and Maria A. Rodriguez.
