The csv files in this repository have the same structure, and the header of each file provides information about each column (separated by commas):

* `ID` -- an id for the instance
* `Sent_(1:7)` -- the sentences for the input sequence 
* `Answer_(1:6)` -- the candidate answers
* `Answer_value_(1:6)` -- the truth value of each candidate answer w.r.t. the problem (is it the correct answer or not) 
* `Answer_label_(1:6)` -- the label of each candidate answer (explained in the paper)

The data contains 3 subsets:

* `type I` data has minimal lexical variation, 
* `type II` has one word (noun/verb) different across the sentences, 
* `type III` data has maximal lexical variations. 

Each subset contains 3 files, one for each of the main syntactic variations: main clause, relative clause, completive clause.

The details are in the paper:

[BLM-AgrF: A New French Benchmark to Investigate Generalization of Agreement in Neural Networks](https://aclanthology.org/2023.eacl-main.99/) 
