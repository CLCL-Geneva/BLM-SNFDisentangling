The csv files in this repository have the same structure:

* `columns 0:6` -- the 7 sentences constituting the input sequence
* `columns 7:12` -- the 6 sentences constituting the candidate answer set
* `columns 13:18` -- boolean values indicating the value of each candidate answer sentence
* `columns 19:24` -- the type of error in each candidate answer sentence

The data contains 3 subsets:

* `type I` data has minimal lexical variation, 
* `type II` has one word (noun/verb) different across the sentences, 
* `type III` data has maximal lexical variations. 

Each subset contains 3 files, one for each of the main syntactic variations: main clause, relative clause, completive clause.

The details are in the paper:

BLM-AgrF: A New French Benchmark to Investigate Generalization of Agreement in Neural Networks
