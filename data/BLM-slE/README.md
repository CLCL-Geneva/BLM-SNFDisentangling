This dataset contains two subsets: ALT-ATL, and ATL-ALT corresponding to the type of alternation in the input sequence, and in the answer set.

The csv files in this repository have the same structure, and the header of each file provides information about each column (separated by tabs):

* `ID` -- an id for the instance
* `Sent_(1:7)` -- the sentences for the input sequence 
* `Answer_(1:6)` -- the candidate answers
* `Answer_value_(1:6)` -- the truth value of each candidate answer w.r.t. the problem (is it the correct answer or not) 
* `Answer_label_(1:6)` -- the label of each candidate answer (explained in the paper)
* `Sent_template_(1:6)` -- the template used to generate the corresponding sentence
* `Answer_template_(1:6)` -- the template used to generate the corresponding answer

The data contains 3 subsets:

* `type I` data has minimal lexical variation, 
* `type II` has one word (noun/verb) different across the sentences, 
* `type III` data has maximal lexical variations. 

Train/test splits are included for each subset in the "datasets" subdirectories.

The details are in the paper:

[BLM-s/lE: A structured dataset of English spray-load verb alternations for testing generalization in LLMs] to appear at EMNLP 2023
