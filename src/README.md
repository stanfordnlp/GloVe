### Package Contents

To train your own GloVe vectors, first you'll need to prepare your corpus as a single text file with all words separated by a single space. If your corpus has multiple documents, simply concatenate documents together with a single space. If your documents are particularly short, it's possible that padding the gap between documents with e.g. 5 "dummy" words will produce better vectors. Once you create your corpus, you can train GloVe vectors using the following 4 tools. An example is included in `demo.sh`, which you can modify as necessary.

This four main tools in this package are: 
#### 1) vocab_count
Constructs unigram counts from a corpus, and optionally thresholds the resulting vocabulary based on total vocabulary size or minimum frequency count. This file should already consist of whitespace-separated tokens. Use something like the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) first on raw text.
#### 2) cooccur
Constructs word-word cooccurrence statistics from a corpus. The user should supply a vocabulary file, as produced by `vocab_count`, and may specify a variety of parameters, as described by running `./build/cooccur`.
#### 3) shuffle
Shuffles the binary file of cooccurrence statistics produced by `cooccur`. For large files, the file is automatically split into chunks, each of which is shuffled and stored on disk before being merged and shuffled togther. The user may specify a number of parameters, as described by running `./build/shuffle`.
#### 4) glove
Train the GloVe model on the specified cooccurrence data, which typically will be the output of the `shuffle` tool. The user should supply a vocabulary file, as given by `vocab_count`, and may specify a number of other parameters, which are described by running `./build/glove`.
