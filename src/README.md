### Package Contents

To train your own GloVe vectors, first you'll need to prepare your corpus as a single text file with all words separated by one or more spaces or tabs. If your corpus has multiple documents, the documents (only) should be separated by new line characters. Cooccurrence contexts for words do not extend past newline characters. Once you create your corpus, you can train GloVe vectors using the following 4 tools. An example is included in `demo.sh`, which you can modify as necessary.

The four main tools in this package are:

#### 1) vocab_count
This tool requires an input corpus that should already consist of whitespace-separated tokens. Use something like the [Stanford Tokenizer](https://nlp.stanford.edu/software/tokenizer.html) first on raw text. From the corpus, it constructs unigram counts from a corpus, and optionally thresholds the resulting vocabulary based on total vocabulary size or minimum frequency count.

#### 2) cooccur
Constructs word-word cooccurrence statistics from a corpus. The user should supply a vocabulary file, as produced by `vocab_count`, and may specify a variety of parameters, as described by running `./build/cooccur`.

#### 3) shuffle
Shuffles the binary file of cooccurrence statistics produced by `cooccur`. For large files, the file is automatically split into chunks, each of which is shuffled and stored on disk before being merged and shuffled together. The user may specify a number of parameters, as described by running `./build/shuffle`.

#### 4) glove
Train the GloVe model on the specified cooccurrence data, which typically will be the output of the `shuffle` tool. The user should supply a vocabulary file, as given by `vocab_count`, and may specify a number of other parameters, which are described by running `./build/glove`.
