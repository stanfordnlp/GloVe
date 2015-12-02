## GloVe: Global Vectors for Word Representation

<em>frog</em> nearest neighbors | Litoria             |  Leptodactylidae | Rana | Eleutherodactylus
-------------------------|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
<li> frogs <li> toad <li> litoria <li> leptodactylidae <li> rana <li> lizard <li> eleutherodactylus | ![](http://nlp.stanford.edu/projects/glove/images/litoria.jpg)  |  ![](http://nlp.stanford.edu/projects/glove/images/leptodactylidae.jpg) |  ![](http://nlp.stanford.edu/projects/glove/images/rana.jpg) |  ![](http://nlp.stanford.edu/projects/glove/images/eleutherodactylus.jpg)

We provide an implementation of the GloVe model for learning word representations. Please see the [project page](http://nlp.stanford.edu/projects/glove/) for more information.

man -> woman             |  city -> zip | comparative -> superlative
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![](http://nlp.stanford.edu/projects/glove/images/man_woman_small.jpg)  |   ![](http://nlp.stanford.edu/projects/glove/images/city_zip_small.jpg) |  ![](http://nlp.stanford.edu/projects/glove/images/comparative_superlative_small.jpg)

## Download pre-trained word vectors
Pre-trained word vectors are made available under the <a href="http://opendatacommons.org/licenses/pddl/">Public Domain Dedication
and License</a>
<div class="entry">
<ul style="padding-left:0px; margin-top:0px; margin-bottom:0px">
  <li> <a href="http://dumps.wikimedia.org/enwiki/20140102/">Wikipedia 2014</a> + <a href="https://catalog.ldc.upenn.edu/LDC2011T07">Gigaword 5</a> (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, &amp; 300d vectors, 822 MB download): <a href="http://nlp.stanford.edu/data/glove.6B.zip">glove.6B.zip</a> </li>
  <li> Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): <a href="http://nlp.stanford.edu/data/glove.42B.300d.zip">glove.42B.300d.zip</a> </li>
  <li> Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): <a href="http://nlp.stanford.edu/data/glove.840B.300d.zip">glove.840B.300d.zip</a> </li>
  <li> Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, &amp; 200d vectors, 1.42 GB download): <a href="http://nlp.stanford.edu/data/glove.twitter.27B.zip">glove.twitter.27B.zip</a> Ruby <a href="preprocess-twitter.rb">script</a> for preprocessing Twitter data </li>
</ul>
</div>

## Train word vectors on a new corpus

    $ git clone http://github.com/stanfordnlp/glove
    $ cd glove && make
    $ ./demo.sh

The demo.sh scipt downloads a small corpus, consisting of the first 100M characters of Wikipedia. It collects unigram counts, constructs and shuffles cooccurrence data, and trains a simple version of the GloVe model. It also runs a word analogy evaluation script in python. Continue reading for further usage details and instructions for how to run on your own corpus.

### Package Contents
This package includes four main tools:
#### 1) vocab_count
Constructs unigram counts from a corpus, and optionally thresholds the resulting vocabulary based on total vocabulary size or minimum frequency count. This file should already consist of whitespace-separated tokens. Use something like the Stanford Tokenizer (http://nlp.stanford.edu/software/tokenizer.shtml) first on raw text.
#### 2) cooccur
Constructs word-word cooccurrence statistics from a corpus. The user should supply a vocabulary file, as produced by 'vocab_count', and may specify a variety of parameters, as described by running './build/cooccur'.
#### 3) shuffle
Shuffles the binary file of cooccurrence statistics produced by 'cooccur'. For large files, the file is automatically split into chunks, each of which is shuffled and stored on disk before being merged and shuffled togther. The user may specify a number of parameters, as described by running './build/shuffle'.
#### 4) glove
Train the GloVe model on the specified cooccurrence data, which typically will be the output of the 'shuffle' tool. The user should supply a vocabulary file, as given by 'vocab_count', and may specify a number of other parameters, which are described by running './build/glove'.

### License
All work contained in this package is licensed under the Apache License, Version 2.0. See the include LICENSE file.
