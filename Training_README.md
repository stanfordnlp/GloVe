## Overview

Five new vectors were trained. Four of these vectors were trained on Gigaword and updated Wikipedia corpus and the fifth was trained on [Dolma](https://allenai.github.io/dolma/). This document includes the scripts used to train these five vectors. 

## Data Download and Preprocessing 

The Wikipedia portion of the corpora was downloaded from the [7/22/2024 dump](https://dumps.wikimedia.org/enwiki/20240720/enwiki-20240720-pages-meta-current.xml.bz2). The text was then extracted using [Wikiextractor](https://github.com/attardi/wikiextractor) with the command 
``` 
python -m wikiextractor.WikiExtractor enwiki-20220620-pages-meta-current.xml --no-templates --output output_dir/
```
The fifth edition of [Gigaword](https://catalog.ldc.upenn.edu/LDC2011T07) was used. Dolma v1.6 was used and 5% of Common Crawl, 40% of C4, 100% of Reddit, and 100% of Project Gutenberg were used. 

For tokenization, version 4.4.1 of Stanford's Stanza tokenizer was used. We used the script 

```java edu.stanford.nlp.process.PTBTokenizer -preserveLines -lowerCase -options 'untokenizable=allKeep'```

When combining the Wikipedia and Gigaword data, Gigaword was added twice to balance the increase in Wikipedia data from the previous training corpus.

## Vector Training 

Here is the bash script used for training the vectors Wiki Giga vectors 50/100/200/300 dimension and Dolma 300 dimension
``` 
CORPUS=<wikigiga, dolma>
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=vectors.txt
OVERFLOW_FILE=overflow
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=20
VECTOR_SIZE=<50, 100, 200, 300>
MAX_ITER=<50,100>
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=10
SEED=<123, 2024>
ALPHA=0.75
ETA=<0.05, 0.075>
$BUILDDIR/vocab_count  -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -overflow-file $OVERFLOW_FILE -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -seed $SEED -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
$BUILDDIR/glove -eta $ETA -alpha $ALPHA -save-file $SAVE_FILE -threads $NUM_THREADS -input-file
    $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY 
    -vocab-file $VOCAB_FILE -verbose $VERBOSE -seed $SEED -checkpoint-every 10

```
### Vocabulary:
For dolma, a maximum vocabulary of 1.2M was used by passing the -max-vocab flag 

### Vector Training:
For Wiki Giga 50 dimension, random seed 123 and learning rate of 0.075 was used. For all other vectors, random seed 2024 and learning rate of 0.05 was used



