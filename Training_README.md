#Overview

Five new vectors were trained. Four of these vectors were trained on Gigaword and updated Wikipedia corpora and the fifth was traind on [Dolma](https://allenai.github.io/dolma/). This document includes the scripts used to train these five vectors. 

## Data Download and Preprocessing 

The Wikipedia portion of the corpora was downloaded from the [7/22/2024 dump](https://dumps.wikimedia.org/enwiki/20240720/enwiki-20240720-pages-meta-current.xml.bz2). The fifth edition of [Gigaword](https://catalog.ldc.upenn.edu/LDC2011T07) was used. TODO DOLMA WHERE TO GET 

For tokenization, version 4.4.1 of Stanford's CoreNLP tokenizer was used. We used the script 
    $java edu.stanford.nlp.process.PTBTokenizer -preserveLines -lowerCase -options 'untokenizable=allKeep'

When combining the Wikipedia and Gigaword data, Gigaword was added twice to balance the increase in Wikipedia data from the previous training corpus.

## Vector Training 

Here is the bash script used for training the vectors
$  CORPUS=<wikigiga, dolma>
$  VOCAB_FILE=<wikigiga_vocab.txt, dolma_vocab.txt>
$  COOCCURRENCE_FILE=vocab.txt
$  COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
$  BUILDDIR=build
$  SAVE_FILE=vectors.txt
$  VERBOSE=2
$  MEMORY=4.0
$  VOCAB_MIN_COUNT=20
$  VECTOR_SIZE=<50, 100, 200, 300>
$  MAX_ITER=<50,100>
$  WINDOW_SIZE=10
$  BINARY=2
$  NUM_THREADS=8
$  X_MAX=10
$  SEED=<123, 2024>
$  ALPHA=0.75
$  ETA=<0.05, 0.075>
$  $BUILDDIR/vocab_count  -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$  $BUILDDIR/cooccur -memory $MEMORY -overflow-file $OVERFLOW_FILE -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$  $BUILDDIR/shuffle -memory $MEMORY -seed $SEED -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

### Vocabulary:
For dolma, a maximum vocabulary of 1.2M was used by passing the -max-vocab flag 

### Cooccurrence building and shuffling :




