# Tests for ensuring randomization is being controlled

make

if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi

# Global constants
CORPUS=text8
VERBOSE=2
BUILDDIR=build
MEMORY=4.0
VOCAB_MIN_COUNT=20

# Re-used files
VOCAB_FILE=vocab.test.txt
COOCCURRENCE_FILE=cooccurrence.test.bin
COOCCURRENCE_SHUF_FILE=cooccurrence_shuf.test.bin

# Make vocab
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

# Make Coocurrences
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size 5 < $CORPUS > $COOCCURRENCE_FILE

# Shuffle Coocurrences
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

# Keep track of failure
num_failed=0

check_exit() {
  eval $2
  failed=$(( $1 != $? ))
  num_failed=$(( $num_failed + $failed ))
  if [[ $failed -eq 0 ]]; then
    echo PASSED
  else
    echo FAILED
  fi
}

# Test control of random seed in shuffle
printf "\n\n--- TEST SET: Control of random seed in shuffle\n"
TEST_FILE=cooc_shuf.test.bin

printf "\n- TEST: Default seed should be 1\n"
$BUILDDIR/shuffle -memory $MEMORY -verbose 0 -seed 1 < $COOCCURRENCE_FILE > $TEST_FILE
check_exit 0 "cmp --quiet $COOCCURRENCE_SHUF_FILE $TEST_FILE"

printf "\n- TEST: Changing the seed should change the shuffle\n"
$BUILDDIR/shuffle -memory $MEMORY -verbose 0 -seed 2 < $COOCCURRENCE_FILE > $TEST_FILE
check_exit 1 "cmp --quiet $COOCCURRENCE_SHUF_FILE $TEST_FILE"

rm $TEST_FILE  # Clean up
# ---

# Control randomization in GloVe
printf "\n\n--- TEST SET: Control of random seed in glove\n"
# Note "-threads" must equal 1 for these to pass, since order in which results come back from individual threads is uncontrolled
BASE_PREFIX=base_vectors
TEST_PREFIX=test_vectors

printf "\n- TEST: Default seed should be 1\n"
$BUILDDIR/glove -save-file $BASE_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0
$BUILDDIR/glove -save-file $TEST_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -seed 1
check_exit 0 "cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

printf "\n- TEST: Changing seed should change the learned vectors\n"
$BUILDDIR/glove -save-file $TEST_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -seed 2
check_exit 1 "cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

printf "\n- TEST: Should be able to save/load initial parameters\n"
$BUILDDIR/glove -save-file $BASE_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -save-init-param 1
$BUILDDIR/glove -save-file $TEST_PREFIX -threads 1 -input-file $COOCCURRENCE_SHUF_FILE -iter 3 -vector-size 10 -binary 1 -vocab-file $VOCAB_FILE -verbose 0 -save-init-param 1 -load-init-param 1 -init-param-file "$BASE_PREFIX.000.bin"
check_exit 0 "cmp --quiet $BASE_PREFIX.000.bin $TEST_PREFIX.000.bin && cmp --quiet $BASE_PREFIX.bin $TEST_PREFIX.bin"

rm "$BASE_PREFIX.000.bin" "$TEST_PREFIX.000.bin" "$BASE_PREFIX.bin" "$TEST_PREFIX.bin" # Clean up

# ----

rm $VOCAB_FILE $COOCCURRENCE_FILE $COOCCURRENCE_SHUF_FILE

echo
echo SUMMARY:
if [[ $num_failed -gt 0 ]]; then
  echo $num_failed tests failed.
  exit 1
else
  echo All tests passed.
  exit 0
fi
